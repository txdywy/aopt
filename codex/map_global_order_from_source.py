"""Map a proven *global* resource order onto a rewritten target DAG.

Unlike per-engine order transfer, this script makes one Kahn topological
extension over every target engine before deriving the individual resource
orders.  Consequently every added capacity edge points forward in that same
global order, so the five projected engine orders are acyclic by construction.

Typical use maps the embedded default 987-cycle schedule into a structural
candidate.  ``SOURCE_ORDERS`` is a comma-separated list of per-engine JSON
files carrying both ``cycles`` and the exact resource ``order``; they can be
materialized from the 987 ``.order`` file with ``search_resource_orders``.
``SOURCE_FULL`` is the corresponding all-operation schedule JSON, including
the non-resource debug operations omitted from those five order files.
The current environment selects the target structural candidate.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import heapq
import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, op_key, real_tail_ops
from problem import SLOT_LIMITS


def freeze(value: object) -> object:
    if isinstance(value, list):
        return tuple(freeze(item) for item in value)
    return value


def operation_ids(ops: list[kernel._Op]) -> list[tuple[tuple[object, ...], int]]:
    occurrences: Counter[tuple[object, ...]] = Counter()
    result = []
    for op in ops:
        key = op_key(op)
        result.append((key, occurrences[key]))
        occurrences[key] += 1
    return result


def cross_key(key: tuple[object, ...]) -> tuple[object, ...] | None:
    """Remove implementation-engine detail for scalar/vector substitutions."""
    if len(key) < 5 or key[1] == "identity":
        return None
    tag = str(key[2])
    if tag.endswith("_vector"):
        tag = tag[:-7]
    return (key[1], tag, key[3], key[4])


def build_ops() -> list[kernel._Op]:
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        if not hasattr(builder, "dag_ops"):
            raise
    return real_tail_ops(builder.dag_ops)


def median(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def main() -> None:
    raw_sources = [
        Path(value)
        for value in os.environ["SOURCE_ORDERS"].split(",")
        if value
    ]
    if not raw_sources:
        raise ValueError("SOURCE_ORDERS must name at least one order JSON")

    full_source = json.loads(Path(os.environ["SOURCE_FULL"]).read_text())
    raw_full_cycles = full_source["cycles"]
    source_cycles = (
        {int(index): int(cycle) for index, cycle in raw_full_cycles.items()}
        if isinstance(raw_full_cycles, dict)
        else {index: int(cycle) for index, cycle in enumerate(raw_full_cycles)}
    )
    source_rank: dict[int, int] = {}
    source_orders: dict[str, list[int]] = {}
    for path in raw_sources:
        payload = json.loads(path.read_text())
        engine = payload["engine"]
        cycles = {int(index): int(cycle) for index, cycle in payload["cycles"].items()}
        order = [int(index) for index in payload["order"]]
        if set(order) != set(cycles):
            raise ValueError(f"source order/cycles disagree: {path}")
        if engine in source_orders:
            raise ValueError(f"duplicate source engine order: {engine}")
        source_orders[engine] = order
        for rank, index in enumerate(order):
            if index not in source_cycles:
                raise ValueError(f"source operation missing from SOURCE_FULL: {index}")
            if index in source_rank:
                raise ValueError(f"duplicate source resource operation {index}")
            source_rank[index] = rank
        if any(index not in cycles for index in order):
            raise AssertionError(engine)
    source_span = max(source_cycles.values()) + 1
    # A per-engine rank is sufficient for a projected calendar, but it loses
    # the cross-engine tie decisions that made the incumbent schedule fast.
    # Keep the total source issue order available as an optional teacher.  The
    # target still goes through a Kahn extension, so a rewritten dependency can
    # override this preference without ever creating an illegal cycle.
    source_global_rank = {
        index: rank
        for rank, (index, _cycle) in enumerate(
            sorted(source_cycles.items(), key=lambda item: (item[1], item[0]))
        )
    }
    use_global_tie = bool(int(os.environ.get("GLOBAL_SOURCE_TIE", "0")))

    # The source is the module-default graph.  Do not call configure_target
    # until after it has been constructed, even though target knobs are
    # already present in this process environment.
    kernel.SCHEDULE_EXACT_CYCLES = None
    source_ops = build_ops()
    if set(source_cycles) != set(range(len(source_ops))):
        raise ValueError(
            f"source order covers {len(source_cycles)} ops, default graph has "
            f"{len(source_ops)}"
        )
    source_ids = operation_ids(source_ops)
    source_index_by_id = {identity: index for index, identity in enumerate(source_ids)}

    source_cross_cycles: dict[tuple[object, ...], list[int]] = defaultdict(list)
    source_tag_cycles: dict[tuple[object, ...], list[int]] = defaultdict(list)
    source_round_cycles: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for index, op in enumerate(source_ops):
        key = op_key(op)
        phase = source_cycles[index]
        if (compact := cross_key(key)) is not None:
            source_cross_cycles[compact].append(phase)
        opcode = op.slot[0] if op.slot else None
        source_tag_cycles[(op.engine, opcode, op.tag, op.group, op.round)].append(phase)
        source_round_cycles[(op.engine, op.group, op.round)].append(phase)

    configure_target()
    kernel.SCHEDULE_EXACT_CYCLES = None
    target_ops = build_ops()
    target_ids = operation_ids(target_ops)
    target_builder = kernel.KernelBuilder()
    try:
        target_builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        if not hasattr(target_builder, "dag_ops"):
            raise
    _, fallback = target_builder._schedule(
        target_builder.dag_ops, int(os.environ.get("POLICY", "4")),
        return_cycles=True,
    )
    if len(fallback) != len(target_ops):
        raise ValueError("target fallback and target graph disagree")
    fallback_span = max(fallback) + 1

    phases: list[int] = [0] * len(target_ops)
    tie_rank: list[int] = [0] * len(target_ops)
    target_from_source: dict[int, int] = {}
    stats: Counter[str] = Counter()
    for index, op in enumerate(target_ops):
        if op.engine not in SLOT_LIMITS:
            continue
        source_index = source_index_by_id.get(target_ids[index])
        if source_index is not None:
            phases[index] = source_cycles[source_index]
            tie_rank[index] = (
                source_global_rank[source_index]
                if use_global_tie
                else source_rank.get(source_index, source_index)
            )
            target_from_source[source_index] = index
            stats["exact"] += 1
            continue
        key = op_key(op)
        cross_values = source_cross_cycles.get(cross_key(key))
        opcode = op.slot[0] if op.slot else None
        tag_values = source_tag_cycles.get(
            (op.engine, opcode, op.tag, op.group, op.round)
        )
        round_values = source_round_cycles.get((op.engine, op.group, op.round))
        if cross_values:
            phases[index] = median(cross_values)
            stats["cross"] += 1
        elif tag_values:
            phases[index] = median(tag_values)
            stats["tag"] += 1
        elif round_values:
            phases[index] = median(round_values)
            stats["round"] += 1
        else:
            phases[index] = round(
                fallback[index] * (source_span - 1) / max(1, fallback_span - 1)
            )
            stats["fallback"] += 1
        # New operations inherit the closest source phase.  Tying them by the
        # target fallback keeps deterministic construction order while the
        # phase key places them beside their semantic neighbours.
        tie_rank[index] = fallback[index]

    # Preserve every transferable source *resource edge* (rather than forming
    # five independently complete target orders).  We later derive all target
    # orders from one global extension, which prevents cross-engine cycles.
    # Keeping only exact source arcs is intentionally conservative: a
    # scalar/vector rewrite can change the number of operations between two
    # slots, so connecting filtered neighbours would invent a stronger edge.
    resource_edges: list[tuple[int, int, str, int]] = []
    for engine, order in source_orders.items():
        if engine not in SLOT_LIMITS:
            continue
        capacity = SLOT_LIMITS[engine]
        for rank, (source_parent, source_child) in enumerate(
            zip(order, order[capacity:])
        ):
            parent = target_from_source.get(source_parent)
            child = target_from_source.get(source_child)
            if parent is None or child is None:
                continue
            if target_ops[parent].engine != engine or target_ops[child].engine != engine:
                continue
            resource_edges.append((parent, child, engine, rank))

    base_children: list[list[int]] = [[] for _ in target_ops]
    base_indegree = [0] * len(target_ops)
    for child, op in enumerate(target_ops):
        for parent in op.parents:
            base_children[parent].append(child)
            base_indegree[child] += 1

    def topological_with_edges(
        active_resource_edges: list[bool],
    ) -> tuple[list[int], list[int], list[list[int]]]:
        children = [list(row) for row in base_children]
        indegree = list(base_indegree)
        for active, (parent, child, _, _) in zip(
            active_resource_edges, resource_edges
        ):
            if active:
                children[parent].append(child)
                indegree[child] += 1
        ready: list[tuple[int, int, int, int]] = []
        for index, degree in enumerate(indegree):
            if degree == 0:
                heapq.heappush(
                    ready, (phases[index], tie_rank[index], fallback[index], index)
                )
        order: list[int] = []
        while ready:
            _, _, _, parent = heapq.heappop(ready)
            order.append(parent)
            for child in children[parent]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(
                        ready,
                        (phases[child], tie_rank[child], fallback[child], child),
                    )
        return order, [i for i, degree in enumerate(indegree) if degree > 0], children

    # Diagnostic mode: derive one globally legal permutation directly from
    # the target DAG and phase teachers, without importing source capacity
    # arcs.  This is useful as a low-conflict seed for the order search: all
    # resource chains are then generated from the same Kahn extension.
    if os.environ.get("IGNORE_SOURCE_RESOURCE_EDGES", "0") == "1":
        resource_edges = []
    active_resource_edges = [True] * len(resource_edges)
    removed_resource_edges = 0
    while True:
        global_order, cyclic_nodes, constrained_children = topological_with_edges(
            active_resource_edges
        )
        if not cyclic_nodes:
            break
        cyclic_set = set(cyclic_nodes)
        # The base target graph is acyclic.  Every remaining directed cycle
        # therefore contains one of these transferable source arcs.  Dropping
        # all such arcs internal to the residual set breaks every current SCC
        # in one pass while retaining edges outside the rewrite conflict.
        removed = [
            edge_index
            for edge_index, (active, (parent, child, _, _)) in enumerate(
                zip(active_resource_edges, resource_edges)
            )
            if active and parent in cyclic_set and child in cyclic_set
        ]
        if not removed:
            raise ValueError("cannot break target cycle without a base edge")
        for edge_index in removed:
            active_resource_edges[edge_index] = False
        removed_resource_edges += len(removed)

    # ``constrained_children`` is the base DAG plus the acyclic subset of
    # mapped source arcs.  The target engine orders below inherit its global
    # extension and hence remain acyclic when converted to capacity chains.
    children = constrained_children

    engines = tuple(engine for engine in SLOT_LIMITS if any(
        op.engine == engine for op in target_ops
    ))
    orders = {
        engine: [index for index in global_order if target_ops[index].engine == engine]
        for engine in engines
    }
    # Compute an actual earliest schedule under the new capacity chains.  The
    # global order remains a topological order after these edges are added.
    scheduled_children = [list(row) for row in children]
    for engine, order in orders.items():
        capacity = SLOT_LIMITS[engine]
        for parent, child in zip(order, order[capacity:]):
            scheduled_children[parent].append(child)
    earliest = [0] * len(target_ops)
    for parent in global_order:
        parent_start = earliest[parent]
        for child in scheduled_children[parent]:
            earliest[child] = max(earliest[child], parent_start + 1)
    makespan = max(earliest[index] + 1 for index in global_order)
    usage = Counter(
        (earliest[index], op.engine)
        for index, op in enumerate(target_ops)
        if op.engine in SLOT_LIMITS
    )
    for (cycle, engine), count in usage.items():
        if count > SLOT_LIMITS[engine]:
            raise AssertionError((cycle, engine, count))

    output_prefix = os.environ.get("OUT_PREFIX", "/tmp/aopt-global-mapped")
    for engine, order in orders.items():
        Path(f"{output_prefix}-{engine}.json").write_text(
            json.dumps(
                {
                    "engine": engine,
                    "horizon": makespan,
                    "cycles": {
                        str(index): earliest[index]
                        for index in order
                    },
                    "order": order,
                }
            )
        )
    report = {
        "source_span": source_span,
        "target_makespan": makespan,
        "target_jobs": len(target_ops),
        "stats": dict(stats),
        "mapped_resource_edges": len(resource_edges),
        "removed_cycle_edges": removed_resource_edges,
        "counts": {engine: len(order) for engine, order in orders.items()},
    }
    report_path = Path(os.environ.get("OUT_REPORT", f"{output_prefix}.json"))
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
