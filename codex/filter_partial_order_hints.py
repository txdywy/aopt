"""Turn an acyclic partial resource seed into a horizon-safe strict seed.

``break_resource_cycles.py`` removes feedback cycles from independently
constructed engine calendars, but the resulting acyclic union can still make
the precedence-only critical path much longer than the target horizon.  This
tool keeps as much of that useful order as possible without making a false
feasibility claim:

1. validate every explicit resource arc with :mod:`partial_order_hints`;
2. take one deterministic global topological extension of the *entire*
   acyclic partial seed (Kahn, lowest original operation index on ties);
3. consider the arcs in a deterministic priority order and retain an arc only
   if its exact current longest-path bound remains within ``TARGET``;
4. write partial-hint JSON files whose retained arcs are all forward in that
   global extension and whose augmented-DAG precedence span is independently
   re-audited.

The output is deliberately still a partial order, not a schedule.  A full
solver must retain every original DAG edge and every engine capacity
constraint.  The useful guarantee is narrower and strict: adding all emitted
``resource_edges`` as ``child >= parent + 1`` cannot create a DAG cycle or a
precedence-only lower bound above ``TARGET``.

Example for the hash193 candidate::

    source codex/sub959_candidate_balanced_hash193.env
    PARTIAL_ORDER_HINTS=/private/tmp/hash193-break-alu.json,... \\
    TARGET=959 OUT_PREFIX=/private/tmp/hash193-forward \\
      python3 -m codex.filter_partial_order_hints
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Sequence

from codex.partial_order_hints import (
    PartialOrderHints,
    _build_target_ops,
    audit_augmented_dag,
    load_partial_order_hints,
)


@dataclass(frozen=True)
class FilterResult:
    """One strict horizon-safe subset of a partial resource-order seed."""

    partial: PartialOrderHints
    source_resource_edges: int
    source_effective_edges: int
    source_by_engine: dict[str, int]
    source_effective_by_engine: dict[str, int]
    full_precedence_span: int
    filtered_precedence_span: int
    topology_tie_break: str
    edge_priority: str
    retained_by_engine: dict[str, int]
    dropped_by_engine: dict[str, int]


def _raw_graph(
    ops: Sequence[object], topological: Sequence[int]
) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]], list[int], list[int]]:
    """Build raw adjacency plus exact earliest/tail labels in ``topological``."""
    children: list[list[tuple[int, int]]] = [[] for _ in ops]
    parents: list[list[tuple[int, int]]] = [[] for _ in ops]
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            children[parent].append((child, lag))
            parents[child].append((parent, lag))

    earliest = [0] * len(ops)
    for child in topological:
        for parent, lag in parents[child]:
            earliest[child] = max(earliest[child], earliest[parent] + lag)
    tail = [0] * len(ops)
    for parent in reversed(topological):
        for child, lag in children[parent]:
            tail[parent] = max(tail[parent], lag + tail[child])
    return children, parents, earliest, tail


def _priority_edges(
    edges: Sequence[tuple[int, int]],
    ops: Sequence[object],
    earliest: Sequence[int],
    tail: Sequence[int],
    priority: str,
) -> list[tuple[int, int]]:
    """Order optional arcs before exact incremental critical-path filtering."""
    if priority == "input":
        return list(edges)
    if priority == "input-reverse":
        return list(reversed(edges))
    if priority == "rawcost-asc":
        return sorted(
            edges,
            key=lambda edge: (earliest[edge[0]] + 1 + tail[edge[1]], edge),
        )
    if priority == "rawcost-desc":
        return sorted(
            edges,
            key=lambda edge: (-(earliest[edge[0]] + 1 + tail[edge[1]]), edge),
        )
    if priority in {"round-robin", "round-robin-reverse"}:
        # This gives every engine an early chance to retain its high-value
        # arcs.  It trades total retained edge count for a less ALU-dominant
        # anchor, which can be useful for an LNS whose bottleneck is a smaller
        # capacity-one resource.
        by_engine: dict[str, list[tuple[int, int]]] = {}
        engine_order: list[str] = []
        for edge in edges:
            engine = ops[edge[1]].engine
            if engine not in by_engine:
                by_engine[engine] = []
                engine_order.append(engine)
            by_engine[engine].append(edge)
        if priority == "round-robin-reverse":
            engine_order.reverse()
        cursor = {engine: 0 for engine in engine_order}
        result: list[tuple[int, int]] = []
        while True:
            progressed = False
            for engine in engine_order:
                position = cursor[engine]
                engine_edges = by_engine[engine]
                if position >= len(engine_edges):
                    continue
                result.append(engine_edges[position])
                cursor[engine] = position + 1
                progressed = True
            if not progressed:
                return result
    raise ValueError(
        "unknown EDGE_PRIORITY "
        f"{priority!r}; choose input, input-reverse, rawcost-asc, rawcost-desc, "
        "round-robin, or round-robin-reverse"
    )


def _relax_earliest(
    children: Sequence[Sequence[tuple[int, int]]],
    earliest: list[int],
    node: int,
    value: int,
) -> None:
    """Incrementally propagate a larger release label through a DAG."""
    if value <= earliest[node]:
        return
    earliest[node] = value
    stack = [node]
    while stack:
        parent = stack.pop()
        parent_earliest = earliest[parent]
        for child, lag in children[parent]:
            candidate = parent_earliest + lag
            if candidate > earliest[child]:
                earliest[child] = candidate
                stack.append(child)


def _relax_tail(
    parents: Sequence[Sequence[tuple[int, int]]],
    tail: list[int],
    node: int,
    value: int,
) -> None:
    """Incrementally propagate a larger tail label backward through a DAG."""
    if value <= tail[node]:
        return
    tail[node] = value
    stack = [node]
    while stack:
        child = stack.pop()
        child_tail = tail[child]
        for parent, lag in parents[child]:
            candidate = child_tail + lag
            if candidate > tail[parent]:
                tail[parent] = candidate
                stack.append(parent)


def subsample_effective_edges(
    ops: Sequence[object],
    partial: PartialOrderHints,
    stride: int,
    offset: int = 0,
) -> PartialOrderHints:
    """Keep a deterministic residue class of optional effective arcs.

    This is chiefly an oracle-friendly diagnostic knob: a capacity-one
    projected solve can reveal how much foreign resource order it can absorb
    before a 959-cycle calendar loses its few remaining bubbles.  Direct DAG
    duplicates remain serialized because they do not alter the constraint
    graph.
    """
    if stride <= 0:
        raise ValueError("SOURCE_EDGE_STRIDE must be positive")
    if not 0 <= offset < stride:
        raise ValueError("SOURCE_EDGE_OFFSET must satisfy 0 <= offset < stride")
    if stride == 1:
        return partial
    effective = tuple(
        edge
        for rank, edge in enumerate(partial.effective_edges)
        if rank % stride == offset
    )
    effective_set = set(partial.effective_edges)
    kept_set = set(effective)
    raw = tuple(
        edge
        for edge in partial.resource_edges
        if edge not in effective_set or edge in kept_set
    )
    return PartialOrderHints(
        paths=partial.paths,
        resource_edges=raw,
        effective_edges=effective,
        by_engine=dict(Counter(ops[child].engine for _, child in raw)),
        effective_by_engine=dict(Counter(ops[child].engine for _, child in effective)),
        direct_dag_redundant=len(raw) - len(effective),
    )


def filter_to_horizon(
    ops: Sequence[object],
    partial: PartialOrderHints,
    horizon: int,
    *,
    edge_priority: str = "input",
) -> FilterResult:
    """Keep a maximum-by-greedy strict subset without exceeding ``horizon``.

    The all-edge partial graph must be acyclic; its deterministic topological
    extension is fixed for this filtering pass.  Consequently every retained
    edge is forward, so the incremental longest-path test is exact: adding
    ``parent -> child`` changes the longest path only through
    ``earliest[parent] + 1 + tail[child]``.

    This is a deterministic greedy maximization, not a proof of the globally
    maximum-cardinality subset.  It is intentionally cheap enough to run
    before a full CP-SAT LNS solve.
    """
    if horizon <= 0:
        raise ValueError("TARGET must be positive")
    full_audit = audit_augmented_dag(ops, partial)
    topological = full_audit.topological
    position = [0] * len(ops)
    for rank, index in enumerate(topological):
        position[index] = rank
    if any(position[parent] >= position[child] for parent, child in partial.effective_edges):
        raise AssertionError("full partial-order topological extension is malformed")

    raw_children, raw_parents, raw_earliest, raw_tail = _raw_graph(ops, topological)
    if max(raw_earliest, default=-1) + 1 > horizon:
        raise ValueError(
            f"raw DAG precedence span {max(raw_earliest, default=-1) + 1} exceeds TARGET={horizon}"
        )
    candidates = _priority_edges(
        partial.effective_edges, ops, raw_earliest, raw_tail, edge_priority
    )

    # These mutable adjacency lists contain exactly the constraints accepted
    # so far.  They remain a subgraph of ``topological`` throughout.
    children = [list(row) for row in raw_children]
    parents = [list(row) for row in raw_parents]
    earliest = list(raw_earliest)
    tail = list(raw_tail)
    retained_effective: list[tuple[int, int]] = []
    dropped_by_engine: Counter[str] = Counter()
    for parent, child in candidates:
        # Since the fixed topological extension has parent before child, the
        # new edge cannot lie on either side of the two labels being queried.
        # The condition is therefore the exact post-add critical-path bound.
        if earliest[parent] + 1 + tail[child] > horizon - 1:
            dropped_by_engine[ops[child].engine] += 1
            continue
        children[parent].append((child, 1))
        parents[child].append((parent, 1))
        retained_effective.append((parent, child))
        _relax_earliest(children, earliest, child, earliest[parent] + 1)
        _relax_tail(parents, tail, parent, tail[child] + 1)

    effective_set = set(partial.effective_edges)
    retained_effective_set = set(retained_effective)
    # Direct DAG arcs of lag >= 1 are already strict constraints.  Keep them
    # in the serialized provenance even though they do not inflate a CP model.
    retained_raw = tuple(
        edge
        for edge in partial.resource_edges
        if edge not in effective_set or edge in retained_effective_set
    )
    retained_by_engine = Counter(ops[child].engine for _, child in retained_effective)
    filtered = PartialOrderHints(
        paths=partial.paths,
        resource_edges=retained_raw,
        effective_edges=tuple(retained_effective),
        by_engine=dict(Counter(ops[child].engine for _, child in retained_raw)),
        effective_by_engine=dict(retained_by_engine),
        direct_dag_redundant=len(retained_raw) - len(retained_effective),
    )
    filtered_audit = audit_augmented_dag(ops, filtered)
    incremental_span = max(earliest, default=-1) + 1
    if filtered_audit.longest_path_span != incremental_span:
        raise AssertionError(
            "incremental critical-path maintenance disagrees with independent audit "
            f"{incremental_span} vs {filtered_audit.longest_path_span}"
        )
    if filtered_audit.longest_path_span > horizon:
        raise AssertionError("horizon filter emitted an overlong partial order")
    return FilterResult(
        partial=filtered,
        source_resource_edges=len(partial.resource_edges),
        source_effective_edges=len(partial.effective_edges),
        source_by_engine=partial.by_engine,
        source_effective_by_engine=partial.effective_by_engine,
        full_precedence_span=full_audit.longest_path_span,
        filtered_precedence_span=filtered_audit.longest_path_span,
        topology_tie_break="Kahn over full partial DAG; lowest original op index",
        edge_priority=edge_priority,
        retained_by_engine=dict(retained_by_engine),
        dropped_by_engine=dict(dropped_by_engine),
    )


def _payloads_by_engine(paths: Sequence[Path]) -> dict[str, dict[str, object]]:
    """Reload validated source payloads so metadata/cycles survive serialization."""
    result: dict[str, dict[str, object]] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        engine = payload.get("engine")
        if not isinstance(engine, str):
            raise AssertionError(f"validated payload unexpectedly lacks engine: {path}")
        result[engine] = payload
    return result


def write_filtered_hints(
    ops: Sequence[object],
    result: FilterResult,
    output_prefix: Path,
    horizon: int,
) -> tuple[list[Path], Path]:
    """Write per-engine JSON and a report, then round-trip audit the files."""
    source_payloads = _payloads_by_engine(result.partial.paths)
    raw_by_engine: dict[str, list[tuple[int, int]]] = {}
    for parent, child in result.partial.resource_edges:
        raw_by_engine.setdefault(ops[child].engine, []).append((parent, child))

    outputs: list[Path] = []
    filtered_counts = Counter(ops[child].engine for _, child in result.partial.resource_edges)
    for engine, payload in source_payloads.items():
        output = Path(f"{output_prefix}-{engine}.json")
        result_payload = dict(payload)
        result_payload["resource_edges"] = [
            [int(parent), int(child)]
            for parent, child in raw_by_engine.get(engine, [])
        ]
        result_payload["strict_global_seed"] = {
            "target": horizon,
            "topology_tie_break": result.topology_tie_break,
            "edge_priority": result.edge_priority,
            "full_partial_precedence_span": result.full_precedence_span,
            "filtered_precedence_span": result.filtered_precedence_span,
            "source_resource_edges": result.source_by_engine.get(engine, 0),
            "source_effective_resource_edges": result.source_effective_by_engine.get(
                engine, 0
            ),
            "filtered_resource_edges": filtered_counts[engine],
            "effective_resource_edges": result.retained_by_engine.get(engine, 0),
            "dropped_effective_edges": result.dropped_by_engine.get(engine, 0),
        }
        output.write_text(json.dumps(result_payload, separators=(",", ":")) + "\n")
        outputs.append(output)

    # A fresh parse catches accidental serialization drift and independently
    # certifies the exact files a full model will consume.
    round_trip = load_partial_order_hints(outputs, ops)
    round_trip_audit = audit_augmented_dag(ops, round_trip)
    if round_trip.resource_edges != result.partial.resource_edges:
        raise AssertionError("serialized resource-edge order changed on round trip")
    if round_trip_audit.longest_path_span != result.filtered_precedence_span:
        raise AssertionError("serialized partial hints changed precedence span")

    report = {
        "strict_global_partial_seed": True,
        "target": horizon,
        "source_paths": [str(path) for path in result.partial.paths],
        "output_paths": [str(path) for path in outputs],
        "topology_tie_break": result.topology_tie_break,
        "edge_priority": result.edge_priority,
        "full_partial_precedence_span": result.full_precedence_span,
        "filtered_precedence_span": result.filtered_precedence_span,
        "source_resource_edges": result.source_resource_edges,
        "source_effective_edges": result.source_effective_edges,
        "source_by_engine": result.source_by_engine,
        "source_effective_by_engine": result.source_effective_by_engine,
        "filtered_resource_edges": len(round_trip.resource_edges),
        "filtered_effective_edges": len(round_trip.effective_edges),
        "direct_dag_redundant": round_trip.direct_dag_redundant,
        "effective_by_engine": round_trip.effective_by_engine,
        "dropped_effective_by_engine": result.dropped_by_engine,
    }
    report_path = Path(f"{output_prefix}.report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return outputs, report_path


def main() -> None:
    raw_paths = [
        value
        for value in os.environ.get("PARTIAL_ORDER_HINTS", "").split(",")
        if value
    ]
    if not raw_paths:
        raise ValueError("PARTIAL_ORDER_HINTS must name one or more JSON files")
    horizon = int(os.environ.get("TARGET", os.environ.get("HORIZON", "959")))
    priority = os.environ.get("EDGE_PRIORITY", "input")
    source_stride = int(os.environ.get("SOURCE_EDGE_STRIDE", "1"))
    source_offset = int(os.environ.get("SOURCE_EDGE_OFFSET", "0"))
    prefix = Path(os.environ.get("OUT_PREFIX", "/tmp/aopt-forward-partial"))

    ops = _build_target_ops()
    partial = subsample_effective_edges(
        ops,
        load_partial_order_hints(raw_paths, ops),
        source_stride,
        source_offset,
    )
    result = filter_to_horizon(ops, partial, horizon, edge_priority=priority)
    outputs, report = write_filtered_hints(ops, result, prefix, horizon)
    print(
        f"strict_partial_filter=OK target={horizon} "
        f"full_span={result.full_precedence_span} "
        f"filtered_span={result.filtered_precedence_span} "
        f"effective_edges={len(result.partial.effective_edges)} "
        f"resource_edges={len(result.partial.resource_edges)} "
        f"retained_by_engine={result.retained_by_engine} "
        f"dropped_by_engine={result.dropped_by_engine} "
        f"tie_break={result.topology_tie_break!r} "
        f"priority={priority} source_stride={source_stride}:{source_offset} "
        f"outputs={','.join(map(str, outputs))} report={report}",
        flush=True,
    )


if __name__ == "__main__":
    main()
