"""Make a set of mapped resource-order hints globally acyclic.

A projected ALU/VALU schedule can be individually feasible while its two
resource orders form a directed cycle through an omitted engine.  This tool
keeps as much of those useful orders as possible: it repeatedly finds one
concrete cycle in the *full* target DAG and deletes exactly one resource edge
from that cycle.  The resulting partial-order hints are safe inputs to
``solve_projected_engine.PARTIAL_ORDER_HINTS``; Cumulative constraints still
enforce capacity for the deliberately released gaps.

It is intentionally a preconditioner, not a scheduler.  Keeping the order
correlation of a near-compatible incumbent is much more useful than asking a
cold global solver to rediscover a 17k-operation permutation.
"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from problem import SLOT_LIMITS


def find_cycle(
    node_count: int,
    edges: list[tuple[int, int, str | None]],
) -> list[int] | None:
    """Return edge indices for one directed cycle, or ``None`` if acyclic."""
    children: list[list[tuple[int, int]]] = [[] for _ in range(node_count)]
    indegree = [0] * node_count
    for edge_index, (parent, child, _) in enumerate(edges):
        children[parent].append((child, edge_index))
        indegree[child] += 1

    ready = [index for index, degree in enumerate(indegree) if not degree]
    cursor = 0
    while cursor < len(ready):
        parent = ready[cursor]
        cursor += 1
        for child, _ in children[parent]:
            indegree[child] -= 1
            if not indegree[child]:
                ready.append(child)
    if len(ready) == node_count:
        return None

    # A DFS over only the residual SCC region gives an actual cycle rather
    # than the much larger set reported by Kahn's algorithm.
    color = [0] * node_count
    parent_edge = [-1] * node_count
    stack: list[tuple[int, int]] = []
    for root in range(node_count):
        if indegree[root] <= 0 or color[root]:
            continue
        color[root] = 1
        stack.append((root, 0))
        while stack:
            node, child_cursor = stack[-1]
            if child_cursor >= len(children[node]):
                color[node] = 2
                stack.pop()
                continue
            child, edge_index = children[node][child_cursor]
            stack[-1] = (node, child_cursor + 1)
            if indegree[child] <= 0:
                continue
            if color[child] == 0:
                color[child] = 1
                parent_edge[child] = edge_index
                stack.append((child, 0))
                continue
            if color[child] != 1:
                continue
            result = [edge_index]
            walk = node
            while walk != child:
                previous = parent_edge[walk]
                if previous < 0:
                    raise AssertionError("malformed DFS parent chain")
                result.append(previous)
                walk = edges[previous][0]
            return result
    raise AssertionError("Kahn residual did not contain a DFS cycle")


def main() -> None:
    configure_target()
    kernel.SCHEDULE_EXACT_CYCLES = None
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        if not hasattr(builder, "dag_ops"):
            raise
    ops = real_tail_ops(builder.dag_ops)

    paths = [Path(value) for value in os.environ["HINTS"].split(",") if value]
    if not paths:
        raise ValueError("HINTS must name one or more mapped engine hints")
    payloads = [json.loads(path.read_text()) for path in paths]
    by_engine = {str(payload["engine"]): payload for payload in payloads}
    if len(by_engine) != len(payloads):
        raise ValueError("HINTS contains duplicate engines")

    edges: list[tuple[int, int, str | None]] = []
    for child, op in enumerate(ops):
        edges.extend((parent, child, None) for parent in op.parents)

    resource_edge_indices: dict[str, list[int]] = {}
    resource_edges_by_engine: dict[str, list[tuple[int, int]]] = {}
    matched_by_engine: dict[str, set[int]] = {}
    for engine, payload in by_engine.items():
        # A mapper carries ``matched_indices`` and explicit resource arcs, but
        # a native projected-engine solve only has issue cycles.  Treat the
        # latter as a complete, locally legal teacher order.  This lets the
        # cycle breaker reconcile independently solved *target* engines too,
        # rather than only orders transferred from a different variant.
        expected = {i for i, op in enumerate(ops) if op.engine == engine}
        raw_matched = payload.get("matched_indices")
        matched = (
            {int(index) for index in raw_matched}
            if raw_matched is not None
            else expected
        )
        if not matched <= expected:
            raise ValueError(f"{engine} matched_indices has wrong endpoint")
        raw_edges = payload.get("resource_edges")
        if raw_edges is None:
            cycles = {
                int(index): int(cycle)
                for index, cycle in payload.get("cycles", {}).items()
            }
            if not matched <= set(cycles):
                raise ValueError(f"{engine} cycles do not cover matched indices")
            raw_order = payload.get("order")
            if raw_order is None:
                order = sorted(matched, key=lambda index: (cycles[index], index))
            else:
                order = [int(index) for index in raw_order if int(index) in matched]
            if len(order) != len(matched) or set(order) != matched:
                raise ValueError(f"{engine} order does not cover matched indices")
            capacity = SLOT_LIMITS[engine]
            raw_edges = list(zip(order, order[capacity:]))
        normalized_edges: list[tuple[int, int]] = []
        for raw_parent, raw_child in raw_edges:
            parent, child = int(raw_parent), int(raw_child)
            if parent not in matched or child not in matched:
                continue
            if ops[parent].engine != engine or ops[child].engine != engine:
                raise ValueError(f"{engine} resource edge has wrong endpoint")
            normalized_edges.append((parent, child))
        matched_by_engine[engine] = matched
        resource_edges_by_engine[engine] = normalized_edges
        resource_edge_indices[engine] = []
        for parent, child in normalized_edges:
            resource_edge_indices[engine].append(len(edges))
            edges.append((parent, child, engine))

    active = [True] * len(edges)
    removed: dict[str, set[tuple[int, int]]] = {
        engine: set() for engine in by_engine
    }
    deletion_count: Counter[str] = Counter()
    max_breaks = int(os.environ.get("MAX_BREAKS", "20000"))
    complete = False
    for iteration in range(max_breaks):
        compact = [edge for index, edge in enumerate(edges) if active[index]]
        cycle_compact = find_cycle(len(ops), compact)
        if cycle_compact is None:
            complete = True
            print(
                f"status=ACYCLIC breaks={iteration} "
                f"remaining_resource_edges="
                f"{ {engine: sum(active[index] for index in indices) for engine, indices in resource_edge_indices.items()} }",
                flush=True,
            )
            break
        cycle = [
            index
            for index, enabled in enumerate(active)
            if enabled
        ]
        cycle = [cycle[index] for index in cycle_compact]
        choices = [index for index in cycle if edges[index][2] is not None]
        if not choices:
            raise AssertionError("the original target DAG is cyclic")

        # Distribute cuts between engines and groups.  Concentrating every
        # cut in one group destroys precisely the local order correlation we
        # are trying to retain; round-robin selection leaves a better shaped
        # residual calendar for the subsequent CP repair.
        def choice_key(index: int) -> tuple[int, int, int, int, int]:
            parent, child, engine = edges[index]
            assert engine is not None
            op = ops[child]
            return (
                deletion_count[engine],
                deletion_count[f"{engine}:{op.group}:{op.round}"],
                op.round,
                op.group,
                index,
            )

        chosen = min(choices, key=choice_key)
        parent, child, engine = edges[chosen]
        assert engine is not None
        active[chosen] = False
        removed[engine].add((parent, child))
        deletion_count[engine] += 1
        deletion_count[f"{engine}:{ops[child].group}:{ops[child].round}"] += 1
        if iteration < 20 or (iteration + 1) % 100 == 0:
            print(
                f"break={iteration + 1} cycle_edges={len(cycle)} "
                f"engine={engine} parent={parent} child={child} "
                f"g={ops[child].group} r={ops[child].round}",
                flush=True,
            )
    if not complete:
        if not bool(int(os.environ.get("ALLOW_PARTIAL", "0"))):
            raise RuntimeError(f"cycle breaker exceeded MAX_BREAKS={max_breaks}")
        print(
            f"status=PARTIAL breaks={max_breaks} "
            f"remaining_resource_edges="
            f"{ {engine: sum(active[index] for index in indices) for engine, indices in resource_edge_indices.items()} }",
            flush=True,
        )

    prefix = Path(os.environ.get("OUT_PREFIX", "/tmp/aopt-acyclic-partial"))
    for engine, payload in by_engine.items():
        result = dict(payload)
        result["matched_indices"] = sorted(matched_by_engine[engine])
        result["resource_edges"] = [
            [int(parent), int(child)]
            for parent, child in resource_edges_by_engine[engine]
            if (int(parent), int(child)) not in removed[engine]
        ]
        result["cycle_breaks"] = len(removed[engine])
        output = Path(f"{prefix}-{engine}.json")
        output.write_text(json.dumps(result))
        print(
            f"engine={engine} removed={len(removed[engine])} output={output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
