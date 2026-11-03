"""Export one exact projected engine DAG for offline scheduling research.

The exporter intentionally mirrors the projection in ``solve_projected_engine``
without adding capacity edges.  It is useful for specialized resource
searches which need to inspect the operation metadata and the weighted
precedence graph directly.
"""

from __future__ import annotations

import heapq
import json
import os
from pathlib import Path
from dataclasses import replace

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from codex.partial_order_hints import load_partial_order_hints


def main() -> None:
    configure_target()
    kernel.SCHEDULE_EXACT_CYCLES = None
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration, ValueError):
        if not hasattr(builder, "dag_ops"):
            raise
    ops = real_tail_ops(builder.dag_ops)
    partial_paths = tuple(
        value for value in os.environ.get("PARTIAL_ORDER_HINTS", "").split(",")
        if value
    )
    if partial_paths:
        partial = load_partial_order_hints(partial_paths, ops)
        parents = [dict(op.parents) for op in ops]
        for parent, child in partial.effective_edges:
            parents[child][parent] = max(parents[child].get(parent, 0), 1)
        ops = [replace(op, parents=parents[index]) for index, op in enumerate(ops)]
        print(
            f"partial_order_hints paths={len(partial.paths)} "
            f"edges={len(partial.effective_edges)}",
            flush=True,
        )
    engine = os.environ.get("ENGINE", "load")
    horizon = int(os.environ.get("TARGET", "959"))

    children: list[list[tuple[int, int]]] = [[] for _ in ops]
    indegree = [len(op.parents) for op in ops]
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            children[parent].append((child, lag))
    ready = [i for i, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    topological: list[int] = []
    while ready:
        parent = heapq.heappop(ready)
        topological.append(parent)
        for child, _ in children[parent]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(topological) != len(ops):
        raise ValueError("kernel DAG is cyclic")

    earliest = [0] * len(ops)
    for child in topological:
        for parent, lag in ops[child].parents.items():
            earliest[child] = max(earliest[child], earliest[parent] + lag)
    tail = [0] * len(ops)
    for parent in reversed(topological):
        tail[parent] = max(
            (lag + tail[child] for child, lag in children[parent]), default=0
        )

    selected = [i for i in topological if ops[i].engine == engine]
    selected_set = set(selected)
    frontier: list[dict[int, int]] = [{} for _ in ops]
    raw_projected: dict[int, dict[int, int]] = {}
    for child in topological:
        incoming: dict[int, int] = {}
        for parent, lag in ops[child].parents.items():
            sources = ({parent: 0} if parent in selected_set else frontier[parent])
            for source, distance in sources.items():
                incoming[source] = max(incoming.get(source, -1), distance + lag)
        if child in selected_set:
            raw_projected[child] = incoming
            frontier[child] = {child: 0}
        else:
            frontier[child] = incoming

    # Match solve_projected_engine's weighted transitive reduction.
    position = {index: rank for rank, index in enumerate(topological)}
    projected: dict[int, dict[int, int]] = {}
    ancestor_distance: dict[int, dict[int, int]] = {}
    for child in selected:
        kept: dict[int, int] = {}
        implied: dict[int, int] = {}
        for parent, lag in sorted(
            raw_projected[child].items(),
            key=lambda item: position[item[0]],
            reverse=True,
        ):
            if implied.get(parent, -1) >= lag:
                continue
            kept[parent] = lag
            implied[parent] = max(implied.get(parent, -1), lag)
            for ancestor, distance in ancestor_distance[parent].items():
                implied[ancestor] = max(implied.get(ancestor, -1), distance + lag)
        projected[child] = kept
        ancestor_distance[child] = implied

    selected_position = {index: rank for rank, index in enumerate(selected)}
    records = []
    for index in selected:
        op = ops[index]
        records.append(
            {
                "index": index,
                "position": selected_position[index],
                "tag": op.tag,
                "group": op.group,
                "round": op.round,
                "earliest": earliest[index],
                "latest": horizon - 1 - tail[index],
                "tail": tail[index],
                "parents": {str(parent): lag for parent, lag in projected[index].items()},
            }
        )
    output = Path(os.environ.get("OUT", f"/tmp/aopt-{engine}-projected-graph.json"))
    output.write_text(
        json.dumps(
            {
                "engine": engine,
                "horizon": horizon,
                "capacity": 2 if engine == "load" else None,
                "ops": records,
                "edges": sum(len(value) for value in projected.values()),
            }
        )
    )
    print(f"engine={engine} jobs={len(selected)} edges={sum(map(len, projected.values()))} output={output}")


if __name__ == "__main__":
    main()
