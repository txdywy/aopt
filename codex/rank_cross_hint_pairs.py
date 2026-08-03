"""Fast cross-engine ranking for pools of individually legal hints.

Generating a 959-cycle ALU or VALU calendar is cheap; evaluating every pair
by starting ``solve_projected_engines`` is not.  This utility builds the
two-engine projected DAG once, then scores all candidate pairs in memory by
their maximum and total cross-engine precedence violation.
"""

from __future__ import annotations

import glob
import heapq
import json
import os
from pathlib import Path

import numpy as np

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops


def projected_cross_edges(
    ops: list[kernel._Op], engines: tuple[str, ...]
) -> list[tuple[int, int, int]]:
    children: list[list[tuple[int, int]]] = [[] for _ in ops]
    indegree = [0] * len(ops)
    for child, op in enumerate(ops):
        indegree[child] = len(op.parents)
        for parent, lag in op.parents.items():
            children[parent].append((child, lag))
    ready = [index for index, degree in enumerate(indegree) if not degree]
    heapq.heapify(ready)
    topological: list[int] = []
    while ready:
        parent = heapq.heappop(ready)
        topological.append(parent)
        for child, _ in children[parent]:
            indegree[child] -= 1
            if not indegree[child]:
                heapq.heappush(ready, child)
    if len(topological) != len(ops):
        raise ValueError("kernel DAG is cyclic")
    position = {index: rank for rank, index in enumerate(topological)}
    selected = {i for i in topological if ops[i].engine in engines}
    frontier: list[dict[int, int]] = [{} for _ in ops]
    projected: dict[int, dict[int, int]] = {}
    for child in topological:
        incoming: dict[int, int] = {}
        for parent, lag in ops[child].parents.items():
            for source, distance in (
                ({parent: 0} if parent in selected else frontier[parent]).items()
            ):
                incoming[source] = max(incoming.get(source, -1), distance + lag)
        if child in selected:
            projected[child] = incoming
            frontier[child] = {child: 0}
        else:
            frontier[child] = incoming

    reduced: dict[int, dict[int, int]] = {}
    ancestors: dict[int, dict[int, int]] = {}
    for child in topological:
        if child not in selected:
            continue
        kept: dict[int, int] = {}
        implied: dict[int, int] = {}
        for parent, lag in sorted(
            projected[child].items(),
            key=lambda item: position[item[0]],
            reverse=True,
        ):
            if implied.get(parent, -1) >= lag:
                continue
            kept[parent] = lag
            implied[parent] = max(implied.get(parent, -1), lag)
            for ancestor, distance in ancestors[parent].items():
                implied[ancestor] = max(implied.get(ancestor, -1), distance + lag)
        reduced[child] = kept
        ancestors[child] = implied
    return [
        (parent, child, lag)
        for child, parents in reduced.items()
        for parent, lag in parents.items()
        if ops[parent].engine != ops[child].engine
    ]


def load_pool(pattern: str, engine: str) -> list[tuple[Path, dict[int, int]]]:
    result = []
    for raw_path in sorted(glob.glob(pattern)):
        path = Path(raw_path)
        payload = json.loads(path.read_text())
        if payload.get("engine") != engine:
            continue
        cycles = {int(index): int(cycle) for index, cycle in payload["cycles"].items()}
        result.append((path, cycles))
    if not result:
        raise FileNotFoundError(f"no {engine} hints matched {pattern!r}")
    return result


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
    edges = projected_cross_edges(ops, ("alu", "valu"))
    alus = load_pool(os.environ["ALU_GLOB"], "alu")
    valus = load_pool(os.environ["VALU_GLOB"], "valu")
    expected = {
        engine: {i for i, op in enumerate(ops) if op.engine == engine}
        for engine in ("alu", "valu")
    }
    for engine, pool in (("alu", alus), ("valu", valus)):
        for path, cycles in pool:
            if set(cycles) != expected[engine]:
                raise ValueError(f"{path} does not cover target {engine} ops")

    def contribution(cycles: dict[int, int], engine: str) -> np.ndarray:
        """One side of ``parent + lag - child`` for every cross edge."""
        return np.fromiter(
            (
                cycles[parent] + lag
                if ops[parent].engine == engine
                else -cycles[child]
                for parent, child, lag in edges
            ),
            dtype=np.int16,
            count=len(edges),
        )

    alu_terms = [contribution(cycles, "alu") for _, cycles in alus]
    valu_terms = [contribution(cycles, "valu") for _, cycles in valus]
    ranked = []
    for alu_index, (alu_path, _) in enumerate(alus):
        for valu_index, (valu_path, _) in enumerate(valus):
            amounts = alu_terms[alu_index] + valu_terms[valu_index]
            positive = amounts[amounts > 0]
            ranked.append(
                (
                    int(positive.max(initial=0)),
                    int(positive.sum(dtype=np.int64)),
                    int(positive.size),
                    str(alu_path),
                    str(valu_path),
                )
            )
    ranked.sort()
    limit = int(os.environ.get("LIMIT", "40"))
    print(
        f"cross_edges={len(edges)} alu_candidates={len(alus)} "
        f"valu_candidates={len(valus)} pairs={len(ranked)}",
        flush=True,
    )
    for maximum, total, count, alu_path, valu_path in ranked[:limit]:
        print(
            json.dumps(
                {
                    "maximum": maximum,
                    "total": total,
                    "count": count,
                    "alu": alu_path,
                    "valu": valu_path,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
