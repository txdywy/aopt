"""Rank engine orders by Hall overload while leaving their cycles flexible."""

from __future__ import annotations

import glob
import heapq
import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from codex.rank_fixed_engine_hints import hall_overload


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
    horizon = int(os.environ.get("TARGET", "959"))
    engine = os.environ.get("ENGINE", "flow")
    verbose = bool(int(os.environ.get("VERBOSE", "0")))
    ranked = []
    for filename in glob.glob(os.environ["HINT_GLOB"]):
        raw_cycles = json.loads(Path(filename).read_text())["cycles"]
        cycles = (
            {int(index): int(cycle) for index, cycle in raw_cycles.items()}
            if isinstance(raw_cycles, dict)
            else raw_cycles
        )
        sequence = sorted(
            (i for i, op in enumerate(ops) if op.engine == engine),
            key=lambda i: (cycles[i], i),
        )
        parents = [dict(op.parents) for op in ops]
        for previous, current in zip(sequence, sequence[1:]):
            parents[current][previous] = max(parents[current].get(previous, 0), 1)
        children: list[list[tuple[int, int]]] = [[] for _ in ops]
        indegree = [0] * len(ops)
        for child, child_parents in enumerate(parents):
            indegree[child] = len(child_parents)
            for parent, lag in child_parents.items():
                children[parent].append((child, lag))
        ready = [index for index, degree in enumerate(indegree) if degree == 0]
        heapq.heapify(ready)
        order = []
        while ready:
            parent = heapq.heappop(ready)
            order.append(parent)
            for child, _ in children[parent]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(ready, child)
        if len(order) != len(ops):
            if verbose:
                print(
                    f"skip=cycle visited={len(order)}/{len(ops)} {filename}"
                )
            continue
        earliest = [0] * len(ops)
        for child in order:
            earliest[child] = max(
                (earliest[parent] + lag for parent, lag in parents[child].items()),
                default=0,
            )
        latest = [horizon - 1] * len(ops)
        for parent in reversed(order):
            latest[parent] = min(
                (latest[child] - lag for child, lag in children[parent]),
                default=horizon - 1,
            )
        if any(left > right for left, right in zip(earliest, latest)):
            if verbose:
                worst = max(
                    (
                        left - right,
                        index,
                        left,
                        right,
                    )
                    for index, (left, right) in enumerate(
                        zip(earliest, latest)
                    )
                )
                print(
                    f"skip=window worst={worst} dag={max(earliest) + 1} "
                    f"{filename}"
                )
            continue
        overloads = tuple(
            hall_overload(ops, earliest, latest, target, horizon)
            for target in ("alu", "valu", "load", "store", "flow")
        )
        key = (
            max(overloads),
            sum(max(0, value) for value in overloads),
            max(earliest),
        )
        ranked.append((key, filename, overloads))
    for key, filename, overloads in sorted(ranked)[:50]:
        print(
            f"max={key[0]:4d} sum={key[1]:4d} dag={key[2] + 1:4d} "
            f"overloads={overloads} {filename}"
        )


if __name__ == "__main__":
    main()
