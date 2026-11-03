"""Report exact full-DAG release/deadline Hall bounds for every engine.

``solve_projected_engine`` has a useful Hall-cut analyser, but constructing
the 20k-operation kernel separately for every resource made semantic-design
experiments unnecessarily slow and encouraged comparing partial results.  This
small companion constructs the effective graph once and reports the same
necessary capacity bound for every physical engine under one shared horizon.

It intentionally does *not* add a resource order: the result is a lower
bound, not evidence that the resources can be jointly scheduled.  That
distinction is especially important for the almost saturated 959-cycle
variants.

Example::

    source /private/tmp/aopt-vrestore1-p6input-m4m23-959.env
    TARGET=959 python3 -m codex.analyze_full_hall
"""

from __future__ import annotations

from collections import Counter
import heapq
import os

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from problem import SLOT_LIMITS


def main() -> None:
    configure_target()
    # Do not let a schedule embedded for a different variant interfere with
    # construction of the effective semantic DAG.
    kernel.SCHEDULE_EXACT_CYCLES = None
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        if not hasattr(builder, "dag_ops"):
            raise
    ops = real_tail_ops(builder.dag_ops)
    horizon = int(os.environ.get("TARGET", "959"))

    children: list[list[tuple[int, int]]] = [[] for _ in ops]
    indegree = [len(op.parents) for op in ops]
    for child, op in enumerate(ops):
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

    earliest = [0] * len(ops)
    for child in topological:
        for parent, lag in ops[child].parents.items():
            earliest[child] = max(earliest[child], earliest[parent] + lag)
    tail = [0] * len(ops)
    for parent in reversed(topological):
        tail[parent] = max(
            (lag + tail[child] for child, lag in children[parent]), default=0
        )
    latest = [horizon - 1 - value for value in tail]
    counts = Counter(op.engine for op in ops)
    dag_lb = max(earliest, default=-1) + 1
    print(
        f"full_hall horizon={horizon} ops={len(ops)} "
        f"edges={sum(len(op.parents) for op in ops)} dag_lb={dag_lb}",
        flush=True,
    )

    for engine, capacity in SLOT_LIMITS.items():
        jobs = [index for index, op in enumerate(ops) if op.engine == engine]
        if not jobs:
            continue
        # Incremental right endpoint scan for every left endpoint.  The
        # horizon is < 1k, so the simple formulation is both transparent and
        # fast enough for dense design sweeps.
        best = (-10**9, 0, horizon - 1, 0)
        for left in range(horizon):
            by_latest = [0] * horizon
            for index in jobs:
                if earliest[index] >= left and 0 <= latest[index] < horizon:
                    by_latest[latest[index]] += 1
            trapped = 0
            for right, count in enumerate(by_latest):
                if right < left:
                    continue
                trapped += count
                overload = trapped - capacity * (right - left + 1)
                candidate = (overload, left, right, trapped)
                if candidate > best:
                    best = candidate
        overload, left, right, trapped = best
        window_jobs = [
            index
            for index in jobs
            if earliest[index] >= left and latest[index] <= right
        ]
        tags = Counter(ops[index].tag for index in window_jobs)
        print(
            f"engine={engine} jobs={len(jobs)} capacity={capacity} "
            f"floor={(len(jobs) + capacity - 1) // capacity} "
            f"slack={horizon * capacity - len(jobs)} "
            f"hall={overload} window={left}:{right} trapped={trapped}/"
            f"{capacity * (right - left + 1)} tags="
            + ",".join(f"{tag}:{count}" for tag, count in tags.most_common(8)),
            flush=True,
        )
        if os.environ.get("DETAIL_HALL") == "1":
            detail = Counter(
                (ops[index].tag, ops[index].group, ops[index].round)
                for index in window_jobs
            )
            print(
                "detail="
                + ",".join(
                    f"{tag}@{group}:{rnd}:{count}"
                    for (tag, group, rnd), count in detail.most_common()
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
