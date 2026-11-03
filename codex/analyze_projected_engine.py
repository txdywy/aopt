"""Small diagnostic for projected one-engine resource windows.

This is intentionally a read-only analysis tool: it builds the configured
semantic DAG, projects all omitted engines through the selected engine, and
reports the tightest necessary Hall interval.  It does not alter the scored
kernel or emit a schedule.
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
    kernel.SCHEDULE_EXACT_CYCLES = None
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration, ValueError):
        if not hasattr(builder, "dag_ops"):
            raise
    ops = real_tail_ops(builder.dag_ops)
    n = len(ops)
    children: list[list[tuple[int, int]]] = [[] for _ in ops]
    indegree = [len(op.parents) for op in ops]
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            children[parent].append((child, lag))
    ready = [i for i, d in enumerate(indegree) if d == 0]
    heapq.heapify(ready)
    topo: list[int] = []
    while ready:
        parent = heapq.heappop(ready)
        topo.append(parent)
        for child, _ in children[parent]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(topo) != n:
        raise ValueError("configured DAG is cyclic")
    engine = os.environ.get("ENGINE", "valu")
    selected = {i for i, op in enumerate(ops) if op.engine == engine}
    frontier: list[dict[int, int]] = [{} for _ in ops]
    projected: dict[int, dict[int, int]] = {}
    for child in topo:
        incoming: dict[int, int] = {}
        for parent, lag in ops[child].parents.items():
            sources = ({parent: 0} if parent in selected else frontier[parent])
            for source, distance in sources.items():
                incoming[source] = max(incoming.get(source, -1), distance + lag)
        if child in selected:
            projected[child] = incoming
            frontier[child] = {child: 0}
        else:
            frontier[child] = incoming
    earliest = {i: 0 for i in selected}
    for child in topo:
        if child not in selected:
            continue
        earliest[child] = max(
            (earliest[parent] + lag for parent, lag in projected[child].items()),
            default=0,
        )
    rev_children: dict[int, list[tuple[int, int]]] = {i: [] for i in selected}
    for child, parents in projected.items():
        for parent, lag in parents.items():
            rev_children[parent].append((child, lag))
    tail = {i: 0 for i in selected}
    for parent in reversed(topo):
        if parent in tail:
            tail[parent] = max(
                (lag + tail[child] for child, lag in rev_children[parent]),
                default=0,
            )
    horizon = int(os.environ.get("TARGET", "959"))
    latest = {i: horizon - 1 - tail[i] for i in selected}
    capacity = SLOT_LIMITS[engine]
    best = (-10**9, 0, 0, 0)
    # The selected-engine projected DAG has one operation per job.  A job is
    # contained in [left,right] when its whole feasible interval is inside;
    # that is the standard necessary Hall overload test.
    for left in range(horizon):
        contained = Counter()
        for i in selected:
            if earliest[i] >= left:
                right = latest[i]
                if right >= left:
                    for endpoint in (right,):
                        contained[endpoint] += 1
        running = 0
        for right in range(left, horizon):
            running += contained[right]
            overload = running - capacity * (right - left + 1)
            if overload > best[0]:
                best = (overload, left, right, running)
    print(
        f"engine={engine} jobs={len(selected)} capacity={capacity} "
        f"horizon={horizon} count_floor={(len(selected)+capacity-1)//capacity} "
        f"max_earliest={max(earliest.values(), default=0)} "
        f"best_hall={best}"
    )
    print(
        "tightest_jobs="
        + repr(
            sorted(
                (
                    earliest[i],
                    latest[i],
                    ops[i].group,
                    ops[i].round,
                    ops[i].tag,
                    i,
                )
                for i in selected
                if best[1] <= earliest[i] and latest[i] <= best[2]
            )[:80]
        )
    )


if __name__ == "__main__":
    main()
