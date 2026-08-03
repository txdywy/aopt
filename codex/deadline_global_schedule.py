"""Deadline-aware global list scheduling oracle.

The production scheduler is intentionally phase/height oriented.  For a
Hall-tight rewritten graph that can leave a large late tail even though every
operation has a feasible earliest/latest window.  This oracle uses the exact
tail bound as a deadline and repeatedly issues the least-slack ready operation
on each engine.  It is a legal full-DAG seed for the strict CP model and for
the anchor LNS tools; it never relaxes a precedence or capacity constraint.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import heapq
import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops, validate
from problem import SLOT_LIMITS


def build_ops():
    configure_target()
    kernel.SCHEDULE_EXACT_CYCLES = None
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        if not hasattr(builder, "dag_ops"):
            raise
    return real_tail_ops(builder.dag_ops)


def main() -> None:
    ops = build_ops()
    n = len(ops)
    children = [[] for _ in ops]
    indegree = [len(op.parents) for op in ops]
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            children[parent].append((child, int(lag)))

    # Exact DAG tail and reach are both cheap, stable criticality measures.
    q = [i for i, d in enumerate(indegree) if d == 0]
    heapq.heapify(q)
    topo = []
    while q:
        p = heapq.heappop(q)
        topo.append(p)
        for c, _ in children[p]:
            indegree[c] -= 1
            if indegree[c] == 0:
                heapq.heappush(q, c)
    if len(topo) != n:
        raise ValueError("kernel graph is cyclic")
    tail = [0] * n
    reach = [0] * n
    for p in reversed(topo):
        tail[p] = max((lag + tail[c] for c, lag in children[p]), default=0)
        reach[p] = min(1_000_000, len(children[p]) + sum(reach[c] for c, _ in children[p]))

    horizon = int(os.environ.get("TARGET", "959"))
    latest = [horizon - 1 - value for value in tail]
    ready_at = [0] * n
    remaining = [len(op.parents) for op in ops]
    scheduled = [-1] * n
    future: dict[int, list[int]] = defaultdict(list)
    heaps = {engine: [] for engine in SLOT_LIMITS}
    hint: dict[int, int] = {}
    for raw in os.environ.get("HINTS", "").split(","):
        if not raw:
            continue
        payload = json.loads(Path(raw).read_text())
        raw_cycles = payload.get("cycles", {})
        if isinstance(raw_cycles, list):
            hint.update({i: int(c) for i, c in enumerate(raw_cycles)})
        else:
            hint.update({int(i): int(c) for i, c in raw_cycles.items()})

    hint_weight = int(os.environ.get("HINT_WEIGHT", "0"))
    reach_weight = int(os.environ.get("REACH_WEIGHT", "1"))
    tail_weight = int(os.environ.get("TAIL_WEIGHT", "1"))
    noise_mod = int(os.environ.get("NOISE", "0"))

    def noise(index: int) -> int:
        if not noise_mod:
            return 0
        word = (0x9E3779B9 * (index + 1)) & 0xFFFFFFFF
        word ^= word >> 16
        return int(word % (2 * noise_mod + 1)) - noise_mod

    def key(index: int) -> tuple[int, ...]:
        # Static least slack is monotone as the calendar advances.  The
        # source hint is only a late tie-breaker so it cannot starve a job
        # whose deadline is approaching.
        return (
            latest[index],
            latest[index] - tail_weight * tail[index],
            -reach_weight * (reach[index] // 16),
            hint_weight * hint.get(index, horizon),
            noise(index),
            index,
        )

    def push(index: int) -> None:
        heapq.heappush(heaps[ops[index].engine], (key(index), index))

    for i, d in enumerate(remaining):
        if d == 0:
            push(i)

    order = tuple(
        e for e in os.environ.get(
            "ENGINE_ORDER", "flow,load,valu,alu,store,debug"
        ).split(",") if e
    )
    cycle = 0
    done = 0
    while done < n:
        for index in future.pop(cycle, ()):
            push(index)
        used = {engine: 0 for engine in heaps}
        progress = False
        # Repeated passes allow lag-0 children to issue in the same bundle
        # when their engine has a free slot, matching the simulator's writes
        # commit at end of cycle semantics.
        while True:
            pass_progress = False
            for engine in order:
                if engine not in heaps or used[engine] >= SLOT_LIMITS[engine]:
                    continue
                heap = heaps[engine]
                chosen = None
                deferred = []
                while heap:
                    item = heapq.heappop(heap)
                    index = item[1]
                    if ready_at[index] <= cycle:
                        chosen = index
                        break
                    deferred.append(item)
                for item in deferred:
                    heapq.heappush(heap, item)
                if chosen is None:
                    continue
                scheduled[chosen] = cycle
                used[engine] += 1
                done += 1
                progress = pass_progress = True
                for child, lag in children[chosen]:
                    remaining[child] -= 1
                    ready_at[child] = max(ready_at[child], cycle + lag)
                    if remaining[child] == 0:
                        if ready_at[child] <= cycle:
                            push(child)
                        else:
                            future[ready_at[child]].append(child)
            if not pass_progress:
                break
        if not progress:
            cycle = min(future, default=cycle + 1)
        else:
            cycle += 1
        if cycle >= horizon and done < n:
            # Continue to produce a diagnostic schedule rather than silently
            # truncating; strict validation below reports the actual span.
            pass

    validate(ops, scheduled)
    makespan = max(scheduled) + 1
    counts = Counter((scheduled[i], op.engine) for i, op in enumerate(ops))
    print(f"makespan={makespan} jobs={n} horizon={horizon}", flush=True)
    print(
        "tail=" + repr({e: max((c for (c, x), v in counts.items() if x == e), default=-1)
                        for e in SLOT_LIMITS}),
        flush=True,
    )
    output = Path(os.environ.get("OUT", ".search959/deadline-schedule.json"))
    output.write_text(json.dumps({"makespan": makespan, "cycles": scheduled}))
    print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
