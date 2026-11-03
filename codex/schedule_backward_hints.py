"""Backward resource-list scheduler for mapped source calendars.

Forward list scheduling tends to consume a saturated engine with early-phase
operations and leaves the final cross-engine cone stranded.  This companion
scheduler starts from sinks and works backwards, using the mapped source cycle
as a soft latest-time preference.  It is a research oracle: every emitted
cycle is checked against the full DAG and capacities before being written.
"""

from __future__ import annotations

from collections import defaultdict
import heapq
import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops, validate
from problem import SLOT_LIMITS


def main() -> None:
    configure_target(); kernel.SCHEDULE_EXACT_CYCLES = None
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        if not hasattr(builder, "dag_ops"):
            raise
    ops = real_tail_ops(builder.dag_ops); n = len(ops)
    hints: dict[int, int] = {}
    for raw in os.environ.get("HINTS", "").split(","):
        if not raw: continue
        p = json.loads(Path(raw).read_text())
        hints.update({int(i): int(c) for i, c in p["cycles"].items()})
    children = [[] for _ in ops]
    outdegree = [0] * n
    for child, op in enumerate(ops):
        outdegree[child] = len(op.parents) * 0
        for parent, lag in op.parents.items():
            children[parent].append((child, int(lag)))
            outdegree[parent] += 1
    # exact distance to a source is useful as a tie breaker when many sinks
    # share the same mapped phase.
    topo = []
    indeg = [len(op.parents) for op in ops]
    q = [i for i, d in enumerate(indeg) if d == 0]; heapq.heapify(q)
    while q:
        p = heapq.heappop(q); topo.append(p)
        for c, _ in children[p]:
            indeg[c] -= 1
            if indeg[c] == 0: heapq.heappush(q, c)
    depth = [0] * n
    for p in topo:
        for c, lag in children[p]:
            depth[c] = max(depth[c], depth[p] + lag + 1)

    latest = [10**9] * n
    heaps = {engine: [] for engine in SLOT_LIMITS}
    scheduled = [-1] * n
    ready_at: dict[int, list[int]] = defaultdict(list)
    horizon = int(os.environ.get("TARGET", "959"))
    cycle = horizon - 1
    # A larger hint is later in the source calendar.  The explicit tail-like
    # depth term prevents independent setup sinks from consuming the window.
    hint_weight = int(os.environ.get("HINT_WEIGHT", "8"))
    depth_weight = int(os.environ.get("DEPTH_WEIGHT", "1"))
    def key(i: int) -> tuple[int, ...]:
        return (
            -(hint_weight * hints.get(i, horizon) + depth_weight * depth[i]),
            -hints.get(i, horizon),
            -depth[i],
            i,
        )
    def push(i: int) -> None:
        heapq.heappush(heaps[ops[i].engine], (key(i), i))
    for i, d in enumerate(outdegree):
        if d == 0: push(i)

    done = 0
    order = tuple(x for x in os.environ.get("ENGINE_ORDER", "store,flow,valu,load,alu,debug").split(",") if x)
    while done < n:
        used = {engine: 0 for engine in heaps}
        progress = True
        while progress:
            progress = False
            for engine in order:
                if engine not in heaps or used[engine] >= SLOT_LIMITS[engine]: continue
                heap = heaps[engine]
                if not heap: continue
                _, i = heapq.heappop(heap)
                # Child already has a latest bound from all scheduled users.
                latest_i = latest[i]
                if latest_i < cycle:
                    cycle = latest_i
                scheduled[i] = cycle; used[engine] += 1; done += 1; progress = True
                for parent, lag in ops[i].parents.items():
                    latest[parent] = min(latest[parent], cycle - lag)
                    outdegree[parent] -= 1
                    if outdegree[parent] == 0: push(parent)
        if not progress:
            cycle -= 1
    low = min(scheduled)
    if low < 0:
        scheduled = [x - low for x in scheduled]
    validate(ops, scheduled)
    makespan = max(scheduled) + 1
    out = Path(os.environ.get("OUT", "/tmp/aopt-backward-hints.json"))
    out.write_text(json.dumps({"cycles": scheduled, "horizon": makespan}))
    print(f"makespan={makespan} low={low} jobs={n} output={out}")


if __name__ == "__main__": main()
