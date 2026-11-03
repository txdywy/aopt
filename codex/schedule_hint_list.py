"""Fast full-DAG list scheduler driven by per-engine projected hints."""

from __future__ import annotations

from collections import defaultdict
import heapq
import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from problem import SLOT_LIMITS


def main() -> None:
    configure_target(); kernel.SCHEDULE_EXACT_CYCLES = None
    b = kernel.KernelBuilder()
    try: b.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        if not hasattr(b, "dag_ops"): raise
    ops = real_tail_ops(b.dag_ops); n = len(ops)
    hint = {}
    for raw in os.environ.get("HINTS", "").split(","):
        if not raw: continue
        p = json.loads(Path(raw).read_text())
        cycles = p["cycles"]
        if isinstance(cycles, list):
            hint.update({i: int(c) for i, c in enumerate(cycles)})
        else:
            hint.update({int(i): int(c) for i, c in cycles.items()})
    children = [[] for _ in ops]; indegree = [len(op.parents) for op in ops]
    earliest = [0] * n
    for c, op in enumerate(ops):
        for p, lag in op.parents.items(): children[p].append((c, int(lag)))
    tail = [0] * n
    # exact topological tail for criticality
    indeg = indegree[:]; q=[i for i,d in enumerate(indeg) if not d]; heapq.heapify(q); topo=[]
    while q:
        p=heapq.heappop(q); topo.append(p)
        for c,_ in children[p]:
            indeg[c]-=1
            if indeg[c]==0: heapq.heappush(q,c)
    for p in reversed(topo): tail[p]=max((lag+tail[c] for c,lag in children[p]), default=0)
    base_target = int(os.environ.get("TARGET", "959"))
    hint_weight = int(os.environ.get("HINT_WEIGHT", "8"))
    tail_weight = int(os.environ.get("TAIL_WEIGHT", "1"))
    # Use the same refill semantics as the production list scheduler.  A
    # one-shot batch leaves capacity idle whenever a lag-0 child becomes ready
    # after its parent; those zero-latency edges are legal in the challenge and
    # are common in the setup/branch trace.  Keeping a heap per engine also
    # avoids repeatedly scanning all 20k operations at every cycle.
    ready_at = earliest[:]
    heaps: dict[str, list[tuple[tuple[int, ...], int]]] = {
        engine: [] for engine in SLOT_LIMITS
    }
    future: dict[int, list[int]] = defaultdict(list)
    scheduled: dict[int, int] = {}
    engine_order = tuple(
        value
        for value in os.environ.get(
            "ENGINE_ORDER", "load,valu,alu,flow,store,debug"
        ).split(",")
        if value
    )
    height_weight = int(os.environ.get("HEIGHT_WEIGHT", "1"))
    reach_weight = int(os.environ.get("REACH_WEIGHT", "0"))
    hint_default = int(os.environ.get("HINT_DEFAULT", str(base_target)))
    # Optional structural bias for a latency-critical group/round cone.  A
    # static source calendar can be badly misleading after lowering adds
    # scalar repair nodes; explicitly pulling the final group through the
    # ready queues is a cheap way to test whether the resource order, rather
    # than the DAG itself, is the remaining bottleneck.  Format:
    # ``31:1000,30:200`` or ``31/14:1000`` (group or group/round).
    critical_bias: dict[tuple[int, int | None], int] = {}
    for raw in os.environ.get("CRITICAL_BIAS", "").split(","):
        if not raw or ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        try:
            if "/" in key:
                group_text, round_text = key.split("/", 1)
                critical_bias[(int(group_text), int(round_text))] = int(value)
            else:
                critical_bias[(int(key), None)] = int(value)
        except ValueError:
            continue
    tail_bias = int(os.environ.get("TAIL_BIAS", "0"))
    priority_mode = os.environ.get("PRIORITY_MODE", "hint")
    deadline_weight = int(os.environ.get("DEADLINE_WEIGHT", "0"))
    earliest_weight = int(os.environ.get("EARLIEST_WEIGHT", "0"))
    noise_seed = int(os.environ.get("NOISE_SEED", "0"))
    noise_amplitude = int(os.environ.get("NOISE_AMPLITUDE", "0"))
    noise = [0] * n
    if noise_amplitude:
        # Stable integer mixer: no process-dependent hash() and no RNG state
        # shared with the kernel builder.  Noise is deliberately a late tie
        # breaker so it explores equal-phase resource order without erasing
        # the semantic calendar.
        for index in range(n):
            word = (noise_seed + 0x9E3779B9 * (index + 1)) & 0xFFFFFFFF
            word ^= word >> 16
            word = (word * 0x7FEB352D) & 0xFFFFFFFF
            word ^= word >> 15
            word = (word * 0x846CA68B) & 0xFFFFFFFF
            word ^= word >> 16
            noise[index] = word % (2 * noise_amplitude + 1) - noise_amplitude
    reach = [0] * n
    for parent in reversed(topo):
        reach[parent] = min(
            1_000_000,
            len(children[parent])
            + sum(reach[child] for child, _ in children[parent]),
        )

    def priority(index: int) -> tuple[int, ...]:
        op = ops[index]
        group_bias = critical_bias.get((op.group, op.round), 0)
        group_bias += critical_bias.get((op.group, None), 0)
        group_bias += tail_bias * (tail[index] // 8)
        # Lower tuple values are preferred by the heap.  The hint term is a
        # static phase teacher; tail/reach terms keep a delayed critical cone
        # from being starved by a dense early calendar.
        h = hint.get(index, hint_default)
        deadline = base_target - 1 - tail[index]
        # A phase teacher is useful for preserving a proven cross-engine
        # calendar, but at a tight horizon a pure phase can starve a late
        # consumer whose producer was delayed by a different resource.  The
        # optional deadline/earliest modes make the same scheduler act like a
        # least-slack or ASAP list scheduler without changing legality.
        if priority_mode == "deadline":
            return (
                deadline_weight * deadline
                - group_bias
                - height_weight * tail[index]
                - reach_weight * (reach[index] // 32),
                deadline,
                h,
                -tail[index],
                -reach[index],
                noise[index],
                index,
            )
        if priority_mode == "earliest":
            return (
                earliest_weight * ready_at[index]
                + hint_weight * h
                - group_bias
                - height_weight * tail[index]
                - reach_weight * (reach[index] // 32),
                ready_at[index],
                h,
                -tail[index],
                -reach[index],
                noise[index],
                index,
            )
        if priority_mode == "slack":
            return (
                deadline_weight * (deadline - h)
                + hint_weight * h
                - group_bias
                - height_weight * tail[index]
                - reach_weight * (reach[index] // 32),
                deadline - h,
                h,
                -tail[index],
                -reach[index],
                noise[index],
                index,
            )
        return (
            hint_weight * h - group_bias
            - height_weight * tail[index]
            - reach_weight * (reach[index] // 32),
            h,
            -tail[index],
            -reach[index],
            noise[index],
            index,
        )

    def push(index: int) -> None:
        heapq.heappush(heaps[ops[index].engine], (priority(index), index))

    for index, degree in enumerate(indegree):
        if degree == 0:
            push(index)

    cycle = 0
    while len(scheduled) < n:
        for index in future.pop(cycle, ()):
            push(index)
        used = {engine: 0 for engine in heaps}
        made_progress = True
        while made_progress:
            made_progress = False
            for engine in engine_order:
                if engine not in heaps or used[engine] >= SLOT_LIMITS.get(engine, 0):
                    continue
                heap = heaps[engine]
                deferred = []
                chosen = None
                while heap:
                    item = heapq.heappop(heap)
                    candidate = item[1]
                    if ready_at[candidate] <= cycle:
                        chosen = candidate
                        break
                    deferred.append(item)
                for item in deferred:
                    heapq.heappush(heap, item)
                if chosen is None:
                    continue
                scheduled[chosen] = cycle
                used[engine] += 1
                made_progress = True
                for child, lag in children[chosen]:
                    indegree[child] -= 1
                    ready_at[child] = max(ready_at[child], cycle + lag)
                    if indegree[child] == 0:
                        if ready_at[child] <= cycle:
                            push(child)
                        else:
                            future[ready_at[child]].append(child)
        if not made_progress:
            next_cycle = min(future, default=cycle + 1)
            cycle = max(cycle + 1, next_cycle)
        else:
            cycle += 1
    makespan=max(scheduled.values())+1
    print(f"makespan={makespan} jobs={n} weights={hint_weight}:{tail_weight}")
    out=Path(os.environ.get("OUT", ".search959/list-hint-full.json"))
    out.write_text(json.dumps({"cycles":[scheduled[i] for i in range(n)],"horizon":makespan}))
    print(f"output={out}")


if __name__ == "__main__": main()
