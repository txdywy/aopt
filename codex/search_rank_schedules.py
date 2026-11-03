"""Search legal full-DAG list calendars with a source-global rank teacher.

The ordinary phase teacher stores only an issue cycle.  That loses the
relative order of operations from different engines that shared a source
cycle, exactly where a saturated VLIW calendar needs a stable cross-engine
tie-break.  This research scheduler rebuilds the default graph first, maps
semantic operations into the target graph, and gives every target operation a
source-global rank (or a conservative tag/group median for newly introduced
operations).  Every trial is still a strict DAG/capacity schedule.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import gc
import heapq
import json
import os
from pathlib import Path
import random
import statistics

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, op_key, real_tail_ops, validate
from problem import SLOT_LIMITS


ENGINE_ORDER = ("load", "valu", "alu", "flow", "store", "debug")
ENGINE_RANK = {name: i for i, name in enumerate(ENGINE_ORDER)}


def operation_ids(ops):
    counts = Counter()
    result = []
    for op in ops:
        key = op_key(op)
        identity = (key, counts[key])
        counts[key] += 1
        result.append(identity)
    return result


def build_source():
    # The target environment has not been applied yet.  The module defaults
    # therefore expose the embedded 987-cycle source graph.
    kernel.SCHEDULE_EXACT_CYCLES = None
    builder = kernel.KernelBuilder()
    # Building without the exact schedule asks the default graph to use its
    # normal scheduler; this is only used for operation identities.  Source
    # cycle values come from the frozen embedded payload below.
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except Exception:
        if not hasattr(builder, "dag_ops"):
            raise
    return real_tail_ops(builder.dag_ops)


def graph_data(ops):
    children = [[] for _ in ops]
    indegree = [len(op.parents) for op in ops]
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            children[parent].append((child, int(lag)))
    ready = [i for i, d in enumerate(indegree) if d == 0]
    heapq.heapify(ready)
    topo = []
    while ready:
        parent = heapq.heappop(ready)
        topo.append(parent)
        for child, _ in children[parent]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(topo) != len(ops):
        raise ValueError("kernel DAG is cyclic")
    earliest = [0] * len(ops)
    for child in topo:
        for parent, lag in ops[child].parents.items():
            earliest[child] = max(earliest[child], earliest[parent] + lag)
    tail = [0] * len(ops)
    reach = [0] * len(ops)
    for parent in reversed(topo):
        tail[parent] = max((lag + tail[c] for c, lag in children[parent]), default=0)
        reach[parent] = min(
            1_000_000,
            len(children[parent]) + sum(reach[c] for c, _ in children[parent]),
        )
    return children, topo, earliest, tail, reach


def build_rank(source_ops, target_ops, source_cycles):
    """Map source operation order and interpolate ranks for new operations."""
    source_ids = operation_ids(source_ops)
    source_index = {identity: i for i, identity in enumerate(source_ids)}
    # A source cycle has no semantic ordering between engines.  The engine
    # order below matches the production list scheduler's first policy and
    # gives a deterministic, useful cross-engine tie break.
    source_rank = {
        i: (int(source_cycles[i]) * 64 + ENGINE_RANK.get(op.engine, 63)) * 65536 + i
        for i, op in enumerate(source_ops)
    }
    by_exact = defaultdict(list)
    by_group_round = defaultdict(list)
    by_tag = defaultdict(list)
    by_engine = defaultdict(list)
    for i, op in enumerate(source_ops):
        rank = source_rank[i]
        by_exact[(op.engine, op.tag, op.group, op.round)].append(rank)
        by_group_round[(op.engine, op.group, op.round)].append(rank)
        by_tag[(op.engine, op.tag)].append(rank)
        by_engine[op.engine].append(rank)

    target_ids = operation_ids(target_ops)
    mapped = {}
    rank = [None] * len(target_ops)
    for i, identity in enumerate(target_ids):
        source = source_index.get(identity)
        if source is not None:
            mapped[i] = source
            rank[i] = source_rank[source]
            continue
        op = target_ops[i]
        candidates = (
            by_exact.get((op.engine, op.tag, op.group, op.round))
            or by_group_round.get((op.engine, op.group, op.round))
            or by_tag.get((op.engine, op.tag))
            or by_engine.get(op.engine)
        )
        rank[i] = int(statistics.median(candidates)) if candidates else i * 65536

    # New nodes are clamped between mapped DAG boundaries when possible.  The
    # pass is intentionally soft: rank remains a tie-break, so the scheduler
    # can move a node outside an inconsistent source interval for legality.
    children, topo, _, _, _ = graph_data(target_ops)
    for i in topo:
        if i in mapped:
            continue
        lower = max((rank[p] + 1 for p in target_ops[i].parents), default=0)
        mapped_child_ranks = [rank[c] for c, _ in children[i] if c in mapped]
        upper = min(mapped_child_ranks, default=10**18)
        if lower <= upper:
            rank[i] = max(lower, min(rank[i], upper))
        else:
            rank[i] = max(rank[i], lower)
    return rank, len(mapped)


def schedule(ops, children, earliest, tail, reach, rank, target, weights, rng):
    # Weights: source-rank, tail, reach, group, round, ready, deadline,
    # engine.  Rank is divided by 65536 so one source cycle is comparable to
    # a critical-path height, while the low bits still break equal phases.
    wrank, wtail, wreach, wgroup, wround, wready, wdeadline, wengine = weights
    n = len(ops)
    # Debug markers are zero-time bookkeeping operations in the challenge;
    # keeping their 64-wide queue makes the result a complete assignment
    # without affecting any scored engine.
    engine_names = tuple(SLOT_LIMITS)
    engine_rank = {e: i for i, e in enumerate(engine_names)}
    noise = [rng.randrange(-4, 5) for _ in ops]
    static = [0] * n
    for i, op in enumerate(ops):
        group = op.group if op.group >= 0 else -1
        rnd = op.round if op.round >= 0 else -1
        static[i] = (
            wrank * (rank[i] // 65536)
            + wtail * tail[i]
            + wreach * (reach[i] // 32)
            + wgroup * group
            + wround * rnd
            + wdeadline * (target - 1 - tail[i])
            + wengine * engine_rank.get(op.engine, 5)
        )
    remaining = [len(op.parents) for op in ops]
    ready_at = [0] * n
    heaps = {e: [] for e in engine_names}
    future = defaultdict(list)
    result = [-1] * n

    def push(i):
        key = (
            static[i] + wready * ready_at[i] + noise[i],
            rank[i],
            -tail[i],
            -reach[i],
            i,
        )
        heapq.heappush(heaps[ops[i].engine], (key, i))

    for i, degree in enumerate(remaining):
        if degree == 0:
            push(i)
    order = list(engine_names)
    rng.shuffle(order)
    done = 0
    cycle = 0
    while done < n:
        for i in future.pop(cycle, ()):
            push(i)
        used = {e: 0 for e in engine_names}
        progress = False
        while True:
            pass_progress = False
            for engine in order:
                if used[engine] >= SLOT_LIMITS[engine]:
                    continue
                heap = heaps[engine]
                chosen = None
                deferred = []
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
                result[chosen] = cycle
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
    validate(ops, result)
    return result


def main():
    source = build_source()
    source_cycles = json.loads(
        Path(os.environ.get("SOURCE", ".search959/current-default-embedded.json")).read_text()
    )["cycles"]
    if isinstance(source_cycles, dict):
        source_cycles = [int(source_cycles[str(i)]) for i in range(len(source))]
    configure_target()
    kernel.SCHEDULE_EXACT_CYCLES = None
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except Exception:
        if not hasattr(builder, "dag_ops"):
            raise
    target_ops = real_tail_ops(builder.dag_ops)
    children, _, earliest, tail, reach = graph_data(target_ops)
    rank, mapped = build_rank(source, target_ops, source_cycles)
    print(f"source_ops={len(source)} target_ops={len(target_ops)} mapped={mapped}", flush=True)
    target = int(os.environ.get("TARGET", "959"))
    center_raw = os.environ.get("WEIGHT_CENTER", "")
    center = tuple(int(x) for x in center_raw.split(",") if x)
    if center and len(center) != 8:
        raise ValueError("WEIGHT_CENTER needs 8 integers")
    radius = int(os.environ.get("WEIGHT_RADIUS", "16"))
    trials = int(os.environ.get("TRIALS", "256"))
    rng = random.Random(int(os.environ.get("SEED", "1")))
    best = None
    best_cycles = None
    for trial in range(trials):
        weights = (
            tuple(rng.randint(x - radius, x + radius) for x in center)
            if center else tuple(rng.randint(-16, 16) for _ in range(8))
        )
        cycles = schedule(
            target_ops, children, earliest, tail, reach, rank, target, weights, rng
        )
        score = max(cycles) + 1
        if best is None or score < best:
            best, best_cycles = score, cycles
            print(f"best={best} trial={trial} weights={weights}", flush=True)
        if trial % 16 == 0:
            gc.collect()
    output = Path(os.environ.get("OUT", ".search959/rank-best.json"))
    output.write_text(json.dumps({"makespan": best, "cycles": best_cycles, "mapped": mapped}))
    print(f"trials={trials} best={best} output={output}", flush=True)


if __name__ == "__main__":
    main()
