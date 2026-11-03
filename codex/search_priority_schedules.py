"""Search a compact family of full-DAG list-scheduler priorities.

This is an offline scheduler oracle, not part of the submission path.  The
important difference from the historical policy sweep is that all engines use
one shared, randomized feature vector: source phase, exact DAG tail, ready
release, group/round locality, and downstream fanout.  A trial is still a
strict topological/capacity schedule, so every accepted result is a real
candidate rather than a projected score.
"""

from __future__ import annotations

from collections import defaultdict
import heapq
import json
import os
from pathlib import Path
import random

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops, validate
from problem import SLOT_LIMITS


def load_hint(path: str, n: int) -> list[int]:
    values = [0] * n
    if not path:
        return values
    payload = json.loads(Path(path).read_text())
    raw = payload.get("cycles")
    if isinstance(raw, list):
        if len(raw) != n:
            raise ValueError(f"{path}: cycle length mismatch")
        return [int(x) for x in raw]
    for index, cycle in raw.items():
        values[int(index)] = int(cycle)
    return values


def build_graph():
    configure_target()
    # Expose the graph without replaying the embedded schedule.  Some research
    # variants cannot color a production bundle yet, but they still expose a
    # complete DAG before that optional pass fails.
    kernel.SCHEDULE_EXACT_CYCLES = []
    builder = kernel.KernelBuilder()
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
    ready = [i for i, degree in enumerate(indegree) if degree == 0]
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
    tail = [0] * len(ops)
    reach = [0] * len(ops)
    for child in topo:
        for parent, lag in ops[child].parents.items():
            earliest[child] = max(earliest[child], earliest[parent] + lag)
    for parent in reversed(topo):
        tail[parent] = max(
            (lag + tail[child] for child, lag in children[parent]), default=0
        )
        reach[parent] = min(
            1_000_000,
            len(children[parent])
            + sum(reach[child] for child, _ in children[parent]),
        )
    return children, earliest, tail, reach


def schedule(ops, children, earliest, tail, reach, hint, horizon, weights, rng):
    """Issue one legal bundle sequence for one static feature vector."""
    # Features are deliberately integer and coarse.  Fine noise breaks large
    # equal-phase classes without making a phase teacher disappear.
    wh, wt, wr, wg, wround, wready, wdeadline, wengine = weights
    engine_names = tuple(SLOT_LIMITS)
    engine_rank = {name: rank for rank, name in enumerate(engine_names)}
    noise = [rng.randrange(-8, 9) for _ in ops]
    static = [0] * len(ops)
    for i, op in enumerate(ops):
        group = op.group if op.group >= 0 else -1
        rnd = op.round if op.round >= 0 else -1
        static[i] = (
            wh * hint[i]
            + wt * tail[i]
            + wr * (reach[i] // 32)
            + wg * group
            + wround * rnd
            + wdeadline * (horizon - 1 - tail[i])
            + wengine * engine_rank.get(op.engine, 0)
        )

    remaining = [len(op.parents) for op in ops]
    ready_at = [0] * len(ops)
    heaps = {engine: [] for engine in engine_names}
    future: dict[int, list[int]] = defaultdict(list)
    result = [-1] * len(ops)

    def push(index: int) -> None:
        # Re-evaluate ready_at at insertion; the heap is otherwise static.
        key = (
            static[index] + wready * ready_at[index] + noise[index],
            hint[index],
            -tail[index],
            -reach[index],
            index,
        )
        heapq.heappush(heaps[ops[index].engine], (key, index))

    for index, degree in enumerate(remaining):
        if degree == 0:
            push(index)

    # Varying this order matters for lag-zero cross-engine diamonds.  It is a
    # small but high-leverage lane-matching decision at a near-saturated cut.
    order = list(engine_names)
    rng.shuffle(order)
    done = 0
    cycle = 0
    while done < len(ops):
        for index in future.pop(cycle, ()):
            push(index)
        used = {engine: 0 for engine in engine_names}
        progress = False
        # Refill until no lag-zero child can use a remaining lane.
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
                    index = item[1]
                    if ready_at[index] <= cycle:
                        chosen = index
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


def main() -> None:
    ops = build_graph()
    children, earliest, tail, reach = graph_data(ops)
    horizon = int(os.environ.get("TARGET", "959"))
    hint_paths = [x for x in os.environ.get("HINTS", "").split(",") if x]
    hints = [0] * len(ops)
    for path in hint_paths:
        incoming = load_hint(path, len(ops))
        for index, value in enumerate(incoming):
            # Per-engine payloads use zero as an absent value.  A full payload
            # is accepted directly; duplicate nonzero phases must agree.
            if value or not any(hints):
                hints[index] = value
    # Normalize an absent teacher to a useful structural phase.
    if not hint_paths:
        hints = earliest[:]
    trials = int(os.environ.get("TRIALS", "256"))
    seed = int(os.environ.get("SEED", "1"))
    rng = random.Random(seed)
    center_raw = os.environ.get("WEIGHT_CENTER", "")
    center = tuple(int(value) for value in center_raw.split(",") if value)
    if center and len(center) != 8:
        raise ValueError("WEIGHT_CENTER must contain eight integers")
    radius = int(os.environ.get("WEIGHT_RADIUS", "16"))
    best = None
    best_cycles = None
    for trial in range(trials):
        # Positive/negative weights are intentional: a good teacher sometimes
        # needs to be traversed from the tail, while another trial should
        # honor it as an ASAP phase.  Keep tail and release terms bounded so
        # no single feature serializes the machine.
        if center:
            weights = tuple(
                rng.randint(value - radius, value + radius) for value in center
            )
        else:
            weights = tuple(rng.randint(-16, 16) for _ in range(8))
        cycles = schedule(
            ops, children, earliest, tail, reach, hints, horizon, weights, rng
        )
        score = max(cycles) + 1
        if best is None or score < best:
            best = score
            best_cycles = cycles
            print(f"best={best} trial={trial} weights={weights}", flush=True)
    if best_cycles is None:
        raise ValueError("no trials")
    output = Path(os.environ.get("OUT", ".search959/priority-best.json"))
    output.write_text(json.dumps({"makespan": best, "cycles": best_cycles}))
    print(f"trials={trials} best={best} output={output}", flush=True)


if __name__ == "__main__":
    main()
