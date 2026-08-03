"""Topological global-list scheduler for compiler experiments.

The normal kernel scheduler chooses one ready queue per engine.  That is fast,
but independent engine queues can produce resource-order cycles when a nearly
saturated candidate is handed to a strict global solver.  This diagnostic
instead chooses one *global* topological order and then assigns each engine's
capacity lanes in that order.  It is a legal schedule by construction and is
useful as a teacher/seed even when it is not yet competitive.

The resource recurrence for an engine of width ``w`` is
``start[i] >= start[queue[-w]] + 1``.  Thus consecutive operations may occupy
the same cycle, while every operation is separated from the operation ``w``
positions earlier by one cycle.
"""

from __future__ import annotations

from collections import Counter
import heapq
import json
import os
from pathlib import Path
import random

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
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


def graph_data(ops):
    children = [[] for _ in ops]
    indegree = [0] * len(ops)
    for child, op in enumerate(ops):
        indegree[child] = len(op.parents)
        for parent, lag in op.parents.items():
            children[parent].append((child, lag))
    topo = []
    queue = [i for i, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(queue)
    while queue:
        parent = heapq.heappop(queue)
        topo.append(parent)
        for child, _ in children[parent]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(queue, child)
    if len(topo) != len(ops):
        raise ValueError("kernel DAG is cyclic")
    tail = [0] * len(ops)
    for parent in reversed(topo):
        tail[parent] = max(
            (lag + tail[child] for child, lag in children[parent]), default=0
        )
    return children, topo, tail


def global_order(ops, children, topo, tail, seed: int, mode: int):
    """Choose one topological order using a deterministic ready priority."""
    rng = random.Random(seed)
    indegree = [len(op.parents) for op in ops]
    ready = set(i for i, degree in enumerate(indegree) if degree == 0)
    # Ready ties are intentionally randomized only in the final key.  All
    # structural keys remain explicit so a good seed is reproducible.
    noise = [rng.randrange(-1000, 1001) for _ in ops]
    engine_rank = {"load": 0, "valu": 1, "alu": 2, "flow": 3, "store": 4, "debug": 5}

    def key(i: int):
        op = ops[i]
        group = op.group if op.group >= 0 else -1
        if mode == 0:
            return (-tail[i], engine_rank.get(op.engine, 9), group, op.round, -i)
        if mode == 1:
            return (-tail[i], group // 4, engine_rank.get(op.engine, 9), op.round, -i)
        if mode == 2:
            return (-tail[i], -op.round, engine_rank.get(op.engine, 9), -group, -i)
        if mode == 3:
            return (-tail[i], engine_rank.get(op.engine, 9), -group, -op.round, noise[i])
        return (-tail[i] + noise[i] // 100, engine_rank.get(op.engine, 9), group, op.round, noise[i])

    order = []
    while ready:
        parent = min(ready, key=key)
        ready.remove(parent)
        order.append(parent)
        for child, _ in children[parent]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.add(child)
    if len(order) != len(ops):
        raise ValueError("ready walk did not cover DAG")
    return order


def schedule_order(ops, children, order):
    cycles = [-1] * len(ops)
    per_engine = {engine: [] for engine in SLOT_LIMITS}
    for index in order:
        op = ops[index]
        release = max(
            (cycles[parent] + lag for parent, lag in op.parents.items()),
            default=0,
        )
        prior = per_engine[op.engine]
        if len(prior) >= SLOT_LIMITS[op.engine]:
            release = max(release, cycles[prior[-SLOT_LIMITS[op.engine]]] + 1)
        cycles[index] = release
        prior.append(index)
    return cycles


def validate(ops, cycles, horizon=None):
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            if cycles[child] < cycles[parent] + lag:
                return False
    usage = Counter((op.engine, cycles[i]) for i, op in enumerate(ops))
    if any(count > SLOT_LIMITS[engine] for (engine, _), count in usage.items()):
        return False
    makespan = max(cycles, default=-1) + 1
    return makespan if horizon is None or makespan <= horizon else False


def main():
    ops = build_ops()
    children, topo, tail = graph_data(ops)
    trials = int(os.environ.get("GLOBAL_TRIALS", "32"))
    best = None
    best_cycles = None
    for trial in range(trials):
        mode = trial % 5
        order = global_order(ops, children, topo, tail, trial + int(os.environ.get("GLOBAL_SEED", "1")), mode)
        cycles = schedule_order(ops, children, order)
        score = validate(ops, cycles)
        if score and (best is None or score < best):
            best, best_cycles = score, cycles
            print(f"trial={trial} mode={mode} makespan={score}", flush=True)
    print(f"global_best={best} ops={len(ops)}", flush=True)
    if best_cycles is None:
        return
    output = os.environ.get("OUT", "")
    if output:
        payload = {
            "engine": None,
            "horizon": best,
            "cycles": best_cycles,
        }
        Path(output).write_text(json.dumps(payload))
        print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
