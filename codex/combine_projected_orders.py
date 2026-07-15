"""Combine independently feasible projected-engine schedules.

Each projected schedule fixes only the relative order on one VLIW engine.
For an engine with capacity ``k``, adding a unit-lag edge from item ``i`` to
item ``i + k`` turns that order into ``k`` resource lanes.  If the union of
those edges is acyclic, its longest-path labels are a legal global schedule
for every covered engine.  This is a cheap compatibility test before asking
CP-SAT to rediscover the same per-engine decisions in the full 20k-op graph.
"""

from __future__ import annotations

from collections import Counter
import heapq
import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops, validate
from problem import SLOT_LIMITS


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

    hint_paths = [
        Path(value)
        for value in os.environ["HINTS"].split(",")
        if value
    ]
    parents = [dict(op.parents) for op in ops]
    edge_engine: dict[tuple[int, int], str] = {}
    covered: set[str] = set()
    for path in hint_paths:
        payload = json.loads(path.read_text())
        engine = payload["engine"]
        if engine in covered:
            raise ValueError(f"duplicate projected engine: {engine}")
        covered.add(engine)
        cycles = {int(index): int(cycle) for index, cycle in payload["cycles"].items()}
        selected = {i for i, op in enumerate(ops) if op.engine == engine}
        if set(cycles) != selected:
            raise ValueError(
                f"{path} covers {len(cycles)} {engine} ops; expected {len(selected)}"
            )
        capacity = SLOT_LIMITS[engine]
        sequence = sorted(selected, key=lambda index: (cycles[index], index))
        for previous, current in zip(sequence, sequence[capacity:]):
            if cycles[previous] >= cycles[current]:
                raise ValueError(
                    f"{path} exceeds {engine} capacity at cycle {cycles[current]}"
                )
            if parents[current].get(previous, 0) < 1:
                parents[current][previous] = 1
                edge_engine[previous, current] = engine

    children: list[list[tuple[int, int]]] = [[] for _ in ops]
    indegree = [0] * len(ops)
    for child, child_parents in enumerate(parents):
        indegree[child] = len(child_parents)
        for parent, lag in child_parents.items():
            children[parent].append((child, lag))

    ready = [index for index, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        parent = heapq.heappop(ready)
        order.append(parent)
        for child, _ in children[parent]:
            indegree[child] -= 1
            if not indegree[child]:
                heapq.heappush(ready, child)
    if len(order) != len(ops):
        remaining = [i for i, degree in enumerate(indegree) if degree]
        counts = Counter(ops[i].engine for i in remaining)
        resource_counts = Counter(
            engine
            for (parent, child), engine in edge_engine.items()
            if parent in set(remaining) and child in set(remaining)
        )
        print(
            f"status=CYCLIC covered={sorted(covered)} remaining={len(remaining)} "
            f"engines={dict(counts)} resource_edges={dict(resource_counts)}"
        )
        return

    cycles = [0] * len(ops)
    reason = [-1] * len(ops)
    for child in order:
        for parent, lag in parents[child].items():
            candidate = cycles[parent] + lag
            if candidate > cycles[child]:
                cycles[child] = candidate
                reason[child] = parent
    makespan = max(cycles) + 1

    usage = Counter(
        (cycles[i], op.engine)
        for i, op in enumerate(ops)
        if op.engine != "debug"
    )
    overloads = Counter()
    for (cycle, engine), count in usage.items():
        overloads[engine] = max(overloads[engine], count - SLOT_LIMITS[engine])
    print(
        f"status=ACYCLIC covered={sorted(covered)} makespan={makespan} "
        f"overloads={dict(overloads)}"
    )

    endpoint = max(range(len(ops)), key=cycles.__getitem__)
    chain: list[int] = []
    node = endpoint
    while node >= 0:
        chain.append(node)
        node = reason[node]
    chain.reverse()
    kinds = Counter(
        edge_engine.get((left, right), "dag")
        for left, right in zip(chain, chain[1:])
    )
    print(f"critical_chain={len(chain)} edge_kinds={dict(kinds)}")
    print(
        "critical_tags="
        + repr(Counter(ops[i].tag for i in chain).most_common(24))
    )
    print(
        "critical_rounds="
        + repr(Counter(ops[i].round for i in chain).most_common(20))
    )
    if bool(int(os.environ.get("PRINT_CHAIN", "0"))):
        for i in chain[-int(os.environ.get("CHAIN_TAIL", "160")):]:
            op = ops[i]
            parent = reason[i]
            print(
                f"c={cycles[i]:4d} i={i:5d} via="
                f"{edge_engine.get((parent, i), 'dag'):5s} "
                f"{op.engine:5s} g={op.group:2d} r={op.round:2d} {op.tag}"
            )

    if all(overloads[engine] <= 0 for engine in overloads):
        validate(ops, cycles)
        output = Path(os.environ.get("OUT", "/tmp/aopt-combined-orders.json"))
        output.write_text(json.dumps({"makespan": makespan, "cycles": cycles}))
        print(f"output={output}")


if __name__ == "__main__":
    main()
