"""Compress one engine's order/gap profile from a full schedule into a target."""

from __future__ import annotations

import heapq
import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops


def shape_cycles(
    ops: list[kernel._Op],
    source: list[int],
    engine: str,
    horizon: int,
    gap_scale: float,
) -> list[int]:
    engine_order = sorted(
        (index for index, op in enumerate(ops) if op.engine == engine),
        key=lambda index: (source[index], index),
    )
    first_cycle = source[engine_order[0]]
    last_cycle = source[engine_order[-1]]
    span = max(1, last_cycle - first_cycle)
    shaped = [
        round((source[index] - first_cycle) * (horizon - 1) / span)
        for index in engine_order
    ]
    for position in range(1, len(shaped)):
        shaped[position] = max(shaped[position], shaped[position - 1] + 1)
    for position in reversed(range(len(shaped) - 1)):
        shaped[position] = min(shaped[position], shaped[position + 1] - 1)

    parents = [dict(op.parents) for op in ops]
    for position in range(1, len(engine_order)):
        previous = engine_order[position - 1]
        current = engine_order[position]
        source_gap = max(1, shaped[position] - shaped[position - 1])
        lag = 1 + round(gap_scale * (source_gap - 1))
        parents[current][previous] = max(parents[current].get(previous, 0), lag)
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
        raise ValueError("engine order introduced a cycle")
    cycles = [0] * len(ops)
    for child in order:
        cycles[child] = max(
            (cycles[parent] + lag for parent, lag in parents[child].items()),
            default=0,
        )
    return cycles


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
    source_path = Path(os.environ["SOURCE"])
    source = json.loads(source_path.read_text())["cycles"]
    engine = os.environ.get("ENGINE", "flow")
    horizon = int(os.environ.get("TARGET", "959"))
    gap_scale = float(os.environ.get("GAP_SCALE", "1"))

    engine_order = sorted(
        (index for index, op in enumerate(ops) if op.engine == engine),
        key=lambda index: (source[index], index),
    )
    first_cycle = source[engine_order[0]]
    last_cycle = source[engine_order[-1]]
    span = max(1, last_cycle - first_cycle)
    shaped = [
        round((source[index] - first_cycle) * (horizon - 1) / span)
        for index in engine_order
    ]
    # Enforce strictly increasing target slots in both directions.
    for position in range(1, len(shaped)):
        shaped[position] = max(shaped[position], shaped[position - 1] + 1)
    for position in reversed(range(len(shaped) - 1)):
        shaped[position] = min(shaped[position], shaped[position + 1] - 1)

    parents = [dict(op.parents) for op in ops]
    for position in range(1, len(engine_order)):
        previous = engine_order[position - 1]
        current = engine_order[position]
        source_gap = max(1, shaped[position] - shaped[position - 1])
        lag = 1 + round(gap_scale * (source_gap - 1))
        parents[current][previous] = max(parents[current].get(previous, 0), lag)

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
        raise ValueError("engine order introduced a cycle")
    cycles = [0] * len(ops)
    for child in order:
        cycles[child] = max(
            (cycles[parent] + lag for parent, lag in parents[child].items()),
            default=0,
        )
    makespan = max(cycles) + 1
    output = Path(os.environ.get("OUT", "/tmp/aopt-shaped-engine.json"))
    output.write_text(
        json.dumps(
            {
                "makespan": makespan,
                "engine": engine,
                "gap_scale": gap_scale,
                "cycles": cycles,
            }
        )
    )
    print(
        f"engine={engine} gap_scale={gap_scale} makespan={makespan} "
        f"output={output}"
    )


if __name__ == "__main__":
    main()
