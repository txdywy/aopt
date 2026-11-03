"""Backward, deadline-aware list scheduling for the complete VLIW DAG."""

from __future__ import annotations

from collections import Counter
import heapq
import json
import os
from pathlib import Path
import random

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
    horizon = int(os.environ.get("TARGET", "959"))
    children: list[list[tuple[int, int]]] = [[] for _ in ops]
    earliest = [0] * len(ops)
    ancestor_reach = [0] * len(ops)
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            children[parent].append((child, lag))
            earliest[child] = max(earliest[child], earliest[parent] + lag)
        ancestor_reach[child] = min(
            1_000_000,
            len(op.parents) + sum(ancestor_reach[p] for p in op.parents),
        )
    tail = [0] * len(ops)
    descendant_reach = [0] * len(ops)
    for parent in reversed(range(len(ops))):
        tail[parent] = max(
            (lag + tail[child] for child, lag in children[parent]),
            default=0,
        )
        descendant_reach[parent] = min(
            1_000_000,
            len(children[parent])
            + sum(descendant_reach[child] for child, _ in children[parent]),
        )

    engines = tuple(engine for engine in SLOT_LIMITS if engine != "debug")
    engine_orders = (
        engines,
        ("flow", "valu", "load", "alu", "store"),
        ("valu", "flow", "load", "alu", "store"),
        ("load", "flow", "valu", "alu", "store"),
        ("alu", "load", "valu", "flow", "store"),
    )

    def schedule(seed: int) -> tuple[int, list[int], int]:
        rng = random.Random(seed)
        early_weight = rng.choice((4, 8, 16, 32, 64, 128))
        ancestor_divisor = rng.choice((8, 16, 32, 64, 128, 256, 512))
        fanin_weight = rng.choice((0, 1, 2, 4, 8, 16, 32))
        unlock_weight = rng.choice((0, 8, 16, 32, 64, 128, 256))
        group_weight = rng.choice((-4, -2, -1, 0, 1, 2, 4))
        noise_amplitude = rng.choice((0, 1, 2, 4, 8, 16))
        noise = [
            rng.randrange(-noise_amplitude, noise_amplitude + 1)
            for _ in ops
        ]
        engine_order = engine_orders[seed % len(engine_orders)]
        successor_count = [len(item) for item in children]
        latest_at = [horizon - 1 - value for value in tail]
        future: dict[int, list[int]] = {}
        for i, count in enumerate(successor_count):
            if not count:
                future.setdefault(latest_at[i], []).append(i)
        ready: dict[str, list[tuple[tuple[int, ...], int]]] = {
            engine: [] for engine in engines
        }
        ready_debug: list[int] = []

        def priority(i: int) -> tuple[int, ...]:
            op = ops[i]
            unlock = sum(
                1 + ancestor_reach[parent] // ancestor_divisor
                for parent in op.parents
                if successor_count[parent] == 1
            )
            group = op.group if op.group is not None and op.group >= 0 else -1
            return (
                early_weight * earliest[i]
                + ancestor_reach[i] // ancestor_divisor
                + fanin_weight * len(op.parents)
                + unlock_weight * unlock
                + group_weight * group
                + noise[i],
                earliest[i],
                ancestor_reach[i],
                len(op.parents),
                i,
            )

        def push(i: int) -> None:
            engine = ops[i].engine
            if engine == "debug":
                ready_debug.append(i)
            else:
                p = priority(i)
                heapq.heappush(ready[engine], (tuple(-x for x in p), i))

        cycles = [-1] * len(ops)
        cycle = horizon - 1
        scheduled = 0
        while scheduled < len(ops):
            for available in sorted(
                (value for value in future if value >= cycle), reverse=True
            ):
                for i in future.pop(available):
                    push(i)
            used = {engine: 0 for engine in engines}
            made_progress = True
            while made_progress:
                made_progress = False
                while ready_debug:
                    child = ready_debug.pop()
                    cycles[child] = cycle
                    scheduled += 1
                    made_progress = True
                    for parent, lag in ops[child].parents.items():
                        successor_count[parent] -= 1
                        latest_at[parent] = min(latest_at[parent], cycle - lag)
                        if not successor_count[parent]:
                            if latest_at[parent] >= cycle:
                                push(parent)
                            else:
                                future.setdefault(latest_at[parent], []).append(parent)
                for engine in engine_order:
                    capacity = SLOT_LIMITS[engine]
                    heap = ready[engine]
                    while heap and used[engine] < capacity:
                        _, child = heapq.heappop(heap)
                        cycles[child] = cycle
                        scheduled += 1
                        used[engine] += 1
                        made_progress = True
                        for parent, lag in ops[child].parents.items():
                            successor_count[parent] -= 1
                            latest_at[parent] = min(
                                latest_at[parent], cycle - lag
                            )
                            if not successor_count[parent]:
                                if latest_at[parent] >= cycle:
                                    push(parent)
                                else:
                                    future.setdefault(
                                        latest_at[parent], []
                                    ).append(parent)
            if any(used.values()) or ready_debug:
                cycle -= 1
            elif future:
                cycle = max(future)
            else:
                raise AssertionError("DAG scheduling stalled")
        violation = max(
            (earliest[i] - cycles[i] for i in range(len(ops))), default=0
        )
        shift = max(0, violation)
        if shift:
            cycles = [value + shift for value in cycles]
        validate(ops, cycles)
        return horizon + shift, cycles, violation

    trials = int(os.environ.get("TRIALS", "100"))
    start = int(os.environ.get("START", "0"))
    best: tuple[int, int, list[int], int] | None = None
    for seed in range(start, start + trials):
        score, cycles, violation = schedule(seed)
        candidate = (score, seed, cycles, violation)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
            print(f"score={score} seed={seed} violation={violation}", flush=True)
    assert best is not None
    output = Path(os.environ.get("OUT", "/tmp/aopt-deadline-schedule.json"))
    output.write_text(
        json.dumps(
            {
                "makespan": best[0],
                "seed": best[1],
                "cycles": best[2],
            }
        )
    )
    usage = Counter((cycles, op.engine) for cycles, op in zip(best[2], ops))
    holes = {
        engine: sum(
            SLOT_LIMITS[engine] - usage[cycle, engine]
            for cycle in range(best[0])
        )
        for engine in engines
    }
    print(f"best={best[0]} seed={best[1]} holes={holes} output={output}")


if __name__ == "__main__":
    main()
