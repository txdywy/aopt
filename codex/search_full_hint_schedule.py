"""Run the production list scheduler with all-engine 959 calendars as a guide.

The exact projected solvers give a legal issue calendar for each individual
engine, but their union can be cyclic.  This tool does not freeze that union.
Instead it feeds every engine's target phase into the production full-DAG list
scheduler as a *soft* priority, preserving its mature zero-lag and packet
handling while exploring the compatible global extensions that it normally
would not consider.

It is a fast source of fully legal schedules for the subsequent exact local
repair/search; unlike an independently merged resource order, every emitted
schedule is checked against the effective full DAG and all capacities.
"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import random

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops, validate


def csv_ints(name: str, default: str) -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in os.environ.get(name, default).split(",")
        if value
    )


def noise_value(index: int, seed: int, amplitude: int) -> int:
    """Stable, process-independent priority perturbation."""
    word = (seed + 0x9E3779B9 * (index + 1)) & 0xFFFFFFFF
    word ^= word >> 16
    word = (word * 0x7FEB352D) & 0xFFFFFFFF
    word ^= word >> 15
    word = (word * 0x846CA68B) & 0xFFFFFFFF
    word ^= word >> 16
    return word % (2 * amplitude + 1) - amplitude


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

    targets: dict[int, int] = {}
    for raw_path in os.environ.get("HINTS", "").split(","):
        if not raw_path:
            continue
        payload = json.loads(Path(raw_path).read_text())
        engine = str(payload["engine"])
        cycles = {int(index): int(cycle) for index, cycle in payload["cycles"].items()}
        expected = {index for index, op in enumerate(ops) if op.engine == engine}
        if set(cycles) != expected:
            raise ValueError(f"{raw_path}: incomplete {engine} guide")
        overlap = set(targets) & set(cycles)
        if overlap:
            raise ValueError(f"duplicate guide job {min(overlap)}")
        targets.update(cycles)
    for index, op in enumerate(ops):
        if index not in targets:
            if op.engine != "debug":
                raise ValueError(f"HINTS omits non-debug op {index}")
            targets[index] = 0

    cycle_weights = csv_ints("CYCLE_WEIGHTS", "1,2,4,8,16")
    height_weights = csv_ints("HEIGHT_WEIGHTS", "1,2,4,8,16")
    policies = csv_ints("POLICIES", "0,1,2,3")
    amplitudes = csv_ints("AMPLITUDES", "0,1,2,4")
    seeds = int(os.environ.get("SEEDS", "16"))
    start = int(os.environ.get("START", "0"))
    limit = int(os.environ.get("LIMIT", "0"))
    output = Path(os.environ.get("OUT", "/tmp/aopt-full-hint-best.json"))

    best: tuple[int, tuple[int, int, int, int, int], list[int]] | None = None
    evaluated = 0
    for cycle_weight in cycle_weights:
        for height_weight in height_weights:
            for policy in policies:
                for amplitude in amplitudes:
                    for seed in range(start, start + seeds):
                        if limit and evaluated >= limit:
                            break
                        scores = [
                            -cycle_weight * targets[index]
                            + noise_value(index, seed, amplitude)
                            for index in range(len(ops))
                        ]
                        _, cycles = builder._schedule(
                            builder.dag_ops,
                            policy,
                            return_cycles=True,
                            external_scores=scores,
                            height_weight=height_weight,
                        )
                        validate(ops, cycles)
                        makespan = max(cycles) + 1
                        evaluated += 1
                        key = (makespan, cycle_weight, height_weight, policy, amplitude, seed)
                        if best is None or key < (best[0], *best[1]):
                            best = (
                                makespan,
                                (cycle_weight, height_weight, policy, amplitude, seed),
                                cycles,
                            )
                            print(
                                f"best={makespan} cycle_weight={cycle_weight} "
                                f"height_weight={height_weight} policy={policy} "
                                f"amplitude={amplitude} seed={seed}",
                                flush=True,
                            )
                    if limit and evaluated >= limit:
                        break
                if limit and evaluated >= limit:
                    break
            if limit and evaluated >= limit:
                break
        if limit and evaluated >= limit:
            break
    if best is None:
        raise ValueError("no scheduling trials requested")
    makespan, params, cycles = best
    usage = Counter((cycle, op.engine) for cycle, op in zip(cycles, ops))
    holes = {
        engine: sum(
            kernel.SLOT_LIMITS[engine] - usage[cycle, engine]
            for cycle in range(makespan)
        )
        for engine in kernel.SLOT_LIMITS
        if engine != "debug"
    }
    output.write_text(
        json.dumps(
            {
                "makespan": makespan,
                "params": {
                    "cycle_weight": params[0],
                    "height_weight": params[1],
                    "policy": params[2],
                    "amplitude": params[3],
                    "seed": params[4],
                },
                "cycles": cycles,
            }
        )
    )
    print(
        f"evaluated={evaluated} best={makespan} holes={holes} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
