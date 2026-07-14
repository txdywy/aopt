"""Randomize a distilled exact-schedule priority without changing the DAG.

The CP-SAT cycle assignment is a strong ordering oracle but cannot be emitted
verbatim because branch-table traces have extra packet-contiguity constraints.
This search keeps the production list scheduler (and therefore all of those
constraints) while perturbing only ties around the oracle order.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel


for env_name, attr in (
    ("SCALAR_FINAL_C5", "SCALAR_FINAL_C5_SET"),
    ("SCALAR_FINAL_JOIN", "SCALAR_FINAL_JOIN_SET"),
    ("SCALAR_FINAL_SHIFT", "SCALAR_FINAL_SHIFT_SET"),
    ("SCALAR_FINAL_HASH23_JOIN", "SCALAR_FINAL_HASH23_JOIN_SET"),
):
    if env_name in os.environ:
        setattr(
            kernel,
            attr,
            frozenset(
                int(value)
                for value in os.environ[env_name].split(",")
                if value
            ),
        )


CYCLES: list[int] = []
HEIGHT_WEIGHT = int(os.environ.get("HEIGHT_WEIGHT", "16"))
CYCLE_WEIGHT = int(os.environ.get("CYCLE_WEIGHT", "16"))


def init_worker(path: str) -> None:
    global CYCLES
    CYCLES = json.loads(Path(path).read_text())["cycles"]


def priority_noise(index: int, seed: int, amplitude: int) -> int:
    word = (seed + 0x9E3779B9 * (index + 1)) & 0xFFFFFFFF
    word ^= word >> 16
    word = (word * 0x7FEB352D) & 0xFFFFFFFF
    word ^= word >> 15
    word = (word * 0x846CA68B) & 0xFFFFFFFF
    word ^= word >> 16
    return word % (2 * amplitude + 1) - amplitude


def evaluate(job: tuple[int, int]) -> tuple[int, int, int]:
    seed, amplitude = job
    kernel.SCHEDULE_EXTERNAL_SCORES = [
        -CYCLE_WEIGHT * cycle + priority_noise(i, seed, amplitude)
        for i, cycle in enumerate(CYCLES)
    ]
    kernel.SCHEDULE_EXTERNAL_HEIGHT_WEIGHT = HEIGHT_WEIGHT
    try:
        builder = kernel.KernelBuilder()
        builder.build_kernel(10, 2047, 256, 16)
        score = next(
            cycle + 1
            for cycle, bundle in enumerate(builder.instrs)
            if ("halt",) in bundle.get("flow", ())
        )
    except (AssertionError, StopIteration, ValueError):
        score = 1_000_000
    return score, seed, amplitude


def main() -> None:
    path = os.environ.get("CYCLES", "/tmp/tail-full-opt992.json")
    seeds = int(os.environ.get("SEEDS", "1000"))
    start = int(os.environ.get("START", "0"))
    amplitudes = tuple(
        int(value) for value in os.environ.get("AMPLITUDES", "1,2,4,8,16").split(",")
    )
    jobs = [(seed, amplitude) for seed in range(start, start + seeds) for amplitude in amplitudes]
    best = (1_000_000, -1, -1)
    with mp.Pool(
        int(os.environ.get("WORKERS", "4")),
        initializer=init_worker,
        initargs=(path,),
    ) as pool:
        for result in pool.imap_unordered(evaluate, jobs, chunksize=4):
            if result < best:
                best = result
                print(
                    json.dumps(
                        {
                            "score": result[0],
                            "seed": result[1],
                            "amplitude": result[2],
                            "height_weight": HEIGHT_WEIGHT,
                            "cycle_weight": CYCLE_WEIGHT,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
