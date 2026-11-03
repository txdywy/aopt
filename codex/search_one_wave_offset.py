"""Exhaust every single-coordinate launch-offset mutation."""

from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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


GROUPS = tuple(
    int(value) for value in os.environ.get("GROUPS", "0:31").replace(":", ",").split(",")
)
if len(GROUPS) == 2 and ":" in os.environ.get("GROUPS", "0:31"):
    GROUPS = tuple(range(GROUPS[0], GROUPS[1] + 1))
VALUES = tuple(
    int(value) for value in os.environ.get("VALUES", "0:24").replace(":", ",").split(",")
)
if len(VALUES) == 2 and ":" in os.environ.get("VALUES", "0:24"):
    VALUES = tuple(range(VALUES[0], VALUES[1] + 1))


def evaluate(job: tuple[int, int]) -> dict[str, int]:
    group, value = job
    offsets = list(kernel.FULL_ROUND_OFFSETS)
    offsets[group] = value
    kernel.FULL_ROUND_OFFSETS = tuple(offsets)
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
    return {"score": score, "group": group, "value": value}


def main() -> None:
    best = (1_000_000, -1, -1)
    jobs = itertools.product(GROUPS, VALUES)
    with mp.Pool(int(os.environ.get("WORKERS", "8"))) as pool:
        for result in pool.imap_unordered(evaluate, jobs, chunksize=2):
            key = (result["score"], result["group"], result["value"])
            if key < best:
                best = key
                print(json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
