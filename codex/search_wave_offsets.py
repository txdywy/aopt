"""Grid-search selected full-wave launch offsets on the current kernel."""

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


GROUPS = tuple(int(value) for value in os.environ.get("GROUPS", "24,25").split(","))
VALUES = tuple(
    int(value) for value in os.environ.get("VALUES", "0:23").replace(":", ",").split(",")
)
if len(VALUES) == 2 and ":" in os.environ.get("VALUES", "0:23"):
    VALUES = tuple(range(VALUES[0], VALUES[1] + 1))


def evaluate(values: tuple[int, ...]) -> dict[str, object]:
    offsets = list(kernel.FULL_ROUND_OFFSETS)
    for group, value in zip(GROUPS, values):
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
    return {"score": score, "offsets": dict(zip(GROUPS, values))}


def main() -> None:
    best = (1_000_000, ())
    jobs = itertools.product(VALUES, repeat=len(GROUPS))
    with mp.Pool(int(os.environ.get("WORKERS", "8"))) as pool:
        for result in pool.imap_unordered(evaluate, jobs, chunksize=2):
            values = tuple(result["offsets"].values())
            key = (int(result["score"]), values)
            if key < best:
                best = key
                print(json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
