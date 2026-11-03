"""Search allocations of the eight direct-branch table records."""

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


GROUPS = tuple(int(value) for value in os.environ.get("GROUPS", "22,26,28").split(","))
MAX_PER_GROUP = int(os.environ.get("MAX_PER_GROUP", "8"))
TOTAL_LIMIT = int(os.environ.get("TOTAL_LIMIT", "9"))
TOTAL_MIN = int(os.environ.get("TOTAL_MIN", "1"))
PRINT_BELOW = int(os.environ.get("PRINT_BELOW", "0"))


def evaluate(counts: tuple[int, ...]) -> dict[str, object]:
    kernel.DIRECT_BRANCH_LOOKUPS = {
        group: tuple(range(count))
        for group, count in zip(GROUPS, counts)
        if count
    }
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
    return {"score": score, "allocation": dict(zip(GROUPS, counts))}


def main() -> None:
    jobs = (
        counts
        for counts in itertools.product(range(MAX_PER_GROUP + 1), repeat=len(GROUPS))
        if TOTAL_MIN <= sum(counts) <= TOTAL_LIMIT
    )
    best = (1_000_000,)
    with mp.Pool(int(os.environ.get("WORKERS", "4"))) as pool:
        for result in pool.imap_unordered(evaluate, jobs, chunksize=2):
            counts = tuple(result["allocation"].values())
            key = (int(result["score"]), sum(counts), counts)
            if PRINT_BELOW and int(result["score"]) <= PRINT_BELOW:
                print(json.dumps(result, separators=(",", ":")), flush=True)
            if key < best:
                best = key
                print(json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
