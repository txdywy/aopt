"""Exhaust lane identities for a fixed direct-branch record allocation."""

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


ALLOCATION = tuple(
    (int(group), int(count))
    for group, count in (
        item.split(":")
        for item in os.environ.get("ALLOCATION", "22:3,26:4").split(",")
        if item
    )
)


def evaluate(job: tuple[tuple[int, ...], ...]) -> dict[str, object]:
    kernel.DIRECT_BRANCH_LOOKUPS = {
        group: lanes for (group, _), lanes in zip(ALLOCATION, job)
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
    return {
        "score": score,
        "lanes": {
            str(group): lanes for (group, _), lanes in zip(ALLOCATION, job)
        },
    }


def main() -> None:
    choices = tuple(
        tuple(itertools.combinations(range(8), count))
        for _, count in ALLOCATION
    )
    jobs = itertools.product(*choices)
    best = (1_000_000, ())
    with mp.Pool(int(os.environ.get("WORKERS", "8"))) as pool:
        for result in pool.imap_unordered(evaluate, jobs, chunksize=2):
            key = (
                int(result["score"]),
                tuple(tuple(x) for x in result["lanes"].values()),
            )
            if key < best:
                best = key
                print(json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
