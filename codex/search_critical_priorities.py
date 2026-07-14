"""Grid-search final-round priorities for the two critical tail groups."""

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


GROUP_A = int(os.environ.get("GROUP_A", "24"))
GROUP_B = int(os.environ.get("GROUP_B", "25"))


VALUES = tuple(
    int(value)
    for value in os.environ.get(
        "VALUES", "0,100,500,1000,2500,5000,10000,20000,50000,100000"
    ).split(",")
)


def evaluate(job: tuple[int, int, int, int]) -> dict[str, int]:
    full_a, tail_a, full_b, tail_b = job
    priorities = dict(kernel.OP_PRIORITY_OFFSETS)
    for group, full, tail in (
        (GROUP_A, full_a, tail_a),
        (GROUP_B, full_b, tail_b),
    ):
        for tag in kernel._FINAL_HASH_FULL_TAGS:
            priorities.pop((tag, group, 15), None)
            priorities[(tag, group, 15)] = full
        for tag in kernel._FINAL_HASH_TAIL_TAGS:
            priorities[(tag, group, 15)] = tail
    kernel.OP_PRIORITY_OFFSETS = priorities
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
        "group_a": GROUP_A,
        "full_a": full_a,
        "tail_a": tail_a,
        "group_b": GROUP_B,
        "full_b": full_b,
        "tail_b": tail_b,
    }


def main() -> None:
    mode = os.environ.get("MODE", "full")
    if mode == "full":
        jobs = ((a, a, b, b) for a, b in itertools.product(VALUES, repeat=2))
    elif mode == "tail":
        full_a = int(os.environ.get("FULL_A", "0"))
        full_b = int(os.environ.get("FULL_B", "100"))
        jobs = (
            (full_a, tail_a, full_b, tail_b)
            for tail_a, tail_b in itertools.product(VALUES, repeat=2)
        )
    else:
        jobs = itertools.product(VALUES, repeat=4)
    best = (1_000_000,)
    with mp.Pool(int(os.environ.get("WORKERS", "8"))) as pool:
        for result in pool.imap_unordered(evaluate, jobs, chunksize=2):
            key = (
                result["score"],
                result["full_a"],
                result["tail_a"],
                result["full_b"],
                result["tail_b"],
            )
            if key < best:
                best = key
                print(json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
