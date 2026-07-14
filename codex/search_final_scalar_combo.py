"""Exhaust final-hash scalarization choices around the verified 991 tail."""

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
    ("SCALAR_FINAL_HASH4", "SCALAR_FINAL_HASH4_SET"),
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


NAME_MAP = {
    "C5": "SCALAR_FINAL_C5_SET",
    "JOIN": "SCALAR_FINAL_JOIN_SET",
    "SHIFT": "SCALAR_FINAL_SHIFT_SET",
    "H23": "SCALAR_FINAL_HASH23_JOIN_SET",
    "H4": "SCALAR_FINAL_HASH4_SET",
}
GROUPS = tuple(int(value) for value in os.environ.get("GROUPS", "29,30,31").split(","))
NAMES = tuple(
    NAME_MAP[value]
    for value in os.environ.get("OPTIONS", "C5,JOIN,SHIFT,H23").split(",")
)


def evaluate(bits: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    kernel.DIRECT_BRANCH_LOOKUPS = {22: (0, 1, 2), 26: (0, 1, 2, 3)}
    original = {name: set(getattr(kernel, name)) for name in NAMES}
    width = len(NAMES)
    for offset, group in enumerate(GROUPS):
        group_bits = bits[offset * width : (offset + 1) * width]
        for name, enabled in zip(NAMES, group_bits):
            values = original[name]
            values.discard(group)
            if enabled:
                values.add(group)
    for name, values in original.items():
        setattr(kernel, name, frozenset(values))
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
    return score, bits


def main() -> None:
    best = (1_000_000, ())
    jobs = itertools.product((0, 1), repeat=len(GROUPS) * len(NAMES))
    with mp.Pool(int(os.environ.get("WORKERS", "8"))) as pool:
        for result in pool.imap_unordered(evaluate, jobs, chunksize=4):
            if result < best:
                best = result
                print(
                    json.dumps(
                        {
                            "score": result[0],
                            "groups": GROUPS,
                            "options": NAMES,
                            "bits": result[1],
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
