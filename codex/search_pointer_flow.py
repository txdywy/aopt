"""Search safe output-pointer updates to migrate from ALU to flow."""

from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import codex.perf_takehome_under1000 as kernel


POSITIONS = tuple(
    int(value)
    for value in os.environ.get("POSITIONS", "0,1,4,5,6,8,9,11,12,13,14").split(",")
)


def evaluate(positions: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    kernel.FLOW_OUTPUT_ADVANCE_POSITIONS = frozenset(positions)
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
    return score, positions


def main() -> None:
    max_size = int(os.environ.get("MAX_SIZE", "3"))
    min_size = int(os.environ.get("MIN_SIZE", "0"))
    jobs = itertools.chain.from_iterable(
        itertools.combinations(POSITIONS, size)
        for size in range(min_size, max_size + 1)
    )
    best = (1_000_000, 0, ())
    with mp.Pool(int(os.environ.get("WORKERS", "4"))) as pool:
        for result in pool.imap_unordered(evaluate, jobs, chunksize=2):
            key = (result[0], -len(result[1]), result[1])
            if key < best:
                best = key
                print(
                    json.dumps(
                        {"score": result[0], "positions": result[1]},
                        separators=(",", ":"),
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
