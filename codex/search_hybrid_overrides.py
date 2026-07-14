"""Find non-critical cached lookups whose bottom MADD can use flow instead."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random

import codex.perf_takehome_under1000 as kernel


OCCURRENCES = tuple(
    [(group, 4) for group in sorted(kernel.FIRST_CACHE_SET)]
    + [(group, 15) for group in sorted(kernel.FINAL_CACHE_SET)]
)


def evaluate(seed: int) -> tuple[int, int, int, tuple[tuple[int, int], ...]]:
    rng = random.Random(seed)
    minimum = int(os.environ.get("MIN_COUNT", "1"))
    maximum = int(os.environ.get("MAX_COUNT", str(len(OCCURRENCES))))
    count = rng.randrange(minimum, maximum + 1)
    selected = tuple(sorted(rng.sample(OCCURRENCES, count)))
    kernel.HYBRID_MADD_PAIRS = 8
    kernel.HYBRID_MADD_OVERRIDES = {key: 7 for key in selected}
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
    return score, -count, seed, selected


def main() -> None:
    start = int(os.environ.get("START", "0"))
    count = int(os.environ.get("SEEDS", "1000"))
    best = (1_000_000, 0, -1, ())
    with mp.Pool(int(os.environ.get("WORKERS", "4"))) as pool:
        for result in pool.imap_unordered(
            evaluate, range(start, start + count), chunksize=4
        ):
            if result < best:
                best = result
                print(
                    json.dumps(
                        {
                            "score": result[0],
                            "count": -result[1],
                            "seed": result[2],
                            "selected": result[3],
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
