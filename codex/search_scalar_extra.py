"""Random subsets of late hash-constant scalarization candidates."""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import codex.perf_takehome_under1000 as kernel


def main() -> None:
    seed = int(sys.argv[1])
    rng = random.Random(seed)
    chosen = set(range(43))
    for index in range(43, 70):
        if rng.random() < 0.32:
            chosen.add(index)
    removed = {(27, 15), (28, 15), (29, 15), (30, 15), (31, 15)}
    kernel.HASH_SCALAR_EXTRA = frozenset(
        (
            kernel._BASE_SCALAR
            | {kernel._SCALAR_CANDIDATES[index] for index in chosen}
        )
        - removed
    )
    builder = kernel.KernelBuilder()
    builder.build_kernel(10, 2047, 256, 16)
    score = next(
        cycle + 1
        for cycle, bundle in enumerate(builder.instrs)
        if ("halt",) in bundle.get("flow", ())
    )
    print(json.dumps({"score": score, "seed": seed, "chosen": sorted(chosen)}))


if __name__ == "__main__":
    main()
