"""Search ALU offloads for hash joins near the saturated VALU tail."""

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
    sets = [set(), set(), set()]
    for rnd, probability in ((13, 0.035), (14, 0.08), (15, 0.16)):
        for group in range(32):
            for stage in range(3):
                if rng.random() < probability:
                    sets[stage].add((group, rnd))
    (
        kernel.SCALAR_HASH1_JOIN_SET,
        kernel.SCALAR_HASH23_JOIN_SET,
        kernel.SCALAR_HASH5_JOIN_SET,
    ) = map(frozenset, sets)
    builder = kernel.KernelBuilder()
    builder.build_kernel(10, 2047, 256, 16)
    score = next(
        cycle + 1
        for cycle, bundle in enumerate(builder.instrs)
        if ("halt",) in bundle.get("flow", ())
    )
    print(
        json.dumps(
            {
                "score": score,
                "seed": seed,
                "sets": [sorted(values) for values in sets],
            }
        )
    )


if __name__ == "__main__":
    main()
