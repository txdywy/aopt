"""Local hill-climb around a Hall-near H959 lowering.

The broad mixed search explores a useful but expensive space.  This driver
keeps the profitable relocation/direct-lookup skeleton fixed and perturbs the
three exchange sets that dominate the ALU/VALU phase cut: final node XOR
vectorization, scalar hash shifts, and a small set of level-4 gathers.  It is
an offline diagnostic only; no production defaults are changed.
"""

from __future__ import annotations

import gc
import os
import random

from codex.search_h959_mix import evaluate


def pairs(values: list[tuple[int, int]]) -> str:
    return ",".join(f"{g}:{r}" for g, r in values)


def main() -> None:
    rng = random.Random(int(os.environ.get("LOCAL_SEED", "959")))
    trials = int(os.environ.get("LOCAL_TRIALS", "500"))
    # This is the best broad-search point observed so far.  Keep all deep
    # dynamic transforms and restores vectorized, then vary the phase-sensitive
    # final node XOR cells and scalar hash count.
    restore = ",".join(map(str, range(32)))
    dynamic = ",".join(f"{g}:{r}" for g in range(7) for r in range(4, 16))
    node_pool = [(g, 15) for g in range(32)]
    best = (10**9, 10**9, 10**9)
    best_payload = None
    for trial in range(trials):
        node_count = rng.randrange(16, 25)
        node_set = rng.sample(node_pool, node_count)
        hash_count = rng.randrange(164, 186)
        gather_group = rng.choice((1, 5, 6, 10, 13, 15, 22, 24))
        gather = f"1:4,{gather_group}:4"
        scalar_groups = sorted(
            rng.sample((2, 8, 13, 15, 19, 22, 24, 26, 30, 31), rng.randrange(0, 5))
        )
        limits = {3: 8, 4: 8, 5: 8, 6: 8, 7: 8, 9: 8, 10: 8}
        for group in rng.sample(tuple(limits), rng.randrange(0, 4)):
            limits[group] = rng.randrange(0, 9)
        overrides = ["1:4:0"] + [f"{g}:4:{limits[g]}" for g in sorted(limits)]
        settings = {
            "VECTOR_RESTORE_LAYOUT_SET": restore,
            "VECTOR_DYNAMIC_XOR_SET": dynamic,
            "VECTOR_NODE_XOR_SET": pairs(node_set),
            "HASH_SCALAR_EXTRA_COUNT": str(hash_count),
            "HYBRID_MADD_OVERRIDES": ",".join(overrides),
            "SCALAR_SECOND_PATH_GROUPS": ",".join(map(str, scalar_groups)),
            "GATHER_LEVEL4": gather,
            "DIRECT_BRANCH_LOOKUPS": "22:8",
        }
        result = evaluate(settings)
        gc.collect()
        if result is None:
            continue
        key, payload = result
        if key < best:
            best = key
            best_payload = payload
            print(
                f"trial={trial} best={best} halls={payload['halls']} "
                f"counts={payload['counts']} settings={settings}",
                flush=True,
            )
            if best[0] <= 0:
                print("HALL_FEASIBLE", flush=True)
                break
    print(f"final_best={best} payload={best_payload}", flush=True)


if __name__ == "__main__":
    main()
