"""Jointly sweep level-4 cache arithmetic and ALU/VALU hash placement."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import codex.perf_takehome as kernel
from codex.analyze_schedule import configure


OFFSETS = (6, 0, 1, 10, 8, 12, 0, 20, 16, 19, 4, 3, 15, 16, 0, 14,
           13, 23, 24, 21, 3, 18, 24, 8, 12, 22, 13, 6, 10, 23, 3, 24)
ASSIGNMENT = (1, 0, 3, 1, 0, 0, 1, 0, 0, 2, 0, 4, 2, 3, 2, 1,
              4, 2, 0, 3, 5, 1, 5, 3, 3, 1, 5, 2, 2, 4, 6, 6)
BASE_EXTRA = {(group, (1 - group) % 4) for group in range(21)}


def setup(cache_count: int, madd_pairs: int) -> None:
    configure()
    kernel.FIRST_CACHE_SET = frozenset()
    kernel.FINAL_CACHE_SET = frozenset(range(cache_count))
    kernel.HYBRID_MADD_PAIRS = madd_pairs
    kernel.INDEPENDENT_ROOT_CACHE = True
    kernel.DIRECT_MIRROR_PATH = False
    kernel.TAIL_EMISSION_MODE = "full_offset"
    kernel.FULL_ROUND_OFFSETS = OFFSETS
    kernel.WORKSPACE_ASSIGNMENT = ASSIGNMENT
    kernel.SECOND_WORKSPACE_FIXED = 8
    kernel.SCHEDULE_POLICIES = (36,)
    kernel.BACKWARD_POLICIES = ()


def main() -> None:
    results = []
    for cache_count, madd_pairs in ((24, 4), (25, 5), (26, 6), (27, 7)):
        setup(cache_count, madd_pairs)
        kernel.HASH_SCALAR_EXTRA = frozenset(BASE_EXTRA)
        baseline = kernel.KernelBuilder()
        baseline.build_kernel(10, 2047, 256, 16)
        _, cycles = baseline._schedule(baseline.dag_ops, 36, return_cycles=True)
        hash_cycle = {}
        for index, op in enumerate(baseline.dag_ops):
            if op.tag in ("hash_1_shift_scalar", "hash_1_shift_vector"):
                hash_cycle[(op.group, op.round)] = cycles[index]
        candidates = [
            pair for pair in hash_cycle
            if sum(pair) % kernel.HASH_SCALAR_MOD != 0 and pair not in BASE_EXTRA
        ]
        orders = {
            "cycle": sorted(candidates, key=lambda pair: hash_cycle[pair], reverse=True),
            "wave": sorted(candidates, key=lambda pair: OFFSETS[pair[0]] + pair[1], reverse=True),
            "round": sorted(candidates, key=lambda pair: (pair[1], OFFSETS[pair[0]]), reverse=True),
            "group": sorted(candidates, key=lambda pair: (OFFSETS[pair[0]], pair[1]), reverse=True),
        }
        for order_name, order in orders.items():
            for extra_count in range(0, 121, 15):
                kernel.HASH_SCALAR_EXTRA = frozenset(BASE_EXTRA | set(order[:extra_count]))
                builder = kernel.KernelBuilder()
                builder.build_kernel(10, 2047, 256, 16)
                scores = [len(builder._schedule(builder.dag_ops, p)) for p in (0, 16, 36, 37, 38, 39)]
                slots = Counter(op.engine for op in builder.dag_ops)
                row = (
                    min(scores), cache_count, madd_pairs, order_name, extra_count,
                    slots["alu"], slots["valu"], slots["load"], slots["flow"],
                )
                results.append(row)
                print(row, flush=True)
    print("BEST")
    for row in sorted(results)[:25]:
        print(row)


if __name__ == "__main__":
    main()
