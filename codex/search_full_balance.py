"""Retune ALU/VALU placement on the best complete-wave schedule."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import codex.perf_takehome as kernel
from codex.analyze_schedule import configure
from codex.search_cache_balance import ASSIGNMENT, BASE_EXTRA, OFFSETS


def main() -> None:
    configure()
    kernel.INDEPENDENT_ROOT_CACHE = True
    kernel.TAIL_EMISSION_MODE = "full_offset"
    kernel.FULL_ROUND_OFFSETS = OFFSETS
    kernel.WORKSPACE_ASSIGNMENT = ASSIGNMENT
    kernel.SECOND_WORKSPACE_FIXED = 8
    kernel.SCHEDULE_POLICIES = (36,)
    kernel.BACKWARD_POLICIES = ()
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
        if sum(pair) % kernel.HASH_SCALAR_MOD and pair not in BASE_EXTRA
    ]
    orders = {
        "cycle": sorted(candidates, key=lambda pair: cycles[next(
            i for i, op in enumerate(baseline.dag_ops)
            if op.group == pair[0] and op.round == pair[1]
            and op.tag in ("hash_1_shift_scalar", "hash_1_shift_vector")
        )], reverse=True),
        "wave": sorted(candidates, key=lambda pair: OFFSETS[pair[0]] + pair[1], reverse=True),
        "round": sorted(candidates, key=lambda pair: (pair[1], OFFSETS[pair[0]]), reverse=True),
        "group": sorted(candidates, key=lambda pair: (OFFSETS[pair[0]], pair[1]), reverse=True),
    }
    results = []
    for name, order in orders.items():
        for count in range(0, 111, 10):
            kernel.HASH_SCALAR_EXTRA = frozenset(BASE_EXTRA | set(order[:count]))
            builder = kernel.KernelBuilder()
            builder.build_kernel(10, 2047, 256, 16)
            policies = (0, 1, 2, 3, 16, 17, 18, 19, 36, 37, 38, 39)
            score, policy = min(
                (len(builder._schedule(builder.dag_ops, p)), p) for p in policies
            )
            slots = Counter(op.engine for op in builder.dag_ops)
            row = (score, policy, name, count, slots["alu"], slots["valu"])
            results.append(row)
            print(row, flush=True)
    print("BEST")
    for row in sorted(results)[:20]:
        print(row)


if __name__ == "__main__":
    main()
