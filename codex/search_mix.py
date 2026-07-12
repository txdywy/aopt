"""Search resource-equivalent ALU/VALU placement choices."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import codex.perf_takehome as kernel
from codex.analyze_schedule import configure


PAIRS = [(group, rnd) for group in range(32) for rnd in range(16)]


def build() -> tuple[kernel.KernelBuilder, list[int]]:
    builder = kernel.KernelBuilder()
    builder.build_kernel(10, 2047, 256, 16)
    schedule, cycles = builder._schedule(builder.dag_ops, 16, return_cycles=True)
    return builder, cycles


def main() -> None:
    configure()
    kernel.BACKWARD_POLICIES = ()
    baseline, baseline_cycles = build()

    node_cycle: dict[tuple[int, int], int] = {}
    hash_cycle: dict[tuple[int, int], int] = {}
    for index, op in enumerate(baseline.dag_ops):
        pair = (op.group, op.round)
        if op.tag == "node_xor":
            node_cycle[pair] = max(node_cycle.get(pair, -1), baseline_cycles[index])
        elif op.tag in ("hash_1_shift_scalar", "hash_1_shift_vector"):
            hash_cycle[pair] = max(hash_cycle.get(pair, -1), baseline_cycles[index])

    vector_modes: dict[str, list[tuple[int, int]]] = {
        "none": [],
        "deep": [(g, r) for g, r in PAIRS if 4 <= r <= 10],
        "gather": [
            (g, r)
            for g, r in PAIRS
            if 4 <= r <= 10 or (r == 15 and g not in kernel.FINAL_CACHE_SET)
        ],
        "hot224": sorted(PAIRS, key=lambda p: node_cycle.get(p, 10**9))[:224],
        "hot280": sorted(PAIRS, key=lambda p: node_cycle.get(p, 10**9))[:280],
    }

    hash_orders: dict[str, list[tuple[int, int]]] = {
        "late_cycle": sorted(PAIRS, key=lambda p: hash_cycle[p], reverse=True),
        "peak_cycle": sorted(PAIRS, key=lambda p: abs(hash_cycle[p] - 750)),
        "late_round": sorted(PAIRS, key=lambda p: (p[1], p[0]), reverse=True),
        "shallow": sorted(PAIRS, key=lambda p: (p[1] % 11, -p[0])),
        "late_group": sorted(PAIRS, key=lambda p: (p[0], p[1]), reverse=True),
    }

    results = []
    for vector_name, vector_order in vector_modes.items():
        vector_set = frozenset(vector_order)
        for net_scalar in (198, 220, 240, 260, 280):
            scalar_count = len(vector_set) + net_scalar
            if scalar_count > len(PAIRS):
                continue
            for hash_name, hash_order in hash_orders.items():
                chosen = set(hash_order[:scalar_count])
                if (0, 0) not in chosen:
                    chosen.remove(hash_order[scalar_count - 1])
                    chosen.add((0, 0))
                kernel.HASH_SCALAR_MOD = 1_000_000
                kernel.HASH_SCALAR_EXTRA = frozenset(chosen)
                kernel.SCALAR_DYNAMIC_XOR_SET = frozenset()
                kernel.VECTOR_NODE_XOR_SET = vector_set
                builder, _ = build()
                score = len(builder.instrs)
                row = (score, vector_name, net_scalar, hash_name, builder.scratch_ptr)
                results.append(row)
                print(row, flush=True)

    print("BEST")
    for row in sorted(results)[:20]:
        print(row)


if __name__ == "__main__":
    main()
