"""One-shot randomized tail-priority candidate for parallel shell search."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import codex.perf_takehome_under1000 as kernel


def main() -> None:
    seed = int(sys.argv[1])
    rng = random.Random(seed)
    if "DIRECT_GROUP" in os.environ:
        kernel.DIRECT_BRANCH_LOOKUPS = {
            int(os.environ["DIRECT_GROUP"]): (0, 1)
        }
    priorities = dict(kernel.OP_PRIORITY_OFFSETS)
    load_groups = {24, 25, 26, 27, 28, 30}

    for group in range(20, 31):
        if rng.random() < 0.22:
            if group in load_groups:
                load_groups.remove(group)
            else:
                load_groups.add(group)
    for group in range(20, 31):
        for tag in ("tree_gather", "node_xor"):
            priorities.pop((tag, group, 15), None)
            if group in load_groups:
                priorities[(tag, group, 15)] = 50_000

    pre_groups = set()
    for group in range(20, 31):
        if rng.random() < 0.35:
            pre_groups.add(group)
            for tag in (
                "hash_0",
                "hash_1_shift_vector",
                "hash_1_const",
                "hash_1_join",
            ):
                priorities[(tag, group, 15)] = 1

    weak_tail_groups = set()
    for group in range(20, 27):
        if rng.random() < 0.4:
            weak_tail_groups.add(group)
            for tag in kernel._FINAL_HASH_TAIL_TAGS:
                priorities[(tag, group, 15)] = 1

    kernel.OP_PRIORITY_OFFSETS = priorities
    scalar_hash = set(kernel.HASH_SCALAR_EXTRA)
    for group in range(20, 32):
        if rng.random() < 0.2:
            pair = (group, 15)
            if pair in scalar_hash:
                scalar_hash.remove(pair)
            else:
                scalar_hash.add(pair)
    kernel.HASH_SCALAR_EXTRA = frozenset(scalar_hash)

    scalar_c5 = set(kernel.SCALAR_FINAL_C5_SET)
    for group in (18, 20, 27, 28):
        if rng.random() < 0.2:
            scalar_c5.symmetric_difference_update({group})
    kernel.SCALAR_FINAL_C5_SET = frozenset(scalar_c5)

    scalar_shift = set(kernel.SCALAR_FINAL_SHIFT_SET)
    for group in (17, 20, 23, 26):
        if rng.random() < 0.2:
            scalar_shift.symmetric_difference_update({group})
    kernel.SCALAR_FINAL_SHIFT_SET = frozenset(scalar_shift)

    scalar_h23 = set(kernel.SCALAR_FINAL_HASH23_JOIN_SET)
    for group in (17, 26):
        if rng.random() < 0.2:
            scalar_h23.symmetric_difference_update({group})
    kernel.SCALAR_FINAL_HASH23_JOIN_SET = frozenset(scalar_h23)

    scalar_join = set(kernel.SCALAR_FINAL_JOIN_SET)
    if rng.random() < 0.2:
        scalar_join.symmetric_difference_update({21})
    kernel.SCALAR_FINAL_JOIN_SET = frozenset(scalar_join)

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
                "load": sorted(load_groups),
                "pre": sorted(pre_groups),
                "weak_tail": sorted(weak_tail_groups),
                "scalar_hash15": sorted(
                    group for group in range(20, 32)
                    if (group, 15) in scalar_hash
                ),
                "scalar_c5": sorted(scalar_c5),
                "scalar_shift": sorted(scalar_shift),
                "scalar_h23": sorted(scalar_h23),
                "scalar_join": sorted(scalar_join),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
