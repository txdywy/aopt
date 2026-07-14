"""Search final-gather arbitration priorities for the two-slot load engine."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random

import codex.perf_takehome_under1000 as kernel


GATHER_GROUPS = (2, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30)


def evaluate(seed: int) -> dict[str, object]:
    rng = random.Random(seed)
    priorities = {
        key: value
        for key, value in kernel.OP_PRIORITY_OFFSETS.items()
        if not (key[2] == 15 and key[0] in {"tree_gather", "node_xor"})
    }
    order = list(GATHER_GROUPS)
    rng.shuffle(order)
    scale = rng.choice((1, 2, 3, 4, 6, 8, 12, 16, 24, 32))
    offsets = {group: scale * rank for rank, group in enumerate(order)}
    # Occasionally preserve a coarse two-tier policy, but randomize membership.
    if seed & 1:
        cutoff = rng.randrange(4, 13)
        for group in order[-cutoff:]:
            offsets[group] += 50_000
    for group, offset in offsets.items():
        priorities[("tree_gather", group, 15)] = offset
        if rng.random() < 0.5:
            priorities[("node_xor", group, 15)] = offset
    kernel.OP_PRIORITY_OFFSETS = priorities
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
    return {
        "score": score,
        "seed": seed,
        "scale": scale,
        "offsets": offsets,
    }


def main() -> None:
    start = int(os.environ.get("START", "0"))
    count = int(os.environ.get("SEEDS", "1000"))
    best: tuple[int, int] = (1_000_000, -1)
    with mp.Pool(int(os.environ.get("WORKERS", "4"))) as pool:
        for result in pool.imap_unordered(
            evaluate, range(start, start + count), chunksize=4
        ):
            key = (int(result["score"]), int(result["seed"]))
            if key < best:
                best = key
                print(json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
