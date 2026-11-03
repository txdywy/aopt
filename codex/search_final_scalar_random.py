"""Randomly rebalance final hashes between VALU and scalar ALU."""

from __future__ import annotations

from collections import Counter
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import codex.perf_takehome_under1000 as kernel


NAMES = (
    "SCALAR_FINAL_C5_SET",
    "SCALAR_FINAL_JOIN_SET",
    "SCALAR_FINAL_SHIFT_SET",
    "SCALAR_FINAL_HASH23_JOIN_SET",
)
GROUPS = tuple(range(16, 32))
CHOICES = tuple((name, group) for name in NAMES for group in GROUPS)
BASE = {name: frozenset(getattr(kernel, name)) for name in NAMES}


def evaluate(seed: int) -> dict[str, object]:
    rng = random.Random(seed)
    selected = set(rng.sample(CHOICES, rng.randint(1, 10)))
    configs: dict[str, frozenset[int]] = {}
    for name in NAMES:
        values = set(BASE[name])
        for candidate_name, group in selected:
            if candidate_name == name:
                values.symmetric_difference_update((group,))
        config = frozenset(values)
        configs[name] = config
        setattr(kernel, name, config)
    try:
        builder = kernel.KernelBuilder()
        builder.build_kernel(10, 2047, 256, 16)
        score = next(
            cycle + 1
            for cycle, bundle in enumerate(builder.instrs)
            if ("halt",) in bundle.get("flow", ())
        )
        counts = Counter(op.engine for op in builder.dag_ops)
    except (AssertionError, StopIteration, ValueError):
        score = 1_000_000
        counts = Counter()
    return {
        "score": score,
        "seed": seed,
        "alu": counts["alu"],
        "valu": counts["valu"],
        "configs": {name: sorted(configs[name]) for name in NAMES},
    }


def main() -> None:
    start = int(os.environ.get("START", "0"))
    count = int(os.environ.get("SEEDS", "1000"))
    best = (1_000_000, 1_000_000, 1_000_000, -1)
    with mp.Pool(int(os.environ.get("WORKERS", "8"))) as pool:
        for result in pool.imap_unordered(
            evaluate, range(start, start + count), chunksize=2
        ):
            key = (
                int(result["score"]),
                int(result["valu"]),
                int(result["alu"]),
                int(result["seed"]),
            )
            if key < best:
                best = key
                print(json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
