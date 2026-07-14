"""Search swaps around the best VALU-5895 final-hash allocation."""

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


ATTRS = {
    "C5": "SCALAR_FINAL_C5_SET",
    "JOIN": "SCALAR_FINAL_JOIN_SET",
    "SHIFT": "SCALAR_FINAL_SHIFT_SET",
    "H23": "SCALAR_FINAL_HASH23_JOIN_SET",
}
BASE = {
    "C5": {18, 20, 30, 31},
    "JOIN": {21, 30},
    "SHIFT": {17, 18, 20, 23, 26, 29, 30},
    "H23": {17, 21, 26, 31},
}
BITS = tuple((name, group) for name in ATTRS for group in range(16, 32))
BASE_BITS = frozenset(
    (name, group) for name, groups in BASE.items() for group in groups
)


def evaluate(seed: int) -> dict[str, object]:
    rng = random.Random(seed)
    selected = set(BASE_BITS)
    remove_count = rng.randint(0, 4)
    add_count = remove_count + rng.randint(1, 3)
    if remove_count:
        selected.difference_update(rng.sample(tuple(selected), remove_count))
    absent = tuple(bit for bit in BITS if bit not in selected)
    selected.update(rng.sample(absent, add_count))
    configs = {
        name: frozenset(group for candidate_name, group in selected if candidate_name == name)
        for name in ATTRS
    }
    for name, attr in ATTRS.items():
        setattr(kernel, attr, configs[name])
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
        "configs": {name: sorted(values) for name, values in configs.items()},
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
