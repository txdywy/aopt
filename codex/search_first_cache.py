"""Search a larger first-pass cache set on the balanced 1023 graph."""

from __future__ import annotations

from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import codex.perf_takehome as kernel
from codex.search_offsets import color
from codex.search_target_offsets import FIRST, configure_target


OFFSETS = (5, 0, 2, 15, 4, 24, 0, 16, 2, 7, 8, 9, 24, 10, 10, 16,
           13, 19, 21, 17, 9, 8, 10, 4, 12, 21, 19, 15, 16, 24, 0, 23)
SIZE = 17


def main() -> None:
    rng = random.Random(53)
    memo = {}

    def evaluate(cache_set: frozenset[int]) -> int:
        if cache_set in memo:
            return memo[cache_set]
        configure_target(OFFSETS)
        kernel.FIRST_CACHE_SET = cache_set
        assignment = color(OFFSETS)
        if assignment is None:
            return 10_000
        kernel.WORKSPACE_ASSIGNMENT = assignment
        builder = kernel.KernelBuilder()
        builder.build_kernel(10, 2047, 256, 16)
        memo[cache_set] = len(builder.instrs)
        return memo[cache_set]

    current = FIRST | frozenset((8, 10, 16, 28))
    current_score = evaluate(current)
    best, best_score = current, current_score
    print(best_score, tuple(sorted(best)), flush=True)
    for iteration in range(1600):
        trial = set(current)
        swaps = 1 if rng.random() < 0.9 else 2
        for old in rng.sample(tuple(trial), swaps):
            trial.remove(old)
        for new in rng.sample(tuple(set(range(32)) - trial), swaps):
            trial.add(new)
        if len(trial) != SIZE:
            continue
        trial_f = frozenset(trial)
        score = evaluate(trial_f)
        temperature = max(0.2, 2.5 * (1 - (iteration % 400) / 400))
        if score <= current_score or rng.random() < 2 ** ((current_score - score) / temperature):
            current, current_score = trial_f, score
        if score < best_score:
            best, best_score = trial_f, score
            print(best_score, tuple(sorted(best)), flush=True)
        if iteration % 400 == 399:
            current, current_score = best, best_score
    print("BEST", best_score, tuple(sorted(best)))


if __name__ == "__main__":
    main()
