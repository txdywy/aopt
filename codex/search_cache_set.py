"""Anneal the fixed-size final level-4 cache set."""

from __future__ import annotations

from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import codex.perf_takehome as kernel
from codex.analyze_schedule import configure
from codex.search_cache_balance import ASSIGNMENT, OFFSETS


def main() -> None:
    configure()
    kernel.INDEPENDENT_ROOT_CACHE = True
    kernel.TAIL_EMISSION_MODE = "full_offset"
    kernel.FULL_ROUND_OFFSETS = OFFSETS
    kernel.WORKSPACE_ASSIGNMENT = ASSIGNMENT
    kernel.SECOND_WORKSPACE_FIXED = 8
    kernel.SCHEDULE_POLICIES = (36,)
    kernel.BACKWARD_POLICIES = ()

    memo = {}

    def evaluate(cache_set: frozenset[int]) -> int:
        if cache_set not in memo:
            kernel.FINAL_CACHE_SET = cache_set
            builder = kernel.KernelBuilder()
            builder.build_kernel(10, 2047, 256, 16)
            memo[cache_set] = len(builder.instrs)
        return memo[cache_set]

    rng = random.Random(19)
    current = frozenset(range(15))
    current_score = evaluate(current)
    best, best_score = current, current_score
    print(best_score, tuple(sorted(best)), flush=True)
    for iteration in range(1200):
        inside = list(current)
        outside = list(set(range(32)) - current)
        remove_count = 1 if rng.random() < 0.9 else 2
        trial = set(current)
        for old in rng.sample(inside, remove_count):
            trial.remove(old)
        for new in rng.sample(outside, remove_count):
            trial.add(new)
        trial_f = frozenset(trial)
        score = evaluate(trial_f)
        temperature = max(0.2, 2.5 * (1 - (iteration % 300) / 300))
        if score <= current_score or rng.random() < 2 ** ((current_score - score) / temperature):
            current, current_score = trial_f, score
        if score < best_score:
            best, best_score = trial_f, score
            print(best_score, tuple(sorted(best)), flush=True)
        if iteration % 300 == 299:
            current, current_score = best, best_score
    print("BEST", best_score, tuple(sorted(best)))


if __name__ == "__main__":
    main()
