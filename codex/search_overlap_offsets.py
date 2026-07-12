"""Retune launch offsets for the overlapped deep-address pipeline."""

from __future__ import annotations

from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import codex.perf_takehome as kernel
from codex.analyze_schedule import configure
from codex.search_cache_balance import OFFSETS
from codex.search_offsets import color, normalize


FIRST = frozenset((7, 17, 18, 19, 22, 25, 29, 31))


def main() -> None:
    configure()
    kernel.FIRST_CACHE_SET = FIRST
    kernel.FINAL_CACHE_SET = frozenset(range(10))
    kernel.HYBRID_MADD_PAIRS = 8
    kernel.OVERLAP_DEEP_ADDRESS = True
    kernel.INDEPENDENT_ROOT_CACHE = True
    kernel.TAIL_EMISSION_MODE = "full_offset"
    kernel.SECOND_WORKSPACE_FIXED = 8
    kernel.SCHEDULE_POLICIES = (36,)
    kernel.BACKWARD_POLICIES = ()

    memo = {}

    def evaluate(offsets: tuple[int, ...]) -> int:
        if offsets in memo:
            return memo[offsets]
        kernel.FULL_ROUND_OFFSETS = offsets
        assignment = color(offsets)
        if assignment is None or max(offsets) > 32:
            return 10_000
        kernel.WORKSPACE_ASSIGNMENT = assignment
        builder = kernel.KernelBuilder()
        builder.build_kernel(10, 2047, 256, 16)
        memo[offsets] = len(builder.instrs)
        return memo[offsets]

    rng = random.Random(31)
    current = normalize(list(OFFSETS))
    current_score = evaluate(current)
    best, best_score = current, current_score
    print(best_score, best, color(best), flush=True)
    for iteration in range(1600):
        trial = list(current)
        for _ in range(1 if rng.random() < 0.85 else rng.randint(2, 4)):
            group = rng.randrange(32)
            trial[group] = max(0, trial[group] + rng.randint(-4, 4))
        trial_t = normalize(trial)
        score = evaluate(trial_t)
        temperature = max(0.2, 2.5 * (1 - (iteration % 400) / 400))
        if score <= current_score or rng.random() < 2 ** ((current_score - score) / temperature):
            current, current_score = trial_t, score
        if score < best_score:
            best, best_score = trial_t, score
            print(best_score, best, color(best), flush=True)
        if iteration % 400 == 399:
            current, current_score = best, best_score
    print("BEST", best_score, best, color(best))


if __name__ == "__main__":
    main()
