"""Launch-offset search for the 1025-cycle balanced target graph."""

from __future__ import annotations

from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import codex.perf_takehome as kernel
from codex.analyze_schedule import configure
from codex.search_offsets import color, normalize


START = (4, 0, 2, 10, 4, 17, 0, 18, 5, 17, 0, 2, 19, 16, 0, 18,
         13, 15, 23, 24, 5, 13, 18, 5, 8, 24, 13, 12, 11, 24, 3, 23)
FIRST = frozenset((5, 7, 9, 12, 13, 15, 17, 18, 19, 22, 25, 29, 31))
BASE_EXTRA = {(group, (1 - group) % 4) for group in range(21)}


def configure_target(offsets: tuple[int, ...]) -> bool:
    configure()
    kernel.FIRST_CACHE_SET = FIRST
    kernel.FINAL_CACHE_SET = frozenset(range(9))
    kernel.INDEPENDENT_ROOT_CACHE = True
    kernel.TAIL_EMISSION_MODE = "full_offset"
    kernel.FULL_ROUND_OFFSETS = offsets
    assignment = color(offsets)
    if assignment is None:
        return False
    kernel.WORKSPACE_ASSIGNMENT = assignment
    kernel.SECOND_WORKSPACE_FIXED = 8
    eligible = [
        (group, rnd)
        for group in range(32)
        for rnd in range(16)
        if (group + rnd) % 4 and (group, rnd) not in BASE_EXTRA
    ]
    eligible.sort(key=lambda pair: (pair[1], offsets[pair[0]]), reverse=True)
    kernel.HASH_SCALAR_EXTRA = frozenset(BASE_EXTRA | set(eligible[:50]))
    kernel.SCHEDULE_POLICIES = (36,)
    kernel.BACKWARD_POLICIES = ()
    return True


def main() -> None:
    rng = random.Random(41)
    memo = {}

    def evaluate(offsets: tuple[int, ...]) -> int:
        if offsets in memo:
            return memo[offsets]
        if max(offsets) > 32 or not configure_target(offsets):
            return 10_000
        builder = kernel.KernelBuilder()
        builder.build_kernel(10, 2047, 256, 16)
        memo[offsets] = len(builder.instrs)
        return memo[offsets]

    current = normalize(list(START))
    current_score = evaluate(current)
    best, best_score = current, current_score
    print(best_score, best, color(best), flush=True)
    for iteration in range(2200):
        trial = list(current)
        for _ in range(1 if rng.random() < 0.85 else rng.randint(2, 4)):
            group = rng.randrange(32)
            trial[group] = max(0, trial[group] + rng.randint(-4, 4))
        trial_t = normalize(trial)
        score = evaluate(trial_t)
        temperature = max(0.15, 2.25 * (1 - (iteration % 440) / 440))
        if score <= current_score or rng.random() < 2 ** ((current_score - score) / temperature):
            current, current_score = trial_t, score
        if score < best_score:
            best, best_score = trial_t, score
            print(best_score, best, color(best), flush=True)
        if iteration % 440 == 439:
            current, current_score = best, best_score
    print("BEST", best_score, best, color(best))


if __name__ == "__main__":
    main()
