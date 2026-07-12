"""Tune per-group list-scheduler biases without rebuilding the DAG."""

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
    kernel.GROUP_FINE_OFFSETS = (0,) * 32
    kernel.SCHEDULE_POLICIES = (36,)
    kernel.BACKWARD_POLICIES = ()
    builder = kernel.KernelBuilder()
    builder.build_kernel(10, 2047, 256, 16)

    memo = {}

    def evaluate(offsets: tuple[int, ...]) -> int:
        if offsets not in memo:
            kernel.GROUP_FINE_OFFSETS = offsets
            memo[offsets] = len(builder._schedule(builder.dag_ops, 36))
        return memo[offsets]

    rng = random.Random(23)
    current = (0,) * 32
    current_score = evaluate(current)
    best, best_score = current, current_score
    print(best_score, best, flush=True)
    for iteration in range(1800):
        trial = list(current)
        for _ in range(1 if rng.random() < 0.85 else rng.randint(2, 4)):
            group = rng.randrange(32)
            trial[group] += rng.randint(-20, 20)
        anchor = trial[0]
        trial_t = tuple(value - anchor for value in trial)
        score = evaluate(trial_t)
        temperature = max(0.25, 3.0 * (1 - (iteration % 450) / 450))
        if score <= current_score or rng.random() < 2 ** ((current_score - score) / temperature):
            current, current_score = trial_t, score
        if score < best_score:
            best, best_score = trial_t, score
            print(best_score, best, flush=True)
        if iteration % 450 == 449:
            current, current_score = best, best_score
    print("BEST", best_score, best)


if __name__ == "__main__":
    main()
