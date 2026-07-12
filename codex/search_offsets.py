"""Hill-climb full-wave group offsets with safe workspace coloring."""

from __future__ import annotations

from pathlib import Path
import heapq
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import codex.perf_takehome as kernel
from codex.analyze_schedule import configure


START = list(kernel.FULL_ROUND_OFFSETS)


def color(offsets: tuple[int, ...], count: int = 8) -> tuple[int, ...] | None:
    assignment = [-1] * 32
    active: list[tuple[int, int]] = []
    free = list(range(count))
    heapq.heapify(free)
    for group in sorted(range(32), key=lambda g: (offsets[g], g)):
        start = offsets[group]
        while active and active[0][0] < start:
            _, workspace = heapq.heappop(active)
            heapq.heappush(free, workspace)
        if not free:
            return None
        workspace = heapq.heappop(free)
        assignment[group] = workspace
        end = start + (4 if group in kernel.FIRST_CACHE_SET else 3)
        heapq.heappush(active, (end, workspace))
    return tuple(assignment)


def normalize(offsets: list[int]) -> tuple[int, ...]:
    low = min(offsets)
    return tuple(value - low for value in offsets)


def evaluate(offsets: tuple[int, ...], memo: dict[tuple[int, ...], int]) -> int:
    if offsets in memo:
        return memo[offsets]
    assignment = color(offsets)
    if assignment is None or max(offsets) > 32:
        return 10_000
    kernel.FULL_ROUND_OFFSETS = offsets
    kernel.WORKSPACE_ASSIGNMENT = assignment
    builder = kernel.KernelBuilder()
    builder.build_kernel(10, 2047, 256, 16)
    score = len(builder.instrs)
    memo[offsets] = score
    return score


def main() -> None:
    configure()
    kernel.INDEPENDENT_ROOT_CACHE = True
    kernel.DIRECT_MIRROR_PATH = False
    kernel.TAIL_EMISSION_MODE = "full_offset"
    kernel.SECOND_WORKSPACE_FIXED = 8
    kernel.SCHEDULE_POLICIES = (36,)
    kernel.BACKWARD_POLICIES = ()

    rng = random.Random(7)
    memo: dict[tuple[int, ...], int] = {}
    current = normalize(START)
    current_score = evaluate(current, memo)
    best, best_score = current, current_score
    print(best_score, best, color(best), flush=True)

    for iteration in range(1200):
        trial = list(current)
        for _ in range(1 if rng.random() < 0.8 else rng.randint(2, 4)):
            group = rng.randrange(32)
            trial[group] = max(0, trial[group] + rng.randint(-4, 4))
        trial_t = normalize(trial)
        score = evaluate(trial_t, memo)
        temperature = max(0.25, 3.0 * (1.0 - (iteration % 300) / 300))
        accept = score <= current_score or rng.random() < 2 ** ((current_score - score) / temperature)
        if accept:
            current, current_score = trial_t, score
        if score < best_score:
            best, best_score = trial_t, score
            print(best_score, best, color(best), flush=True)
        if iteration % 300 == 299:
            current, current_score = best, best_score

    print("BEST", best_score, best, color(best))


if __name__ == "__main__":
    main()
