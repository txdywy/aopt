"""Offline coordinate search for the scheduler's group launch offsets."""

from __future__ import annotations

import sys
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import codex.perf_takehome as kernel


kernel.SCHEDULE_POLICIES = (26,)
kernel.FINAL_CACHE_SET = frozenset(range(20))
kernel.N_WORKSPACES = 8
builder = kernel.KernelBuilder()
builder.build_kernel(10, 2047, 256, 16)

group_offsets = [
    2, -7, 3, 15,
    -63, -58, -56, -69,
    -170, -150, -167, -162,
    -200, -176, -207, -216,
    -296, -308, -296, -304,
    -385, -342, -359, -325,
    -378, -362, -364, -387,
    -408, -410, -398, -407,
]


def evaluate(offsets: list[int]) -> int:
    kernel.GROUP_PRIORITY_OFFSETS = tuple(offsets)
    return len(builder._schedule(builder.dag_ops, 80))


best = evaluate(group_offsets)
print(best, group_offsets, flush=True)
rng = random.Random(1)
for _ in range(750):
    trial = list(group_offsets)
    for _ in range(rng.randint(1, 3)):
        group = rng.randrange(32)
        trial[group] += rng.randint(-20, 20)
    score = evaluate(trial)
    if score < best:
        group_offsets = trial
        best = score
        print(best, group_offsets, flush=True)
