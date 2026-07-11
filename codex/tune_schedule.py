"""Offline coordinate search for the scheduler's group launch offsets."""

from __future__ import annotations

import sys
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import codex.perf_takehome as kernel


kernel.SCHEDULE_POLICIES = (26,)
kernel.FINAL_CACHE_SET = frozenset()
builder = kernel.KernelBuilder()
builder.build_kernel(10, 2047, 256, 16)

group_offsets = [
    0, 0, 0, -1,
    -63, -63, -63, -65,
    -162, -162, -162, -162,
    -206, -192, -206, -206,
    -290, -302, -312, -305,
    -332, -332, -332, -332,
    -393, -388, -376, -387,
    -412, -410, -410, -407,
]


def evaluate_groups(offsets: list[int]) -> int:
    kernel.GROUP_PRIORITY_OFFSETS = tuple(offsets)
    return len(builder._schedule(builder.dag_ops, 80))


best = evaluate_groups(group_offsets)
print(best, group_offsets, flush=True)
rng = random.Random(0)
for _ in range(500):
    trial = group_offsets.copy()
    for _ in range(rng.randint(1, 3)):
        group = rng.randrange(32)
        trial[group] += rng.randint(-16, 16)
    score = evaluate_groups(trial)
    if score < best:
        group_offsets = trial
        best = score
        print(best, group_offsets, flush=True)
