"""Search legal tail wavefronts and list-scheduler priorities."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import codex.perf_takehome as kernel
from codex.analyze_schedule import configure


def main() -> None:
    configure()
    policies = tuple(range(32)) + tuple(range(48, 54)) + tuple(range(72, 87))
    configs = [("split", 0, 0), ("group", 0, 0), ("round", 0, 0)]
    configs += [
        ("wave", cohorts, stagger)
        for cohorts in (2, 4, 8, 16, 32)
        for stagger in (1, 2, 3, 4)
    ]
    best = []
    for mode, cohorts, stagger in configs:
        kernel.TAIL_EMISSION_MODE = mode
        kernel.TAIL_EMISSION_COHORTS = cohorts
        kernel.TAIL_EMISSION_STAGGER = stagger
        kernel.SCHEDULE_POLICIES = (16,)
        kernel.BACKWARD_POLICIES = ()
        builder = kernel.KernelBuilder()
        builder.build_kernel(10, 2047, 256, 16)
        scores = [(len(builder._schedule(builder.dag_ops, policy)), policy) for policy in policies]
        score, policy = min(scores)
        row = (score, mode, cohorts, stagger, policy)
        best.append(row)
        print(row, flush=True)
    print("BEST")
    for row in sorted(best)[:20]:
        print(row)


if __name__ == "__main__":
    main()
