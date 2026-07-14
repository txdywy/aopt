"""Interpolate critical and global-phase priorities for one constrained engine."""

from __future__ import annotations

import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target


def main() -> None:
    configure_target()
    kernel.SCHEDULE_EXACT_CYCLES = None
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        if not hasattr(builder, "dag_ops"):
            raise
    ops = builder.dag_ops
    source = json.loads(Path(os.environ["SOURCE"]).read_text())["cycles"]
    constrained = os.environ.get("ENGINE", "flow")
    horizon = int(os.environ.get("TARGET", "959"))
    score_weights = tuple(
        int(value)
        for value in os.environ.get(
            "SCORE_WEIGHTS", "1,2,3,4,6,8,12,16,24,32"
        ).split(",")
    )
    height_weights = tuple(
        int(value)
        for value in os.environ.get("HEIGHT_WEIGHTS", "1,2,3,4,6,8").split(",")
    )
    prefix = os.environ.get("OUT_PREFIX", "/tmp/aopt-guided-engine")
    original_limits = dict(kernel.SLOT_LIMITS)
    if bool(int(os.environ.get("RELAX_OTHER", "1"))):
        for engine in kernel.SLOT_LIMITS:
            if engine not in {constrained, "debug"}:
                kernel.SLOT_LIMITS[engine] = len(ops)
    accepted = 0
    best = 10**9
    best_payload: dict[str, object] | None = None
    try:
        for score_weight in score_weights:
            external_scores = [-score_weight * cycle for cycle in source]
            for height_weight in height_weights:
                for policy in range(4):
                    _, cycles = builder._schedule(
                        ops,
                        policy,
                        return_cycles=True,
                        external_scores=external_scores,
                        height_weight=height_weight,
                    )
                    makespan = max(cycles) + 1
                    if makespan < best:
                        best = makespan
                        best_payload = {
                            "makespan": makespan,
                            "score_weight": score_weight,
                            "height_weight": height_weight,
                            "policy": policy,
                            "cycles": cycles,
                        }
                    if makespan <= horizon:
                        name = f"s{score_weight}-h{height_weight}-p{policy}"
                        Path(f"{prefix}-{name}.json").write_text(
                            json.dumps(
                                {
                                    "makespan": makespan,
                                    "score_weight": score_weight,
                                    "height_weight": height_weight,
                                    "policy": policy,
                                    "cycles": cycles,
                                }
                            )
                        )
                        accepted += 1
    finally:
        kernel.SLOT_LIMITS.update(original_limits)
    if best_payload is not None and "OUT" in os.environ:
        Path(os.environ["OUT"]).write_text(json.dumps(best_payload))
    print(f"accepted={accepted} best={best} prefix={prefix}")


if __name__ == "__main__":
    main()
