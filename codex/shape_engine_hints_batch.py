"""Batch-compress engine orders from many source schedules."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from codex.shape_engine_hint import shape_cycles


def main() -> None:
    configure_target()
    kernel.SCHEDULE_EXACT_CYCLES = None
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        if not hasattr(builder, "dag_ops"):
            raise
    ops = real_tail_ops(builder.dag_ops)
    engine = os.environ.get("ENGINE", "flow")
    horizon = int(os.environ.get("TARGET", "959"))
    gap_scale = float(os.environ.get("GAP_SCALE", "0"))
    output_prefix = os.environ.get("OUT_PREFIX", "/tmp/aopt-shaped-batch")
    accepted = 0
    best = 10**9
    for filename in glob.glob(os.environ["SOURCE_GLOB"]):
        payload = json.loads(Path(filename).read_text())
        cycles = shape_cycles(
            ops, payload["cycles"], engine, horizon, gap_scale
        )
        makespan = max(cycles) + 1
        best = min(best, makespan)
        if makespan <= horizon:
            policy = payload.get("policy", Path(filename).stem)
            Path(f"{output_prefix}-{policy}.json").write_text(
                json.dumps(
                    {
                        "makespan": makespan,
                        "engine": engine,
                        "source": filename,
                        "cycles": cycles,
                    }
                )
            )
            accepted += 1
    print(f"accepted={accepted} best={best} prefix={output_prefix}")


if __name__ == "__main__":
    main()
