"""Construct a precedence-valid schedule with only one engine capacity-bound."""

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
    constrained = os.environ.get("ENGINE", "flow")
    original_limits = dict(kernel.SLOT_LIMITS)
    for engine in kernel.SLOT_LIMITS:
        if engine not in {constrained, "debug"}:
            kernel.SLOT_LIMITS[engine] = len(ops)
    try:
        best: tuple[int, int, list[int]] | None = None
        save_max = int(os.environ.get("SAVE_MAX", "-1"))
        save_prefix = os.environ.get("SAVE_PREFIX", "")
        for policy in range(
            int(os.environ.get("POLICY_START", "0")),
            int(os.environ.get("POLICY_END", "179")),
        ):
            _, cycles = builder._schedule(ops, policy, return_cycles=True)
            candidate = (max(cycles) + 1, policy, cycles)
            if save_prefix and candidate[0] <= save_max:
                Path(f"{save_prefix}-{policy}.json").write_text(
                    json.dumps(
                        {
                            "makespan": candidate[0],
                            "policy": policy,
                            "cycles": cycles,
                        }
                    )
                )
            if best is None or candidate[:2] < best[:2]:
                best = candidate
                print(
                    f"makespan={candidate[0]} engine={constrained} policy={policy}",
                    flush=True,
                )
    finally:
        kernel.SLOT_LIMITS.update(original_limits)
    if best is None:
        raise ValueError("empty policy range")
    output = Path(os.environ.get("OUT", f"/tmp/aopt-{constrained}-only.json"))
    output.write_text(
        json.dumps({"makespan": best[0], "policy": best[1], "cycles": best[2]})
    )
    print(f"best={best[0]} policy={best[1]} output={output}")


if __name__ == "__main__":
    main()
