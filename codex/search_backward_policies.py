"""Search list schedules constructed backward from the epilogue."""

from __future__ import annotations

import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops, validate


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
    checked_ops = real_tail_ops(ops)
    reversed_ops = builder._reverse_ops(ops)
    best: tuple[int, int, list[int]] | None = None
    for policy in range(
        int(os.environ.get("POLICY_START", "0")),
        int(os.environ.get("POLICY_END", "179")),
    ):
        _, reversed_cycles = builder._schedule(
            reversed_ops, policy, return_cycles=True
        )
        maximum = max(reversed_cycles)
        cycles = [
            maximum - reversed_cycles[len(ops) - 1 - index]
            for index in range(len(ops))
        ]
        validate(checked_ops, cycles)
        candidate = (maximum + 1, policy, cycles)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
            print(f"makespan={candidate[0]} policy={policy}", flush=True)
    if best is None:
        raise ValueError("empty policy range")
    output = Path(os.environ.get("OUT", "/tmp/aopt-best-backward.json"))
    output.write_text(
        json.dumps({"makespan": best[0], "policy": best[1], "cycles": best[2]})
    )
    print(f"best={best[0]} policy={best[1]} output={output}")


if __name__ == "__main__":
    main()
