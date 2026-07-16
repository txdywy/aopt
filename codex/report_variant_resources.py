"""Report exact engine counts for an environment-configured kernel variant."""

from __future__ import annotations

from collections import Counter
import os

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from problem import SLOT_LIMITS


def main() -> None:
    configure_target()
    kernel.SCHEDULE_EXACT_CYCLES = (
        [] if bool(int(os.environ.get("SKIP_SCHEDULE", "0"))) else None
    )
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration, ValueError):
        if not hasattr(builder, "dag_ops"):
            raise
    ops = real_tail_ops(builder.dag_ops)
    counts = Counter(op.engine for op in ops)
    floors = {
        engine: (counts[engine] + capacity - 1) // capacity
        for engine, capacity in SLOT_LIMITS.items()
    }
    slack = {
        engine: 959 * capacity - counts[engine]
        for engine, capacity in SLOT_LIMITS.items()
    }
    print(f"scratch={builder.scratch_ptr}")
    print(f"counts={dict(counts)}")
    print(f"floors={floors}")
    print(f"slack_at_959={slack}")


if __name__ == "__main__":
    main()
