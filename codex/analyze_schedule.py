"""Small offline diagnostics for the experimental DAG scheduler."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import codex.perf_takehome as kernel


BEST_ASSIGNMENT = (
    8, 0, 7, 8, 7, 3, 2, 1, 6, 4, 8, 0, 7, 2, 6, 7,
    5, 8, 3, 7, 5, 8, 1, 6, 2, 4, 7, 2, 7, 1, 2, 0,
)


def configure() -> None:
    kernel.WORKSPACE_ASSIGNMENT = BEST_ASSIGNMENT
    kernel.GROUP_FINE_OFFSETS = (0,) * kernel.N_GROUPS
    kernel.FIRST_CACHE_SET = frozenset((29, 30, 31))
    kernel.FINAL_CACHE_SET = frozenset(range(15))
    kernel.SECOND_WORKSPACE_STRIDE = 3
    kernel.SCALAR_DYNAMIC_XOR_SET = frozenset(
        (group, rnd) for group in range(7) for rnd in range(4, 16)
    )
    kernel.SCHEDULE_POLICIES = (16,)
    kernel.BACKWARD_POLICIES = ()


def main() -> None:
    configure()
    builder = kernel.KernelBuilder()
    builder.build_kernel(10, 2047, 256, 16)
    schedule, cycles = builder._schedule(builder.dag_ops, 16, return_cycles=True)
    slots = Counter(op.engine for op in builder.dag_ops)
    print(f"cycles={len(schedule)} scratch={builder.scratch_ptr} slots={dict(slots)}")

    for start in range(0, len(schedule), 50):
        stop = min(start + 50, len(schedule))
        used = Counter()
        for bundle in schedule[start:stop]:
            for engine, engine_slots in bundle.items():
                used[engine] += len(engine_slots)
        util = " ".join(
            f"{engine}:{used[engine]}/{(stop-start)*kernel.SLOT_LIMITS[engine]}"
            for engine in ("alu", "valu", "load", "flow", "store")
        )
        print(f"{start:4d}-{stop-1:4d} {util}")

    completion = [-1] * kernel.N_GROUPS
    for index, op in enumerate(builder.dag_ops):
        if op.group >= 0:
            completion[op.group] = max(completion[op.group], cycles[index])
    print("group_completion=" + repr(tuple(completion)))

    tail = Counter()
    for index, op in enumerate(builder.dag_ops):
        if cycles[index] >= len(schedule) - 100:
            tail[(op.engine, op.tag, op.round)] += 1
    for key, count in tail.most_common(30):
        print(count, key)


if __name__ == "__main__":
    main()
