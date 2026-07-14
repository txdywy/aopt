"""Report necessary per-engine interval capacity for a configured DAG."""

from __future__ import annotations

import os

import numpy as np

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from problem import SLOT_LIMITS


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

    earliest = [0] * len(ops)
    reason = [-1] * len(ops)
    children: list[list[tuple[int, int]]] = [[] for _ in ops]
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            candidate = earliest[parent] + lag
            if candidate > earliest[child]:
                earliest[child] = candidate
                reason[child] = parent
            children[parent].append((child, lag))
    tail = [0] * len(ops)
    for parent in reversed(range(len(ops))):
        for child, lag in children[parent]:
            tail[parent] = max(tail[parent], lag + tail[child])

    horizon = int(os.environ.get("TARGET", "959"))
    latest = [horizon - 1 - value for value in tail]
    print(
        f"horizon={horizon} dag_lb={max(earliest) + 1} "
        f"ops={len(ops)}"
    )
    if "TRACE_INDEX" in os.environ:
        node = int(os.environ["TRACE_INDEX"])
        chain = []
        while node >= 0:
            chain.append(node)
            node = reason[node]
        print("trace:")
        for index in reversed(chain):
            op = ops[index]
            print(
                f"  i={index:5d} es={earliest[index]:3d} "
                f"g={op.group:2d} r={op.round:2d} {op.engine:5s} {op.tag}"
            )
    for engine, capacity in SLOT_LIMITS.items():
        if engine == "debug":
            continue
        matrix = np.zeros((horizon, horizon), dtype=np.int32)
        invalid = 0
        for index, op in enumerate(ops):
            if op.engine != engine:
                continue
            lower = earliest[index]
            upper = latest[index]
            if lower > upper or lower >= horizon or upper < 0:
                invalid += 1
                continue
            matrix[lower, upper] += 1
        # contained[a, b] counts jobs whose complete feasible domain
        # [earliest, latest] is contained in interval [a, b].
        contained = matrix[::-1].cumsum(axis=0)[::-1].cumsum(axis=1)
        best = (-10**9, 0, 0, 0)
        for left in range(horizon):
            counts = contained[left, left:]
            lengths = np.arange(1, horizon - left + 1, dtype=np.int32)
            overloads = counts - capacity * lengths
            offset = int(np.argmax(overloads))
            candidate = int(overloads[offset])
            if candidate > best[0]:
                right = left + offset
                best = (candidate, left, right, int(counts[offset]))
        overload, left, right, count = best
        print(
            f"{engine:5s} count={sum(op.engine == engine for op in ops):5d} "
            f"invalid={invalid} max_overload={overload:4d} "
            f"window=[{left},{right}] jobs={count} "
            f"capacity={capacity * (right - left + 1)}"
        )
        if bool(int(os.environ.get("WINDOW_DETAILS", "0"))) and overload >= 0:
            contained_indices = [
                index
                for index, op in enumerate(ops)
                if op.engine == engine
                and earliest[index] >= left
                and latest[index] <= right
            ]
            boundary = sorted(
                contained_indices,
                key=lambda index: (
                    min(earliest[index] - left, right - latest[index]),
                    earliest[index],
                    -latest[index],
                ),
            )[:40]
            for index in boundary:
                op = ops[index]
                print(
                    f"  i={index:5d} es={earliest[index]:3d} "
                    f"ls={latest[index]:3d} g={op.group:2d} r={op.round:2d} "
                    f"{op.tag}"
                )


if __name__ == "__main__":
    main()
