"""Rank single-engine schedules by remaining engines' Hall-window overload."""

from __future__ import annotations

import glob
import json
import os

import numpy as np

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from problem import SLOT_LIMITS


def hall_overload(
    ops: list[kernel._Op],
    earliest: list[int],
    latest: list[int],
    engine: str,
    horizon: int,
) -> int:
    matrix = np.zeros((horizon, horizon), dtype=np.int32)
    for index, op in enumerate(ops):
        if op.engine == engine:
            matrix[earliest[index], latest[index]] += 1
    contained = matrix[::-1].cumsum(axis=0)[::-1].cumsum(axis=1)
    capacity = SLOT_LIMITS[engine]
    best = -10**9
    for left in range(horizon):
        lengths = np.arange(1, horizon - left + 1, dtype=np.int32)
        best = max(
            best,
            int(np.max(contained[left, left:] - capacity * lengths)),
        )
    return best


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
    horizon = int(os.environ.get("TARGET", "959"))
    fixed_engine = os.environ.get("ENGINE", "flow")
    pattern = os.environ.get("HINT_GLOB", "/tmp/aopt-flow-policy-*.json")
    ranked = []
    for filename in glob.glob(pattern):
        cycles = json.loads(open(filename).read())["cycles"]
        earliest = [0] * len(ops)
        valid = True
        for child, op in enumerate(ops):
            required = max(
                (earliest[parent] + lag for parent, lag in op.parents.items()),
                default=0,
            )
            if op.engine == fixed_engine:
                earliest[child] = cycles[child]
                valid &= cycles[child] >= required
            else:
                earliest[child] = required
        latest = [horizon - 1] * len(ops)
        for index, op in enumerate(ops):
            if op.engine == fixed_engine:
                latest[index] = cycles[index]
        children: list[list[tuple[int, int]]] = [[] for _ in ops]
        for child, op in enumerate(ops):
            for parent, lag in op.parents.items():
                children[parent].append((child, lag))
        for parent in reversed(range(len(ops))):
            upper = min(
                (latest[child] - lag for child, lag in children[parent]),
                default=horizon - 1,
            )
            if ops[parent].engine == fixed_engine:
                valid &= cycles[parent] <= upper
            else:
                latest[parent] = min(latest[parent], upper)
            valid &= earliest[parent] <= latest[parent]
        if not valid:
            ranked.append(((10**9,), filename, ()))
            continue
        overloads = tuple(
            hall_overload(ops, earliest, latest, engine, horizon)
            for engine in ("alu", "valu", "load", "store")
        )
        key = (max(overloads), sum(max(0, value) for value in overloads))
        ranked.append((key, filename, overloads))
    for key, filename, overloads in sorted(ranked)[:30]:
        print(
            f"max={key[0]:4d} sum={key[1] if len(key) > 1 else -1:4d} "
            f"overloads={overloads} {filename}"
        )


if __name__ == "__main__":
    main()
