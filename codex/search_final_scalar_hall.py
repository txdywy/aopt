"""Rank final-hash scalarization placements by exact Hall pressure.

Each enabled placement trades one VALU operation for eight scalar ALU
operations.  Aggregate counts alone cannot distinguish good placements:
late groups may place some lanes in the two cycles outside the saturated ALU
window, while an early placement traps all eight lanes inside it.
"""

from __future__ import annotations

from collections import Counter
import json
import os

import numpy as np

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from problem import SLOT_LIMITS


ATTRS = {
    "C5": "SCALAR_FINAL_C5_SET",
    "JOIN": "SCALAR_FINAL_JOIN_SET",
    "SHIFT": "SCALAR_FINAL_SHIFT_SET",
    "H23": "SCALAR_FINAL_HASH23_JOIN_SET",
    "H4": "SCALAR_FINAL_HASH4_SET",
}


def build_ops() -> list[kernel._Op]:
    kernel.SCHEDULE_EXACT_CYCLES = []
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration, ValueError):
        if not hasattr(builder, "dag_ops"):
            raise
    return real_tail_ops(builder.dag_ops)


def hall(
    ops: list[kernel._Op],
    earliest: list[int],
    latest: list[int],
    engine: str,
    horizon: int,
) -> tuple[int, int, int]:
    matrix = np.zeros((horizon, horizon), dtype=np.int32)
    for index, op in enumerate(ops):
        if op.engine != engine:
            continue
        lower, upper = earliest[index], latest[index]
        if 0 <= lower <= upper < horizon:
            matrix[lower, upper] += 1
    contained = matrix[::-1].cumsum(axis=0)[::-1].cumsum(axis=1)
    capacity = SLOT_LIMITS[engine]
    best = (-10**9, 0, 0)
    for left in range(horizon):
        lengths = np.arange(1, horizon - left + 1, dtype=np.int32)
        overloads = contained[left, left:] - capacity * lengths
        offset = int(np.argmax(overloads))
        candidate = int(overloads[offset])
        if candidate > best[0]:
            best = (candidate, left, left + offset)
    return best


def evaluate(ops: list[kernel._Op], horizon: int) -> dict[str, object]:
    earliest = [0] * len(ops)
    children: list[list[tuple[int, int]]] = [[] for _ in ops]
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            earliest[child] = max(earliest[child], earliest[parent] + lag)
            children[parent].append((child, lag))
    tail = [0] * len(ops)
    for parent in reversed(range(len(ops))):
        tail[parent] = max(
            (lag + tail[child] for child, lag in children[parent]),
            default=0,
        )
    latest = [horizon - 1 - value for value in tail]
    counts = Counter(op.engine for op in ops)
    return {
        "ops": len(ops),
        "alu_count": counts["alu"],
        "valu_count": counts["valu"],
        "alu": hall(ops, earliest, latest, "alu", horizon),
        "valu": hall(ops, earliest, latest, "valu", horizon),
    }


def main() -> None:
    configure_target()
    horizon = int(os.environ.get("TARGET", "962"))
    names = tuple(
        value
        for value in os.environ.get(
            "OPTIONS", ",".join(ATTRS)
        ).split(",")
        if value
    )
    unknown = set(names) - set(ATTRS)
    if unknown:
        raise ValueError(f"unknown OPTIONS: {sorted(unknown)}")
    groups = tuple(
        int(value)
        for value in os.environ.get(
            "GROUPS", ",".join(map(str, range(kernel.N_GROUPS)))
        ).split(",")
        if value
    )
    base = {
        name: frozenset(getattr(kernel, attribute))
        for name, attribute in ATTRS.items()
    }
    baseline = evaluate(build_ops(), horizon)
    print("baseline=" + json.dumps(baseline, separators=(",", ":")), flush=True)

    results = []
    for name in names:
        attribute = ATTRS[name]
        for group in groups:
            for candidate_name, candidate_attribute in ATTRS.items():
                setattr(kernel, candidate_attribute, base[candidate_name])
            values = set(base[name])
            action = "remove" if group in values else "add"
            values.symmetric_difference_update((group,))
            setattr(kernel, attribute, frozenset(values))
            try:
                result = evaluate(build_ops(), horizon)
            except (AssertionError, StopIteration, ValueError):
                continue
            result.update(
                {
                    "option": name,
                    "group": group,
                    "action": action,
                }
            )
            results.append(result)

    baseline_alu = int(baseline["alu"][0])
    baseline_valu = int(baseline["valu"][0])
    results.sort(
        key=lambda result: (
            int(result["alu"][0]),
            int(result["valu"][0]),
            str(result["action"]),
            str(result["option"]),
            int(result["group"]),
        )
    )
    for result in results:
        result["alu_delta"] = int(result["alu"][0]) - baseline_alu
        result["valu_delta"] = int(result["valu"][0]) - baseline_valu
        print(json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
