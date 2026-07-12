"""Iteratively remove cycles from an existing legal VLIW schedule.

This is a large-neighborhood instruction scheduler: operations outside a
small time window are fixed, while CP-SAT is allowed to reschedule the window.
Each successful solve removes exactly one cycle and becomes the seed for the
next repair.  It is dramatically smaller than solving the 20k-op DAG globally.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import time

from ortools.sat.python import cp_model

import codex.perf_takehome_under1000 as kernel
from problem import SLOT_LIMITS


OUT = Path(os.environ.get("SCHEDULE_OUT", "/tmp/aopt-repaired-schedule.json"))
TARGET = int(os.environ.get("TARGET", "999"))
TIME_LIMIT = float(os.environ.get("TIME_LIMIT", "8"))
WORKERS = int(os.environ.get("WORKERS", "8"))
WINDOWS = tuple(int(x) for x in os.environ.get("WINDOWS", "18,28,42,64").split(","))
MAX_CANDIDATES = int(os.environ.get("MAX_CANDIDATES", "24"))


def validate(ops: list[kernel._Op], cycles: list[int]) -> None:
    assert len(ops) == len(cycles)
    assert min(cycles) >= 0
    usage: dict[tuple[int, str], int] = Counter(
        (cycles[i], op.engine) for i, op in enumerate(ops)
    )
    for (cycle, engine), count in usage.items():
        assert count <= SLOT_LIMITS[engine], (cycle, engine, count)
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            assert cycles[child] >= cycles[parent] + lag, (
                parent,
                child,
                lag,
                cycles[parent],
                cycles[child],
            )


def deletion_candidates(
    ops: list[kernel._Op], cycles: list[int], radius: int
) -> list[int]:
    """Return resource-feasible windows, preferring the most aggregate slack."""
    horizon = max(cycles) + 1
    use = [Counter() for _ in range(horizon)]
    for i, op in enumerate(ops):
        use[cycles[i]][op.engine] += 1

    prefix = {e: [0] * (horizon + 1) for e in SLOT_LIMITS if e != "debug"}
    for t in range(horizon):
        for engine in prefix:
            prefix[engine][t + 1] = prefix[engine][t] + use[t][engine]

    ranked = []
    margin = radius + 2
    new_width = 2 * radius + 1
    for deleted in range(margin, horizon - margin):
        # The mapped [d-r, d+r] neighborhood contains old cycles
        # [d-r, d+r+1], hence one extra cycle of work.
        old_lo = deleted - radius
        old_hi = deleted + radius + 1
        counts = {
            e: prefix[e][old_hi + 1] - prefix[e][old_lo] for e in prefix
        }
        if any(counts[e] > SLOT_LIMITS[e] * new_width for e in counts):
            continue
        normalized_slack = sum(
            (SLOT_LIMITS[e] * new_width - counts[e]) / SLOT_LIMITS[e]
            for e in counts
        )
        ranked.append((-normalized_slack, use[deleted]["alu"], deleted))
    return [item[2] for item in sorted(ranked)[:MAX_CANDIDATES]]


def repair_one(
    ops: list[kernel._Op],
    old: list[int],
    deleted: int,
    radius: int,
) -> list[int] | None:
    old_horizon = max(old) + 1
    horizon = old_horizon - 1
    lo = max(0, deleted - radius)
    hi = min(horizon - 1, deleted + radius)

    # The natural schedule after deleting a cycle.  Every operation whose old
    # position maps into [lo, hi] is part of the neighborhood; the rest is fixed.
    base = [t if t < deleted else t - 1 for t in old]
    local = [i for i, t in enumerate(base) if lo <= t <= hi]
    local_set = set(local)

    model = cp_model.CpModel()
    starts: dict[int, cp_model.IntVar] = {
        i: model.new_int_var(lo, hi, f"s{i}") for i in local
    }

    # Precedence edges become either ordinary difference constraints or bounds
    # against fixed operations.  Zero-lag anti-dependencies are preserved too.
    for child, op in enumerate(ops):
        child_local = child in local_set
        for parent, lag in op.parents.items():
            parent_local = parent in local_set
            if child_local and parent_local:
                model.add(starts[child] >= starts[parent] + lag)
            elif child_local:
                model.add(starts[child] >= base[parent] + lag)
            elif parent_local:
                model.add(base[child] >= starts[parent] + lag)
            elif base[child] < base[parent] + lag:
                # The deleted cycle crossed a fixed positive-latency edge.  This
                # candidate needs a wider neighborhood including an endpoint.
                return None

    # Cumulative resources include one aggregated fixed-profile interval per
    # occupied cycle, avoiding thousands of redundant fixed task variables.
    fixed_use: dict[str, Counter[int]] = defaultdict(Counter)
    for i, op in enumerate(ops):
        if i not in local_set and lo <= base[i] <= hi:
            fixed_use[op.engine][base[i]] += 1

    for engine, capacity in SLOT_LIMITS.items():
        if engine == "debug":
            continue
        intervals = []
        demands = []
        for i in local:
            if ops[i].engine == engine:
                intervals.append(model.new_fixed_size_interval_var(starts[i], 1, f"i{i}"))
                demands.append(1)
        for t, demand in fixed_use[engine].items():
            intervals.append(model.new_fixed_size_interval_var(t, 1, f"fixed_{engine}_{t}"))
            demands.append(demand)
        if intervals:
            model.add_cumulative(intervals, demands, capacity)

    for i in local:
        model.add_hint(starts[i], base[i])

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = TIME_LIMIT
    solver.parameters.num_workers = WORKERS
    solver.parameters.cp_model_presolve = True
    solver.parameters.repair_hint = True
    solver.parameters.hint_conflict_limit = 100_000
    status = solver.solve(model)
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return None

    result = base.copy()
    for i in local:
        result[i] = solver.value(starts[i])
    validate(ops, result)
    return result


def save(cycles: list[int]) -> None:
    OUT.write_text(json.dumps({"makespan": max(cycles) + 1, "cycles": cycles}))


def main() -> None:
    builder = kernel.KernelBuilder()
    builder.build_kernel(10, 2047, 256, 16)
    ops = builder.dag_ops
    _, cycles = builder._schedule(ops, kernel.SCHEDULE_POLICIES[0], return_cycles=True)
    if OUT.exists():
        saved = json.loads(OUT.read_text())
        if len(saved.get("cycles", ())) == len(ops):
            cycles = saved["cycles"]
            validate(ops, cycles)
    print(f"start={max(cycles) + 1} target={TARGET} ops={len(ops)}", flush=True)

    while max(cycles) + 1 > TARGET:
        before = max(cycles) + 1
        success = None
        attempts = 0
        started = time.monotonic()
        for radius in WINDOWS:
            candidates = deletion_candidates(ops, cycles, radius)
            print(f"radius={radius} candidates={candidates}", flush=True)
            for deleted in candidates:
                attempts += 1
                candidate = repair_one(ops, cycles, deleted, radius)
                if candidate is not None:
                    success = candidate
                    print(
                        f"{before}->{before - 1} delete={deleted} radius={radius} "
                        f"attempts={attempts} elapsed={time.monotonic() - started:.1f}s",
                        flush=True,
                    )
                    break
            if success is not None:
                break
        if success is None:
            print(f"stuck={before} attempts={attempts}", flush=True)
            break
        cycles = success
        save(cycles)

    print(f"final={max(cycles) + 1} output={OUT}", flush=True)


if __name__ == "__main__":
    main()
