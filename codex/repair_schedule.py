"""Iteratively remove cycles from an existing legal VLIW schedule.

This is a large-neighborhood instruction scheduler: operations outside a
small time window are fixed, while CP-SAT is allowed to reschedule the window.
Each successful solve removes exactly one cycle and becomes the seed for the
next repair.  It is dramatically smaller than solving the 20k-op DAG globally.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import json
import os
from pathlib import Path
import time

from ortools.sat.python import cp_model

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from problem import SLOT_LIMITS


OUT = Path(os.environ.get("SCHEDULE_OUT", "/tmp/aopt-repaired-schedule.json"))
TARGET = int(os.environ.get("TARGET", "999"))
TIME_LIMIT = float(os.environ.get("TIME_LIMIT", "8"))
WORKERS = int(os.environ.get("WORKERS", "8"))
WINDOWS = tuple(int(x) for x in os.environ.get("WINDOWS", "18,28,42,64").split(","))
MAX_CANDIDATES = int(os.environ.get("MAX_CANDIDATES", "24"))


def model_real_tail_stores(ops: list[kernel._Op]) -> list[kernel._Op]:
    """Match the independent late-store postpass used by the real kernel."""
    if not bool(int(os.environ.get("REAL_TAIL_STORES", "1"))):
        return ops
    first_tail_group = kernel.N_GROUPS - kernel.INDEPENDENT_TAIL_GROUP_COUNT
    last_prefix_store = max(
        i
        for i, op in enumerate(ops)
        if op.tag == "output_store"
        and op.group is not None
        and op.group < first_tail_group
    )
    adjusted = []
    for i, op in enumerate(ops):
        if op.tag == "pointer_advance" and i > last_prefix_store:
            adjusted.append(
                replace(op, engine="debug", parents={}, reads=(), writes=())
            )
        elif (
            op.tag == "output_store"
            and op.group is not None
            and op.group >= first_tail_group
        ):
            adjusted.append(
                replace(
                    op,
                    parents={
                        parent: lag
                        for parent, lag in op.parents.items()
                        if ops[parent].tag not in {"output_pointer", "pointer_advance"}
                    },
                )
            )
        else:
            adjusted.append(op)
    return adjusted


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
    excluded = {
        int(value)
        for value in os.environ.get("DELETE_EXCLUDE", "").split(",")
        if value
    }
    margin = radius + 2
    new_width = 2 * radius + 1
    delete_min = max(margin, int(os.environ.get("DELETE_MIN", str(margin))))
    delete_max = min(
        horizon - margin,
        int(os.environ.get("DELETE_MAX", str(horizon - margin))),
    )
    for deleted in range(delete_min, delete_max):
        if deleted in excluded:
            continue
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
    ordered = [item[2] for item in sorted(ranked)]
    offset = int(os.environ.get("CANDIDATE_OFFSET", "0"))
    return ordered[offset : offset + MAX_CANDIDATES]


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

    if bool(int(os.environ.get("PACK_BRANCH_TRACES", "1"))):
        copy_tags = {
            "direct_branch_copy",
            "paired_direct_branch_copy",
            "paired_branch_copy",
            "paired_branch_delayed_copy",
        }
        target_tags = {
            "direct_branch_target",
            "paired_direct_branch_target",
            "paired_branch_target",
            "paired_branch_delayed_target",
        }

        def exact_distance(left: int, right: int, distance: int) -> bool:
            left_local = left in local_set
            right_local = right in local_set
            if left_local and right_local:
                model.add(starts[left] == starts[right] + distance)
            elif left_local:
                model.add(starts[left] == base[right] + distance)
            elif right_local:
                model.add(base[left] == starts[right] + distance)
            elif base[left] != base[right] + distance:
                return False
            return True

        for child, op in enumerate(ops):
            if op.tag in copy_tags:
                jump_parent = next(
                    (
                        parent
                        for parent in op.parents
                        if ops[parent].tag.endswith("branch_jump")
                    ),
                    None,
                )
                if jump_parent is not None and not exact_distance(
                    child, jump_parent, 1
                ):
                    return None
            elif op.tag in target_tags:
                for parent in op.parents:
                    if ops[parent].tag in copy_tags and not exact_distance(
                        child, parent, 0
                    ):
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
    # OR-Tools 9.15 can abort in MinimizeL1DistanceWithHint when repair_hint
    # races with multi-worker fixed search on larger neighborhoods.  Keep the
    # ordinary hints, but make the unstable repair mode explicitly opt-in.
    solver.parameters.repair_hint = bool(int(os.environ.get("REPAIR_HINT", "0")))
    solver.parameters.hint_conflict_limit = 100_000
    solver.parameters.random_seed = int(os.environ.get("RANDOM_SEED", "1"))
    solver.parameters.randomize_search = bool(
        int(os.environ.get("RANDOMIZE_SEARCH", "0"))
    )
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
    configure_target()
    input_path = os.environ.get("SCHEDULE_IN")
    input_cycles = None
    if input_path:
        input_cycles = json.loads(Path(input_path).read_text())["cycles"]
        kernel.SCHEDULE_EXACT_CYCLES = input_cycles
    builder = kernel.KernelBuilder()
    builder.build_kernel(10, 2047, 256, 16)
    original_ops = builder.dag_ops
    if input_cycles is None:
        _, cycles = builder._schedule(
            original_ops, kernel.SCHEDULE_POLICIES[0], return_cycles=True
        )
        ops = model_real_tail_stores(original_ops)
    else:
        cycles = input_cycles
        ops = real_tail_ops(original_ops)
        validate(ops, cycles)
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
