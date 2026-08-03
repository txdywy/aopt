"""Reschedule a fixed suffix/window without forcing a one-cycle deletion.

``repair_schedule`` is conservative: it only accepts a calendar after
removing one named cycle and shifts every later operation by exactly one.  A
near-saturated VLIW schedule often needs a local exchange first (move a late
VALU/ALU chain into an earlier hole, then move a different chain right), so
the deletion neighbourhood can report ``stuck`` even when a shorter calendar
exists.  This tool fixes operations outside a window and lets CP-SAT
repack the complete DAG inside it.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path

from ortools.sat.python import cp_model

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops, validate
from problem import SLOT_LIMITS


def main() -> None:
    configure_target()
    source = Path(os.environ["SCHEDULE_IN"])
    cycles = list(json.loads(source.read_text())["cycles"])
    kernel.SCHEDULE_EXACT_CYCLES = None
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except Exception:
        if not hasattr(builder, "dag_ops"):
            raise
    ops = real_tail_ops(builder.dag_ops)
    validate(ops, cycles)
    old_horizon = max(cycles) + 1
    lo = int(os.environ.get("WINDOW_LO", max(0, old_horizon - int(os.environ.get("WINDOW", "192")))))
    hi = int(os.environ.get("WINDOW_HI", old_horizon - 1))
    target_hi = int(os.environ.get("TARGET_HI", hi))
    if not 0 <= lo <= hi < old_horizon:
        raise ValueError(f"bad window {lo}:{hi} for horizon {old_horizon}")
    local = [i for i, cycle in enumerate(cycles) if lo <= cycle <= hi]
    local_set = set(local)
    if not local:
        raise ValueError("empty local window")
    model = cp_model.CpModel()
    starts = {
        i: model.new_int_var(lo, target_hi, f"s{i}")
        for i in local
    }
    # Keep every original precedence edge, including zero-lag WAR edges.
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            if child in local_set and parent in local_set:
                model.add(starts[child] >= starts[parent] + lag)
            elif child in local_set:
                model.add(starts[child] >= cycles[parent] + lag)
            elif parent in local_set:
                model.add(cycles[child] >= starts[parent] + lag)
            elif cycles[child] < cycles[parent] + lag:
                raise AssertionError("input schedule is not a DAG witness")

    fixed_use: dict[str, Counter[int]] = defaultdict(Counter)
    for i, op in enumerate(ops):
        if i not in local_set and lo <= cycles[i] <= target_hi:
            fixed_use[op.engine][cycles[i]] += 1
    for engine, capacity in SLOT_LIMITS.items():
        if engine == "debug":
            continue
        intervals = []
        demands = []
        for i in local:
            if ops[i].engine == engine:
                intervals.append(model.new_fixed_size_interval_var(starts[i], 1, f"i{i}"))
                demands.append(1)
        for cycle, demand in fixed_use[engine].items():
            intervals.append(model.new_fixed_size_interval_var(cycle, 1, f"fixed_{engine}_{cycle}"))
            demands.append(demand)
        if intervals:
            model.add_cumulative(intervals, demands, capacity)

    # Minimize the suffix endpoint first, then total movement.  A tiny
    # movement term makes CP-SAT retain the phase teacher whenever multiple
    # calendars have the same endpoint, which keeps later windows stable.
    endpoint = model.new_int_var(lo, target_hi, "endpoint")
    for i in local:
        model.add(endpoint >= starts[i])
    model.minimize(endpoint * 100000 + sum(starts[i] for i in local))
    for i in local:
        model.add_hint(starts[i], cycles[i])
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(os.environ.get("TIME_LIMIT", "20"))
    solver.parameters.num_workers = int(os.environ.get("WORKERS", "8"))
    solver.parameters.random_seed = int(os.environ.get("RANDOM_SEED", "1"))
    solver.parameters.repair_hint = bool(int(os.environ.get("REPAIR_HINT", "0")))
    solver.parameters.log_search_progress = bool(int(os.environ.get("LOG", "0")))
    status = solver.solve(model)
    print(
        f"status={solver.status_name(status)} window={lo}:{hi} target_hi={target_hi} "
        f"local={len(local)} wall={solver.wall_time:.2f}s",
        flush=True,
    )
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return
    result = cycles[:]
    for i in local:
        result[i] = solver.value(starts[i])
    validate(ops, result)
    out = Path(os.environ.get("SCHEDULE_OUT", "/tmp/aopt-window.json"))
    out.write_text(json.dumps({"makespan": max(result) + 1, "cycles": result}))
    print(f"makespan={max(result)+1} output={out}", flush=True)


if __name__ == "__main__":
    main()
