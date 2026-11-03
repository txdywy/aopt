"""CP-SAT large-neighborhood scheduler that releases a suffix of SIMD groups."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path

from ortools.sat.python import cp_model

import codex.perf_takehome_under1000 as kernel
from problem import SLOT_LIMITS


def main() -> None:
    first_group = int(os.environ.get("MIN_GROUP", "16"))
    horizon = int(os.environ.get("TARGET", "999"))
    limit = float(os.environ.get("TIME_LIMIT", "180"))
    builder = kernel.KernelBuilder()
    builder.build_kernel(10, 2047, 256, 16)
    ops = builder.dag_ops
    _, old = builder._schedule(ops, kernel.SCHEDULE_POLICIES[0], return_cycles=True)

    # Release the selected group suffix and every operation which otherwise
    # lies beyond the requested horizon (notably rolling output pointers).
    local_set = {
        i for i, op in enumerate(ops)
        if op.group >= first_group or old[i] >= horizon
    }
    # Close over zero-lag cross-boundary cycles in both directions.  A fixed
    # operation beyond a variable parent would otherwise pin its latest time.
    changed = True
    while changed:
        changed = False
        for child, op in enumerate(ops):
            for parent, lag in op.parents.items():
                if not lag and ((child in local_set) != (parent in local_set)):
                    candidate = parent if child in local_set else child
                    if candidate not in local_set:
                        local_set.add(candidate)
                        changed = True
    local = sorted(local_set)
    print(
        f"old={max(old)+1} groups={first_group}..31 target={horizon} local={len(local)}",
        flush=True,
    )

    model = cp_model.CpModel()
    starts = {}
    for i in local:
        lower = 0
        upper = horizon - 1
        for parent, lag in ops[i].parents.items():
            if parent not in local_set:
                lower = max(lower, old[parent] + lag)
        starts[i] = model.new_int_var(lower, upper, f"s{i}")

    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            child_local = child in local_set
            parent_local = parent in local_set
            if child_local and parent_local:
                model.add(starts[child] >= starts[parent] + lag)
            elif child_local:
                model.add(starts[child] >= old[parent] + lag)
            elif parent_local:
                model.add(old[child] >= starts[parent] + lag)

    fixed_use: dict[str, Counter[int]] = defaultdict(Counter)
    for i, op in enumerate(ops):
        if i not in local_set and old[i] < horizon:
            fixed_use[op.engine][old[i]] += 1

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
            intervals.append(model.new_fixed_size_interval_var(cycle, 1, f"f_{engine}_{cycle}"))
            demands.append(demand)
        model.add_cumulative(intervals, demands, capacity)

    for i in local:
        model.add_hint(starts[i], min(old[i], horizon - 1))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = limit
    solver.parameters.num_workers = int(os.environ.get("WORKERS", "8"))
    solver.parameters.repair_hint = True
    solver.parameters.hint_conflict_limit = 500_000
    solver.parameters.cp_model_presolve = True
    solver.parameters.log_search_progress = bool(int(os.environ.get("LOG", "0")))
    status = solver.solve(model)
    print("status", solver.status_name(status), flush=True)
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        result = old.copy()
        for i in local:
            result[i] = solver.value(starts[i])
        output = Path(os.environ.get("OUT", "/tmp/aopt-group-schedule.json"))
        output.write_text(json.dumps({"makespan": max(result) + 1, "cycles": result}))
        print("makespan", max(result) + 1, "output", output, flush=True)


if __name__ == "__main__":
    main()
