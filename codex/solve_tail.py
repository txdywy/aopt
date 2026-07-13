"""Exact epilogue rescheduler with the steady-state prefix held fixed."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from dataclasses import replace

from ortools.sat.python import cp_model

import codex.perf_takehome_under1000 as kernel
from problem import SLOT_LIMITS


def main() -> None:
    cutoff = int(os.environ.get("CUTOFF", "850"))
    horizon = int(os.environ.get("TARGET", "999"))
    limit = float(os.environ.get("TIME_LIMIT", "180"))
    early_slack = int(os.environ.get("EARLY_SLACK", "0"))
    ancestor_steps = int(os.environ.get("ANCESTOR_STEPS", "0"))
    kernel.PREPROCESS_MAX_DEPTH = int(os.environ.get("PREPROCESS_DEPTH", "4"))
    scalar_count = int(os.environ.get("SCALAR_COUNT", "65"))
    kernel.HASH_SCALAR_EXTRA = frozenset(
        kernel._BASE_SCALAR | set(kernel._SCALAR_CANDIDATES[:scalar_count])
    )
    def group_set(name: str) -> frozenset[int]:
        value = os.environ.get(name, "").strip()
        return frozenset(int(x) for x in value.split(",") if x)

    kernel.SCALAR_FINAL_C5_SET = group_set("SCALAR_FINAL_C5")
    kernel.SCALAR_FINAL_JOIN_SET = group_set("SCALAR_FINAL_JOIN")
    kernel.SCALAR_FINAL_SHIFT_SET = group_set("SCALAR_FINAL_SHIFT")
    kernel.SCALAR_FINAL_HASH23_JOIN_SET = group_set("SCALAR_FINAL_HASH23_JOIN")
    if "OFFSET14" in os.environ:
        offsets = list(kernel.FULL_ROUND_OFFSETS)
        offsets[14] = int(os.environ["OFFSET14"])
        kernel.FULL_ROUND_OFFSETS = tuple(offsets)
    policy = int(os.environ.get("POLICY", str(kernel.SCHEDULE_POLICIES[0])))

    builder = kernel.KernelBuilder()
    builder.build_kernel(10, 2047, 256, 16)
    ops = builder.dag_ops
    _, old = builder._schedule(ops, policy, return_cycles=True)
    if bool(int(os.environ.get("DROP_ZERO", "0"))):
        ops = [
            replace(op, parents={p: lag for p, lag in op.parents.items() if lag})
            for op in ops
        ]
    local_set = {i for i, cycle in enumerate(old) if cycle >= cutoff}
    frontier = set(local_set)
    for _ in range(ancestor_steps):
        parents = {
            parent
            for child in frontier
            for parent in ops[child].parents
            if old[parent] >= cutoff - early_slack
        }
        frontier = parents - local_set
        if not frontier:
            break
        local_set.update(frontier)
    local = sorted(local_set)
    print(f"old={max(old)+1} cutoff={cutoff} target={horizon} local={len(local)}", flush=True)

    counts = Counter(op.engine for i, op in enumerate(ops) if i in local_set)
    print("tail counts", counts, flush=True)
    for engine, count in counts.items():
        print(engine, "lower", (count + SLOT_LIMITS[engine] - 1) // SLOT_LIMITS[engine], flush=True)

    model = cp_model.CpModel()
    domain_start = max(0, cutoff - early_slack)
    starts = {
        i: model.new_int_var(domain_start, horizon - 1, f"s{i}") for i in local
    }
    for child, op in enumerate(ops):
        child_local = child in local_set
        for parent, lag in op.parents.items():
            parent_local = parent in local_set
            if child_local and parent_local:
                model.add(starts[child] >= starts[parent] + lag)
            elif child_local:
                model.add(starts[child] >= old[parent] + lag)
            elif parent_local:
                model.add(old[child] >= starts[parent] + lag)

    fixed_use: dict[str, Counter[int]] = defaultdict(Counter)
    for i, op in enumerate(ops):
        if i not in local_set and domain_start <= old[i] < horizon:
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

    old_span = max(old) - cutoff or 1
    new_span = horizon - 1 - cutoff
    for i in local:
        hint = max(
            domain_start,
            cutoff + round((old[i] - cutoff) * new_span / old_span),
        )
        model.add_hint(starts[i], hint)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = limit
    solver.parameters.num_workers = int(os.environ.get("WORKERS", "8"))
    solver.parameters.repair_hint = bool(int(os.environ.get("REPAIR_HINT", "0")))
    solver.parameters.hint_conflict_limit = 200_000
    solver.parameters.cp_model_presolve = True
    solver.parameters.log_search_progress = bool(int(os.environ.get("LOG", "0")))
    status = solver.solve(model)
    print("status", solver.status_name(status), flush=True)
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        result = old.copy()
        for i in local:
            result[i] = solver.value(starts[i])
        output = Path(os.environ.get("OUT", "/tmp/aopt-tail-schedule.json"))
        output.write_text(json.dumps({"makespan": max(result) + 1, "cycles": result}))
        print("makespan", max(result) + 1, "output", output, flush=True)


if __name__ == "__main__":
    main()
