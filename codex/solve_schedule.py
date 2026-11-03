"""Exact CP-SAT scheduling experiment for the extreme VLIW DAG."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ortools.sat.python import cp_model

import codex.perf_takehome_under1000 as kernel
from problem import SLOT_LIMITS


def main() -> None:
    scalar_count = int(os.environ.get("SCALAR_COUNT", "69"))
    kernel.HASH_SCALAR_EXTRA = frozenset(
        kernel._BASE_SCALAR | set(kernel._SCALAR_CANDIDATES[:scalar_count])
    )
    kernel.PER_GROUP_OUTPUT_POINTERS = bool(
        int(os.environ.get("PRIVATE_OUTPUT", "0"))
    )
    builder = kernel.KernelBuilder()
    builder.build_kernel(10, 2047, 256, 16)
    ops = builder.dag_ops
    _, hint = builder._schedule(ops, 36, return_cycles=True)

    horizon = int(os.environ.get("TARGET", "999"))
    model = cp_model.CpModel()
    hint_horizon = max(hint) or 1
    compressed_hint = [
        round(value * (horizon - 1) / hint_horizon) for value in hint
    ]
    band = int(os.environ.get("BAND", "0"))
    starts = [
        model.new_int_var(
            max(0, compressed_hint[i] - band) if band else 0,
            min(horizon - 1, compressed_hint[i] + band) if band else horizon - 1,
            f"s{i}",
        )
        for i in range(len(ops))
    ]
    intervals = [
        model.new_fixed_size_interval_var(starts[i], 1, f"i{i}")
        for i in range(len(ops))
    ]

    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            model.add(starts[child] >= starts[parent] + lag)

    for engine, capacity in SLOT_LIMITS.items():
        if engine == "debug":
            continue
        chosen = [intervals[i] for i, op in enumerate(ops) if op.engine == engine]
        model.add_cumulative(chosen, [1] * len(chosen), capacity)

    # Compress the best list schedule into the requested horizon as a dense,
    # near-feasible starting point.  CP-SAT's hint repair then focuses on the
    # small set of capacity/dependency collisions instead of rediscovering the
    # full software-pipeline phase structure from scratch.
    for variable, value in zip(starts, compressed_hint):
        model.add_hint(variable, value)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(os.environ.get("TIME_LIMIT", "600"))
    solver.parameters.num_workers = 8
    solver.parameters.log_search_progress = True
    solver.parameters.cp_model_presolve = True
    solver.parameters.repair_hint = True
    solver.parameters.hint_conflict_limit = 200_000
    status = solver.solve(model)
    print("status", solver.status_name(status))
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        cycles = [solver.value(start) for start in starts]
        makespan = max(cycles) + 1
        result = {
            "cycles": cycles,
            "makespan": makespan,
        }
        output = Path("/tmp/aopt-cpsat-cycles.json")
        output.write_text(json.dumps(result))
        print("makespan", result["makespan"])
        print("output", output)


if __name__ == "__main__":
    main()
