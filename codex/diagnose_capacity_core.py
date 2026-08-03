"""Find a sufficient physical-engine capacity core at a fixed horizon.

This is an offline compiler diagnostic.  It keeps the complete semantic DAG
and makes each cumulative engine constraint optional under a CP-SAT
assumption.  The returned core answers a useful question that per-engine Hall
cuts cannot: which *combination* of otherwise individually schedulable
resources creates the 959-cycle contradiction?
"""

from __future__ import annotations

import os

from ortools.sat.python import cp_model

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from codex.solve_full_dag import topological_data
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
    _, _, earliest, tail = topological_data(ops)
    horizon = int(os.environ.get("TARGET", "959"))
    model = cp_model.CpModel()
    starts = [
        model.new_int_var(earliest[i], horizon - 1 - tail[i], f"s{i}")
        for i in range(len(ops))
    ]
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            model.add(starts[child] >= starts[parent] + lag)

    assumptions = {}
    for engine, capacity in SLOT_LIMITS.items():
        indices = [i for i, op in enumerate(ops) if op.engine == engine]
        if not indices:
            continue
        intervals = [
            model.new_fixed_size_interval_var(starts[i], 1, f"i{i}")
            for i in indices
        ]
        literal = model.new_bool_var(f"cap_{engine}")
        # Cumulative is an optional-capacity constraint in OR-Tools.  If a
        # build ships an older API, fail loudly rather than silently dropping
        # the diagnostic constraint.
        model.add_cumulative(intervals, [1] * len(indices), capacity).only_enforce_if(
            literal
        )
        assumptions[engine] = literal

    model.add_assumptions(list(assumptions.values()))
    print(
        f"capacity_core ops={len(ops)} horizon={horizon} "
        f"engines={sorted(assumptions)}",
        flush=True,
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(os.environ.get("TIME_LIMIT", "30"))
    solver.parameters.num_workers = int(os.environ.get("WORKERS", "8"))
    solver.parameters.random_seed = int(os.environ.get("RANDOM_SEED", "1"))
    status = solver.solve(model)
    print(
        f"status={solver.status_name(status)} wall={solver.wall_time:.3f}s "
        f"conflicts={solver.num_conflicts} branches={solver.num_branches}",
        flush=True,
    )
    if status != cp_model.INFEASIBLE:
        print("result=NO_UNSAT_CORE", flush=True)
        return
    by_index = {literal.index: engine for engine, literal in assumptions.items()}
    core = [
        by_index[index]
        for index in solver.sufficient_assumptions_for_infeasibility()
        if index in by_index
    ]
    print(f"raw_core size={len(core)} engines={core}", flush=True)


if __name__ == "__main__":
    main()
