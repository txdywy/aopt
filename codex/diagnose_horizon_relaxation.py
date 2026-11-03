"""Find which VALU/Flow jobs must be capacity-relaxed at a fixed horizon.

This is a diagnosis tool, not a scheduler and never emits a submission
candidate.  It retains every raw DAG edge, then gives selected resource jobs
an optional artificial escape hatch: an escaped job still has a legal start
time and all of its data dependencies, but does not occupy its physical
engine.  Minimizing escapes produces a compact explanation of the capacity
coupling that prevents a target horizon.

For the 959-cycle designs, VALU and Flow are the useful first pair.  The
result tells the compiler search which *semantic operations* need to move or
be retimed, rather than merely saying that an aggregate resource count is
tight.
"""

from __future__ import annotations

from collections import Counter
import os

from ortools.sat.python import cp_model

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from codex.solve_full_dag import topological_data
from problem import SLOT_LIMITS


def build_ops() -> list[kernel._Op]:
    configure_target()
    kernel.SCHEDULE_EXACT_CYCLES = None
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        if not hasattr(builder, "dag_ops"):
            raise
    return real_tail_ops(builder.dag_ops)


def csv_set(name: str, default: str) -> frozenset[str]:
    return frozenset(value for value in os.environ.get(name, default).split(",") if value)


def main() -> None:
    ops = build_ops()
    horizon = int(os.environ.get("TARGET", "959"))
    relaxed_engines = csv_set("RELAX_ENGINES", "valu,flow")
    unknown = relaxed_engines - set(SLOT_LIMITS)
    if unknown:
        raise ValueError(f"unknown RELAX_ENGINES: {sorted(unknown)}")
    _, _, earliest, tail = topological_data(ops)
    if any(earliest[i] > horizon - 1 - tail[i] for i in range(len(ops))):
        raise ValueError("target is already impossible from raw precedence")

    model = cp_model.CpModel()
    starts = [
        model.new_int_var(earliest[index], horizon - 1 - tail[index], f"s{index}")
        for index in range(len(ops))
    ]
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            model.add(starts[child] >= starts[parent] + lag)

    drops: dict[int, cp_model.IntVar] = {}
    for engine in relaxed_engines:
        capacity = SLOT_LIMITS[engine]
        indices = [index for index, op in enumerate(ops) if op.engine == engine]
        microslots: list[cp_model.IntVar] = []
        # A dropped job receives a unique value beyond the real calendar, so
        # one AllDifferent models all live capacity slots without optional
        # intervals or an accidental collision between two dropped jobs.
        for rank, index in enumerate(indices):
            drop = model.new_bool_var(f"drop_{engine}_{index}")
            drops[index] = drop
            lane = model.new_int_var(0, capacity - 1, f"lane_{engine}_{index}")
            microslot = model.new_int_var(
                0,
                capacity * horizon + len(indices) - 1,
                f"microslot_{engine}_{index}",
            )
            model.add(microslot == capacity * starts[index] + lane).only_enforce_if(
                drop.Not()
            )
            model.add(microslot == capacity * horizon + rank).only_enforce_if(drop)
            microslots.append(microslot)
        model.add_all_different(microslots)

    if not drops:
        raise ValueError("RELAX_ENGINES selects no jobs")
    engine_weight = {
        engine: int(os.environ.get(f"{engine.upper()}_DROP_WEIGHT", "1"))
        for engine in relaxed_engines
    }
    model.minimize(
        sum(engine_weight[ops[index].engine] * drop for index, drop in drops.items())
    )
    print(
        f"strict_relaxation ops={len(ops)} edges={sum(len(op.parents) for op in ops)} "
        f"horizon={horizon} relaxed={sorted(relaxed_engines)} "
        f"jobs={Counter(ops[index].engine for index in drops)} weights={engine_weight}",
        flush=True,
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(os.environ.get("TIME_LIMIT", "60"))
    solver.parameters.num_workers = int(os.environ.get("WORKERS", "1"))
    solver.parameters.random_seed = int(os.environ.get("RANDOM_SEED", "1"))
    solver.parameters.randomize_search = bool(int(os.environ.get("RANDOMIZE", "0")))
    solver.parameters.log_search_progress = bool(int(os.environ.get("LOG", "0")))
    status = solver.solve(model)
    print(
        f"status={solver.status_name(status)} wall={solver.wall_time:.3f}s "
        f"conflicts={solver.num_conflicts} branches={solver.num_branches} "
        f"objective={solver.objective_value:g} bound={solver.best_objective_bound:g}",
        flush=True,
    )
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return
    dropped = [index for index, drop in drops.items() if solver.value(drop)]
    print(
        f"dropped={len(dropped)} by_engine={dict(Counter(ops[index].engine for index in dropped))}",
        flush=True,
    )
    for index in dropped:
        op = ops[index]
        print(
            f"  i={index} engine={op.engine} cycle={solver.value(starts[index])} "
            f"g={op.group} r={op.round} opcode={op.slot[0]} tag={op.tag}",
            flush=True,
        )


if __name__ == "__main__":
    main()
