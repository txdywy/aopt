"""Extract a sufficient ALU/Load capacity core by semantic cell.

Every raw DAG edge remains hard.  A candidate cell has a Boolean ``keep``
literal: when kept, its operations occupy physical ALU/Load capacity; when
not kept, they receive unique diagnostic-only slots.  Asking CP-SAT to assume
all keeps yields the original strict A/L problem.  Its unsatisfiable-assumption
core therefore names a set of semantic cells whose *simultaneous physical
occupancy* is sufficient for the target horizon to fail.

This is a compiler-search diagnostic, not a legal schedule.  The core does
not say that a listed cell may be deleted, only that a real lowering or
retiming must change at least one capacity interaction in the set.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import os

from ortools.sat.python import cp_model

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from codex.solve_full_dag import topological_data
from problem import SLOT_LIMITS


def integer_set(name: str, default: str) -> frozenset[int]:
    try:
        return frozenset(
            int(value)
            for value in os.environ.get(name, default).split(",")
            if value
        )
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated integer set") from error


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


def main() -> None:
    horizon = int(os.environ.get("TARGET", "959"))
    mirror_rounds = integer_set("MIRROR_BIT_ROUNDS", "0,1,2,3")
    ops = build_ops()
    _, _, earliest, tail = topological_data(ops)
    if any(earliest[i] > horizon - 1 - tail[i] for i in range(len(ops))):
        raise ValueError("raw DAG is already impossible at this horizon")

    cells: dict[tuple[str, str, int, int], list[int]] = defaultdict(list)
    for index, op in enumerate(ops):
        if op.group < 0:
            continue
        if op.engine == "load" and op.tag == "tree_gather":
            cells[(op.engine, op.tag, op.group, op.round)].append(index)
        elif (
            op.engine == "alu"
            and op.tag == "mirror_bit"
            and op.round in mirror_rounds
        ):
            cells[(op.engine, op.tag, op.group, op.round)].append(index)
    ordered_cells = sorted(cells)
    if not ordered_cells:
        raise ValueError("candidate selection produced no cells")
    cell_of_index = {
        index: key for key, indices in cells.items() for index in indices
    }

    model = cp_model.CpModel()
    starts = [
        model.new_int_var(earliest[index], horizon - 1 - tail[index], f"s{index}")
        for index in range(len(ops))
    ]
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            model.add(starts[child] >= starts[parent] + lag)

    keep = {
        key: model.new_bool_var(f"keep_{key[0]}_{key[1]}_g{key[2]}_r{key[3]}")
        for key in ordered_cells
    }
    for engine in ("alu", "load"):
        capacity = SLOT_LIMITS[engine]
        indices = [index for index, op in enumerate(ops) if op.engine == engine]
        microslots: list[cp_model.IntVar] = []
        for rank, index in enumerate(indices):
            lane = model.new_int_var(0, capacity - 1, f"lane_{engine}_{index}")
            microslot = model.new_int_var(
                0,
                capacity * horizon + len(indices) - 1,
                f"micro_{engine}_{index}",
            )
            key = cell_of_index.get(index)
            if key is None:
                model.add(microslot == capacity * starts[index] + lane)
            else:
                model.add(microslot == capacity * starts[index] + lane).only_enforce_if(
                    keep[key]
                )
                model.add(microslot == capacity * horizon + rank).only_enforce_if(
                    keep[key].Not()
                )
            microslots.append(microslot)
        model.add_all_different(microslots)

    error = model.validate()
    if error:
        raise ValueError(error)
    by_literal = {literal.index: key for key, literal in keep.items()}
    print(
        f"alu_load_assumption_core strict_dag=1 ops={len(ops)} "
        f"edges={sum(len(op.parents) for op in ops)} horizon={horizon} "
        f"cells={len(ordered_cells)} candidate_jobs={len(cell_of_index)} "
        f"mirror_rounds={sorted(mirror_rounds)}",
        flush=True,
    )

    def solve_with(keys: list[tuple[str, str, int, int]]) -> tuple[int, cp_model.CpSolver]:
        model.clear_assumptions()
        model.add_assumptions([keep[key] for key in keys])
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(
            os.environ.get("TIME_LIMIT", "20")
        )
        solver.parameters.num_workers = int(os.environ.get("WORKERS", "1"))
        solver.parameters.random_seed = int(os.environ.get("RANDOM_SEED", "1"))
        solver.parameters.log_search_progress = bool(int(os.environ.get("LOG", "0")))
        return solver.solve(model), solver

    status, solver = solve_with(ordered_cells)
    print(
        f"initial_status={solver.status_name(status)} wall={solver.wall_time:.3f}s "
        f"conflicts={solver.num_conflicts} branches={solver.num_branches}",
        flush=True,
    )
    if status != cp_model.INFEASIBLE:
        print("result=NO_EXACT_UNSAT_CORE", flush=True)
        return
    core = [
        by_literal[index]
        for index in solver.sufficient_assumptions_for_infeasibility()
        if index in by_literal
    ]
    print(
        f"raw_core_cells={len(core)} by_tag_round="
        f"{dict(sorted(Counter((key[1], key[3]) for key in core).items()))}",
        flush=True,
    )

    # The native core is sufficient but need not be deletion-minimal.  Each
    # removal is an exact re-solve; UNKNOWN is deliberately treated as
    # undecided, never as permission to remove a cell.
    if bool(int(os.environ.get("SHRINK_CORE", "1"))):
        position = 0
        while position < len(core):
            trial = core[:position] + core[position + 1 :]
            trial_status, trial_solver = solve_with(trial)
            if trial_status == cp_model.INFEASIBLE:
                print(
                    f"shrink_removed={core[position]} "
                    f"wall={trial_solver.wall_time:.3f}s",
                    flush=True,
                )
                core = trial
            else:
                print(
                    f"shrink_kept={core[position]} "
                    f"status={trial_solver.status_name(trial_status)} "
                    f"wall={trial_solver.wall_time:.3f}s",
                    flush=True,
                )
                position += 1
    print(
        f"local_core_cells={len(core)} by_tag_round="
        f"{dict(sorted(Counter((key[1], key[3]) for key in core).items()))}",
        flush=True,
    )
    for key in core:
        print(f"  {key} jobs={len(cells[key])}", flush=True)


if __name__ == "__main__":
    main()
