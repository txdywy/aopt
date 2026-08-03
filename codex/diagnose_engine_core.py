"""Extract a semantic-cell UNSAT core for one exact engine calendar.

The ordinary Hall test only uses independent earliest/latest windows.  This
diagnostic keeps the complete DAG and the engine's physical micro-slots, then
relaxes selected operation cells under assumptions.  A returned core is a
small certificate that at least one of those cells must be retimed or lowered
to make the requested horizon possible.

Examples::

    ENGINE=valu GROUPING=tag_round TARGET=959 \
      python3 -m codex.diagnose_engine_core
"""

from __future__ import annotations

from collections import Counter, defaultdict
import os

from ortools.sat.python import cp_model

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from codex.solve_full_dag import topological_data
from problem import SLOT_LIMITS


def label(op: kernel._Op, grouping: str) -> tuple[object, ...]:
    if grouping == "tag":
        return (op.tag,)
    if grouping == "tag_round":
        return (op.tag, op.round)
    if grouping == "tag_group":
        return (op.tag, op.group)
    if grouping == "group_round":
        return (op.group, op.round)
    if grouping == "group":
        return (op.group,)
    raise ValueError(f"unknown GROUPING={grouping}")


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
    engine = os.environ.get("ENGINE", "valu")
    grouping = os.environ.get("GROUPING", "tag_round")
    horizon = int(os.environ.get("TARGET", "959"))
    ops = build_ops()
    _, _, earliest, tail = topological_data(ops)
    if any(earliest[i] > horizon - 1 - tail[i] for i in range(len(ops))):
        raise ValueError("raw DAG is already impossible at this horizon")

    cells: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for index, op in enumerate(ops):
        if op.engine == engine:
            cells[label(op, grouping)].append(index)
    ordered = sorted(cells, key=str)
    cell_of = {index: key for key, indices in cells.items() for index in indices}
    if not ordered:
        raise ValueError(f"no {engine} cells")

    model = cp_model.CpModel()
    starts = [
        model.new_int_var(earliest[i], horizon - 1 - tail[i], f"s{i}")
        for i in range(len(ops))
    ]
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            model.add(starts[child] >= starts[parent] + lag)

    assumptions = {
        key: model.new_bool_var(f"keep_{engine}_{rank}")
        for rank, key in enumerate(ordered)
    }
    capacity = SLOT_LIMITS[engine]
    indices = [i for i, op in enumerate(ops) if op.engine == engine]
    microslots = []
    # Unkept cells receive a disjoint diagnostic suffix, so only selected
    # cells consume the physical calendar.  This preserves all DAG edges.
    suffix_base = capacity * horizon
    for rank, index in enumerate(indices):
        lane = model.new_int_var(0, capacity - 1, f"lane_{engine}_{index}")
        micro = model.new_int_var(
            0, suffix_base + len(indices) - 1, f"micro_{engine}_{index}"
        )
        key = cell_of[index]
        model.add(micro == capacity * starts[index] + lane).only_enforce_if(
            assumptions[key]
        )
        model.add(micro == suffix_base + rank).only_enforce_if(
            assumptions[key].Not()
        )
        microslots.append(micro)
    model.add_all_different(microslots)
    by_literal = {literal.index: key for key, literal in assumptions.items()}

    print(
        f"engine_core engine={engine} grouping={grouping} ops={len(ops)} "
        f"engine_jobs={len(indices)} cells={len(ordered)} horizon={horizon}",
        flush=True,
    )

    def solve_with(keys: list[tuple[object, ...]]):
        model.clear_assumptions()
        model.add_assumptions([assumptions[key] for key in keys])
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(
            os.environ.get("TIME_LIMIT", "30")
        )
        solver.parameters.num_workers = int(os.environ.get("WORKERS", "8"))
        solver.parameters.random_seed = int(os.environ.get("RANDOM_SEED", "1"))
        return solver.solve(model), solver

    status, solver = solve_with(ordered)
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
        f"raw_core={len(core)} tags="
        f"{dict(Counter(key[0] if grouping.startswith('tag') else key for key in core))}",
        flush=True,
    )
    if bool(int(os.environ.get("SHRINK_CORE", "0"))):
        position = 0
        while position < len(core):
            trial = core[:position] + core[position + 1 :]
            trial_status, trial_solver = solve_with(trial)
            if trial_status == cp_model.INFEASIBLE:
                print(f"shrink_removed={core[position]}", flush=True)
                core = trial
            else:
                position += 1
    print(f"local_core={len(core)}", flush=True)
    for key in core:
        indices_for_key = cells[key]
        sample = indices_for_key[0]
        op = ops[sample]
        print(f"  {key} jobs={len(indices_for_key)} sample={sample}:{op.tag}:g{op.group}:r{op.round}", flush=True)


if __name__ == "__main__":
    main()
