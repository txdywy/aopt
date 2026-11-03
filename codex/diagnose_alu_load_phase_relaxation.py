"""Minimize semantic ALU/Load capacity escapes at a fixed horizon.

The full dependency graph remains intact.  Only capacity occupancy of a
selected semantic cell may be escaped, so the diagnostic isolates which
source-level operations must be retimed or lowered differently before an
ALU/Load 959-cycle schedule can exist.  It is not a submission scheduler and
an escaped cell is never a legal implementation by itself.

By default the candidates are all ``tree_gather`` Load cells and the early
``mirror_bit`` ALU cells (rounds 0..3).  The latter map directly to
``VECTOR_PARITY_SET``: one vector parity replaces the cell's eight scalar ALU
operations with one VALU operation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path

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


def load_seed(path: str | None, size: int) -> list[int] | None:
    if not path:
        return None
    cycles = json.loads(Path(path).read_text()).get("cycles")
    if not isinstance(cycles, list) or len(cycles) != size:
        raise ValueError(f"{path}: expected a {size}-entry cycles list")
    return [int(cycle) for cycle in cycles]


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
    if not cells:
        raise ValueError("candidate selection produced no cells")
    ordered_cells = sorted(cells)
    cell_of_index = {
        index: key for key, indices in cells.items() for index in indices
    }
    seed = load_seed(os.environ.get("SEED"), len(ops))
    hint_radius = int(os.environ.get("HINT_RADIUS", "-1"))
    if hint_radius < -1:
        raise ValueError("HINT_RADIUS must be -1 or non-negative")

    model = cp_model.CpModel()
    starts: list[cp_model.IntVar] = []
    bounds: list[tuple[int, int]] = []
    for index in range(len(ops)):
        lower = earliest[index]
        upper = horizon - 1 - tail[index]
        if hint_radius >= 0 and seed is not None:
            cycle = seed[index]
            lower = max(lower, cycle - hint_radius)
            upper = min(upper, cycle + hint_radius)
            if lower > upper:
                raise ValueError(
                    f"SEED is outside the radius/domain at op {index}: "
                    f"cycle={cycle} domain={earliest[index]}:{horizon - 1 - tail[index]}"
                )
        bounds.append((lower, upper))
        starts.append(model.new_int_var(lower, upper, f"s{index}"))
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            model.add(starts[child] >= starts[parent] + lag)

    drop = {
        key: model.new_bool_var(
            f"drop_{key[0]}_{key[1]}_g{key[2]}_r{key[3]}"
        )
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
                    drop[key].Not()
                )
                model.add(microslot == capacity * horizon + rank).only_enforce_if(
                    drop[key]
                )
            microslots.append(microslot)
        model.add_all_different(microslots)

    tree_weight = int(os.environ.get("TREE_GATHER_WEIGHT", "1"))
    mirror_weight = int(os.environ.get("MIRROR_BIT_WEIGHT", "1"))
    if tree_weight <= 0 or mirror_weight <= 0:
        raise ValueError("cell weights must be positive")
    model.minimize(
        sum(
            (tree_weight if key[1] == "tree_gather" else mirror_weight)
            * variable
            for key, variable in drop.items()
        )
    )

    if seed is not None:
        for index, cycle in enumerate(seed):
            lower, upper = bounds[index]
            model.add_hint(starts[index], min(upper, max(lower, cycle)))
    for variable in drop.values():
        model.add_hint(variable, 1)

    error = model.validate()
    if error:
        raise ValueError(error)
    print(
        f"alu_load_phase_relaxation strict_dag=1 ops={len(ops)} "
        f"edges={sum(len(op.parents) for op in ops)} horizon={horizon} "
        f"cells={len(ordered_cells)} candidate_jobs={len(cell_of_index)} "
        f"mirror_rounds={sorted(mirror_rounds)} "
        f"weights=tree:{tree_weight},mirror:{mirror_weight} "
        f"hint_radius={hint_radius}",
        flush=True,
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(os.environ.get("TIME_LIMIT", "60"))
    solver.parameters.num_workers = int(os.environ.get("WORKERS", "1"))
    solver.parameters.random_seed = int(os.environ.get("RANDOM_SEED", "1"))
    solver.parameters.randomize_search = bool(int(os.environ.get("RANDOMIZE", "0")))
    solver.parameters.repair_hint = bool(int(os.environ.get("REPAIR_HINT", "1")))
    status = solver.solve(model)
    print(
        f"status={solver.status_name(status)} wall={solver.wall_time:.3f}s "
        f"conflicts={solver.num_conflicts} branches={solver.num_branches} "
        f"objective={solver.objective_value:g} bound={solver.best_objective_bound:g}",
        flush=True,
    )
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return
    selected = [key for key in ordered_cells if solver.value(drop[key])]
    by_tag_round = Counter((key[1], key[3]) for key in selected)
    print(
        f"escaped_cells={len(selected)} escaped_jobs="
        f"{sum(len(cells[key]) for key in selected)} "
        f"by_tag_round={dict(sorted(by_tag_round.items()))}",
        flush=True,
    )
    for key in selected:
        print(f"  {key} jobs={len(cells[key])}", flush=True)
    output = os.environ.get("OUT")
    if output:
        Path(output).write_text(
            json.dumps(
                {
                    "cycles": [solver.value(start) for start in starts],
                    "horizon": horizon,
                    "relaxed_cells": [list(key) for key in selected],
                    "strict_precedence": True,
                }
            )
        )
        print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
