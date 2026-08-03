"""Find a small set of level-4 selection cells that must be capacity-relaxed.

This is a *diagnostic*, never a submission scheduler.  It keeps every raw DAG
edge at the requested horizon and models only the two coupled physical
resources (VALU and Flow) exactly.  A Boolean escape variable is shared by
all capacity uses in one semantic ``(engine, tag, group, round)`` cell.  When
enabled, the cell still executes and retains all data dependencies, but its
operations are placed in unique artificial capacity slots.  Minimizing these
escapes identifies a compact multi-group pressure set instead of a misleading
single-operation explanation.

The default target is the two phase pair exposed by the strict experiments:
``tree_select`` and ``level4_hybrid_bottom`` at rounds 4 and 15.
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


def csv_set(name: str, default: str) -> frozenset[str]:
    return frozenset(
        value for value in os.environ.get(name, default).split(",") if value
    )


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
    payload = json.loads(Path(path).read_text())
    cycles = payload.get("cycles")
    if not isinstance(cycles, list) or len(cycles) != size:
        raise ValueError(f"{path}: expected a {size}-entry cycles list")
    return [int(cycle) for cycle in cycles]


def validate_relaxed(
    ops: list[kernel._Op],
    cycles: list[int],
    horizon: int,
    escaped: frozenset[int],
    capacity_engines: frozenset[str],
) -> None:
    if min(cycles, default=0) < 0 or max(cycles, default=-1) >= horizon:
        raise AssertionError("solution lies outside requested horizon")
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            if cycles[child] < cycles[parent] + lag:
                raise AssertionError(f"precedence {parent}->{child} violated")
    usage = Counter(
        (cycle, op.engine)
        for index, (cycle, op) in enumerate(zip(cycles, ops))
        if index not in escaped and op.engine in capacity_engines
    )
    for (cycle, engine), demand in usage.items():
        if demand > SLOT_LIMITS[engine]:
            raise AssertionError(
                f"capacity {engine}@{cycle}: {demand}>{SLOT_LIMITS[engine]}"
            )


def main() -> None:
    ops = build_ops()
    horizon = int(os.environ.get("TARGET", "959"))
    capacity_engines = csv_set("CAPACITY_ENGINES", "valu,flow")
    if capacity_engines != frozenset(("valu", "flow")):
        raise ValueError("this focused diagnostic supports exactly valu,flow")
    tags = csv_set("TAGS", "tree_select,level4_hybrid_bottom")
    rounds = integer_set("ROUNDS", "4,15")
    if any(rnd < 0 for rnd in rounds):
        raise ValueError("ROUNDS must be non-negative")
    _, _, earliest, tail = topological_data(ops)
    empty = [
        index
        for index in range(len(ops))
        if earliest[index] > horizon - 1 - tail[index]
    ]
    if empty:
        raise ValueError(f"raw DAG alone is impossible at {horizon}: op {empty[0]}")

    cells: dict[tuple[str, str, int, int], list[int]] = defaultdict(list)
    for index, op in enumerate(ops):
        if (
            op.engine in capacity_engines
            and op.tag in tags
            and op.group >= 0
            and op.round in rounds
        ):
            cells[(op.engine, op.tag, op.group, op.round)].append(index)
    if not cells:
        raise ValueError("selection produced no relaxable cells")
    ordered_cells = sorted(cells)
    cell_of_index = {
        index: key for key, indices in cells.items() for index in indices
    }
    counts = Counter(op.engine for op in ops)
    print(
        f"select_phase_relaxation strict_dag=1 ops={len(ops)} "
        f"edges={sum(len(op.parents) for op in ops)} horizon={horizon} "
        f"counts={dict(counts)} tags={sorted(tags)} rounds={sorted(rounds)} "
        f"cells={len(ordered_cells)} candidate_jobs={len(cell_of_index)}",
        flush=True,
    )

    model = cp_model.CpModel()
    starts: list[cp_model.IntVar] = []
    bounds: list[tuple[int, int]] = []
    for index in range(len(ops)):
        lower = earliest[index]
        upper = horizon - 1 - tail[index]
        bounds.append((lower, upper))
        starts.append(model.new_int_var(lower, upper, f"s{index}"))
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            model.add(starts[child] >= starts[parent] + lag)

    drop = {
        key: model.new_bool_var(
            f"drop_{key[0]}_{key[1]}_{key[2]}_{key[3]}"
        )
        for key in ordered_cells
    }
    for engine in sorted(capacity_engines):
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

    objective_mode = os.environ.get("OBJECTIVE", "cells")
    if objective_mode == "cells":
        objective = sum(drop.values())
    elif objective_mode == "jobs":
        objective = sum(len(cells[key]) * variable for key, variable in drop.items())
    else:
        raise ValueError("OBJECTIVE must be cells or jobs")
    model.minimize(objective)

    seed = load_seed(os.environ.get("SEED"), len(ops))
    fix_seed = bool(int(os.environ.get("FIX_SEED", "0")))
    if fix_seed and seed is None:
        raise ValueError("FIX_SEED=1 requires SEED")
    if seed is not None:
        for index, cycle in enumerate(seed):
            lower, upper = bounds[index]
            clamped = min(upper, max(lower, cycle))
            model.add_hint(starts[index], clamped)
            if fix_seed:
                if clamped != cycle:
                    raise ValueError(f"SEED cycle outside exact bound for op {index}")
                model.add(starts[index] == cycle)
    # The all-escaped assignment is guaranteed to have no select/MADD capacity
    # conflict; it supplies a feasible upper bound when paired with a valid
    # strict-DAG seed, while CP-SAT is free to turn cells back on.
    for key, variable in drop.items():
        model.add_hint(variable, 1)

    error = model.validate()
    if error:
        raise ValueError(error)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(os.environ.get("TIME_LIMIT", "60"))
    solver.parameters.num_workers = int(os.environ.get("WORKERS", "1"))
    solver.parameters.random_seed = int(os.environ.get("RANDOM_SEED", "1"))
    solver.parameters.randomize_search = bool(int(os.environ.get("RANDOMIZE", "0")))
    solver.parameters.repair_hint = bool(int(os.environ.get("REPAIR_HINT", "0")))
    status = solver.solve(model)
    print(
        f"status={solver.status_name(status)} wall={solver.wall_time:.3f}s "
        f"conflicts={solver.num_conflicts} branches={solver.num_branches} "
        f"objective={solver.objective_value:g} bound={solver.best_objective_bound:g} "
        f"fixed_seed={int(fix_seed)}",
        flush=True,
    )
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return
    selected = [key for key in ordered_cells if solver.value(drop[key])]
    escaped = frozenset(index for key in selected for index in cells[key])
    result = [solver.value(start) for start in starts]
    validate_relaxed(ops, result, horizon, escaped, capacity_engines)
    print(
        f"escaped_cells={len(selected)} escaped_jobs={len(escaped)} "
        f"by_engine={dict(Counter(key[0] for key in selected))}",
        flush=True,
    )
    for key in selected:
        print(f"  {key} jobs={len(cells[key])}", flush=True)
    output = os.environ.get("OUT")
    if output:
        Path(output).write_text(
            json.dumps(
                {
                    "cycles": result,
                    "horizon": horizon,
                    "relaxed_cells": [list(key) for key in selected],
                    "relaxed_indices": sorted(escaped),
                    "strict_precedence": True,
                }
            )
        )
        print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
