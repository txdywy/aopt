"""Explain ALU/Load pressure on a *fixed*, precedence-valid calendar.

This is deliberately narrower than :mod:`solve_full_dag`.  Given a relaxed
full-DAG schedule, it finds the smallest collection of semantic cells whose
ALU or Load occupancy must be ignored for that particular calendar to meet
the physical slot limits.  It is useful for turning a broad relaxed witness
into concrete compiler targets such as ``tree_gather g:r`` or
``mirror_bit g:r``.

It never claims a global unsatisfiable core: starts are fixed to ``SEED``, so
the answer is an exact minimum only for that calendar.  All original
precedence edges are checked before the cover model is built.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path

from ortools.sat.python import cp_model

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
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


def load_cycles(path: str, expected_size: int) -> list[int]:
    payload = json.loads(Path(path).read_text())
    cycles = payload.get("cycles")
    if not isinstance(cycles, list) or len(cycles) != expected_size:
        raise ValueError(f"{path}: expected a {expected_size}-entry cycles list")
    return [int(value) for value in cycles]


def main() -> None:
    seed_path = os.environ.get("SEED")
    if not seed_path:
        raise ValueError("SEED is required")
    horizon = int(os.environ.get("TARGET", "959"))
    tags = csv_set("CANDIDATE_TAGS", "tree_gather,mirror_bit")
    mirror_rounds = integer_set("MIRROR_BIT_ROUNDS", "0,1,2,3")
    if "tree_gather" not in tags or "mirror_bit" not in tags:
        raise ValueError(
            "the fixed-calendar diagnostic currently requires both "
            "tree_gather and mirror_bit candidate tags"
        )

    ops = build_ops()
    cycles = load_cycles(seed_path, len(ops))
    if min(cycles, default=0) < 0 or max(cycles, default=-1) >= horizon:
        raise ValueError("SEED has a cycle outside the requested horizon")
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            if cycles[child] < cycles[parent] + lag:
                raise ValueError(
                    f"SEED violates precedence {parent}->{child} lag={lag}"
                )

    # A tree-gather cell represents its entire load burst.  For mirror bits,
    # concentrate on the pre-gather parity phase; its rounds are independently
    # configurable because tree gathers themselves begin later in the round.
    cells: dict[tuple[str, str, int, int], list[int]] = defaultdict(list)
    for index, op in enumerate(ops):
        if op.group < 0:
            continue
        if op.tag == "tree_gather":
            cells[(op.engine, op.tag, op.group, op.round)].append(index)
        elif op.tag == "mirror_bit" and op.round in mirror_rounds:
            cells[(op.engine, op.tag, op.group, op.round)].append(index)
    if not cells:
        raise ValueError("candidate selection produced no cells")
    ordered_cells = sorted(cells)
    cell_of_index = {
        index: key for key, indices in cells.items() for index in indices
    }

    # Every noncandidate operation is a hard reservation.  A candidate cell is
    # either wholly kept or wholly escaped, matching the granularity of a
    # practical source-level lowering rather than cherry-picking one lane.
    hard_usage: Counter[tuple[str, int]] = Counter()
    cell_usage: dict[tuple[str, str, int, int], Counter[tuple[str, int]]] = {
        key: Counter() for key in ordered_cells
    }
    for index, op in enumerate(ops):
        if op.engine not in ("alu", "load"):
            continue
        key = cell_of_index.get(index)
        if key is None:
            hard_usage[op.engine, cycles[index]] += 1
        else:
            cell_usage[key][op.engine, cycles[index]] += 1

    model = cp_model.CpModel()
    drop = {
        key: model.new_bool_var(
            f"drop_{key[0]}_{key[1]}_g{key[2]}_r{key[3]}"
        )
        for key in ordered_cells
    }
    incident: dict[tuple[str, int], list[tuple[tuple[str, str, int, int], int]]] = (
        defaultdict(list)
    )
    for key, usage in cell_usage.items():
        for resource_time, multiplicity in usage.items():
            incident[resource_time].append((key, multiplicity))
    for resource_time, members in incident.items():
        engine, _ = resource_time
        # hard + sum(multiplicity * (1 - drop)) <= capacity
        # Rearrangement keeps CP-SAT's expression purely linear in Booleans.
        required_drop = (
            hard_usage[resource_time]
            + sum(multiplicity for _, multiplicity in members)
            - SLOT_LIMITS[engine]
        )
        if required_drop > 0:
            model.add(
                sum(multiplicity * drop[key] for key, multiplicity in members)
                >= required_drop
            )

    model.minimize(sum(drop.values()))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(os.environ.get("TIME_LIMIT", "30"))
    solver.parameters.num_workers = int(os.environ.get("WORKERS", "1"))
    solver.parameters.random_seed = int(os.environ.get("RANDOM_SEED", "1"))
    status = solver.solve(model)
    print(
        f"fixed_calendar_alu_load strict_precedence=1 ops={len(ops)} "
        f"horizon={horizon} seed={seed_path} cells={len(ordered_cells)} "
        f"candidate_jobs={len(cell_of_index)} mirror_rounds={sorted(mirror_rounds)}",
        flush=True,
    )
    print(
        f"status={solver.status_name(status)} wall={solver.wall_time:.3f}s "
        f"objective={solver.objective_value:g} "
        f"bound={solver.best_objective_bound:g}",
        flush=True,
    )
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return
    selected = [key for key in ordered_cells if solver.value(drop[key])]
    print(
        f"escaped_cells={len(selected)} escaped_jobs="
        f"{sum(len(cells[key]) for key in selected)} by_tag_round="
        f"{dict(Counter((key[1], key[3]) for key in selected))}",
        flush=True,
    )
    for key in selected:
        print(f"  {key} jobs={len(cells[key])}", flush=True)


if __name__ == "__main__":
    main()
