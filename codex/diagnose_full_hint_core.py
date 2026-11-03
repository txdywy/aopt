"""Extract a strict full-DAG unsatisfiable core for hint-radius anchors.

``solve_full_dag`` intentionally treats hints as ordinary hints.  This tool
instead turns every selected hint window into an *assumption* and asks CP-SAT
for a sufficient infeasibility core.  All original precedence edges and all
engine capacities stay hard.  Thus a reported core says which independent
engine phases cannot coexist in the requested local neighborhood; it is never
a projection artefact.

Use ``CORE_GROUPING=engine`` for a coarse answer and
``CORE_GROUPING=engine_group`` to isolate particular software-pipeline groups.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import os

from ortools.sat.python import cp_model

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from codex.solve_full_dag import load_hints, topological_data
from problem import SLOT_LIMITS


def group_name(op: kernel._Op, mode: str) -> str:
    if mode == "engine":
        return op.engine
    if mode == "engine_group":
        return f"{op.engine}:g{op.group}"
    if mode == "engine_round":
        return f"{op.engine}:g{op.group}:r{op.round}"
    raise ValueError("CORE_GROUPING must be engine, engine_group, or engine_round")


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
    radius = int(os.environ.get("HINT_RADIUS", "640"))
    if radius < 0:
        raise ValueError("HINT_RADIUS must be non-negative")
    grouping = os.environ.get("CORE_GROUPING", "engine")
    hints, _ = load_hints(os.environ.get("HINTS", "").split(","), ops)
    if not hints:
        raise ValueError("HINTS is required")

    model = cp_model.CpModel()
    starts = [
        model.new_int_var(earliest[index], horizon - 1 - tail[index], f"s{index}")
        for index in range(len(ops))
    ]
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            model.add(starts[child] >= starts[parent] + lag)

    indices_by_group: dict[str, list[int]] = defaultdict(list)
    for index in hints:
        indices_by_group[group_name(ops[index], grouping)].append(index)
    assumptions: dict[str, cp_model.IntVar] = {
        name: model.new_bool_var(f"assume_{name.replace(':', '_')}")
        for name in sorted(indices_by_group)
    }
    constrained = Counter()
    for name, indices in indices_by_group.items():
        literal = assumptions[name]
        for index in indices:
            cycle = hints[index]
            lower = max(earliest[index], cycle - radius)
            upper = min(horizon - 1 - tail[index], cycle + radius)
            if lower > earliest[index]:
                model.add(starts[index] >= lower).only_enforce_if(literal)
                constrained[name] += 1
            if upper < horizon - 1 - tail[index]:
                model.add(starts[index] <= upper).only_enforce_if(literal)
                constrained[name] += 1
    for engine, capacity in SLOT_LIMITS.items():
        indices = [index for index, op in enumerate(ops) if op.engine == engine]
        intervals = [
            model.new_fixed_size_interval_var(starts[index], 1, f"i{index}")
            for index in indices
        ]
        model.add_cumulative(intervals, [1] * len(indices), capacity)

    by_index = {literal.index: name for name, literal in assumptions.items()}

    def solve_with(names: list[str]) -> tuple[int, cp_model.CpSolver]:
        model.clear_assumptions()
        model.add_assumptions([assumptions[name] for name in names])
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(
            os.environ.get("TIME_LIMIT", "30")
        )
        solver.parameters.num_workers = int(os.environ.get("WORKERS", "1"))
        solver.parameters.random_seed = int(os.environ.get("RANDOM_SEED", "1"))
        solver.parameters.log_search_progress = bool(int(os.environ.get("LOG", "0")))
        return solver.solve(model), solver

    names = list(assumptions)
    print(
        f"strict_hint_core ops={len(ops)} edges={sum(len(op.parents) for op in ops)} "
        f"horizon={horizon} radius={radius} grouping={grouping} "
        f"assumptions={len(names)} constrained={dict(constrained)}",
        flush=True,
    )
    status, solver = solve_with(names)
    print(
        f"initial_status={solver.status_name(status)} wall={solver.wall_time:.3f}s",
        flush=True,
    )
    if status != cp_model.INFEASIBLE:
        print("result=NO_UNSAT_CORE", flush=True)
        return
    core = [by_index[index] for index in solver.sufficient_assumptions_for_infeasibility()]
    print(f"raw_core size={len(core)} labels={core}", flush=True)

    # Deletion-based shrinking turns CP-SAT's sufficient core into a compact
    # deterministic local minimum whenever each quick re-solve remains exact.
    if bool(int(os.environ.get("SHRINK_CORE", "1"))):
        position = 0
        while position < len(core):
            candidate = core[:position] + core[position + 1 :]
            trial_status, trial_solver = solve_with(candidate)
            if trial_status == cp_model.INFEASIBLE:
                print(
                    f"shrink removed={core[position]} wall={trial_solver.wall_time:.3f}s",
                    flush=True,
                )
                core = candidate
            else:
                position += 1
        print(f"minimal_local_core size={len(core)} labels={core}", flush=True)

    core_set = set(core)
    index_label = {
        index: group_name(ops[index], grouping) if index in hints else None
        for index in range(len(ops))
    }
    edges: Counter[tuple[str, str, int]] = Counter()
    examples: dict[tuple[str, str, int], tuple[int, int]] = {}
    for child, op in enumerate(ops):
        child_label = index_label[child]
        for parent, lag in op.parents.items():
            parent_label = index_label[parent]
            if parent_label not in core_set or child_label not in core_set:
                continue
            if parent_label == child_label:
                continue
            key = (parent_label, child_label, lag)
            edges[key] += 1
            examples.setdefault(key, (parent, child))
    print("core_cross_edges=", flush=True)
    for (parent_label, child_label, lag), count in edges.most_common(24):
        parent, child = examples[parent_label, child_label, lag]
        parent_op = ops[parent]
        child_op = ops[child]
        print(
            f"  {parent_label}->{child_label} lag={lag} count={count} "
            f"example={parent}:{parent_op.tag}->"
            f"{child}:{child_op.tag}",
            flush=True,
        )


if __name__ == "__main__":
    main()
