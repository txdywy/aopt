"""Strict H-horizon CP-SAT with a selected set of partial resource-order arcs.

This is an experiment driver for a cycle-broken global resource-order seed.
It retains all raw effective-DAG edges, adds only explicitly validated resource
arcs from ``PARTIAL_ORDER_HINTS``, and enforces every engine capacity.  Unlike
the projected solvers, there is no graph contraction anywhere in this model.

The default partial-arc policy is intentionally conservative: a 512-arc
evenly spaced batch.  Use ``PARTIAL_ARC_MAX=0 PARTIAL_ARC_STRIDE=1`` only for
an intentional full-hardening run after auditing the augmented span.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import heapq
import json
import os
from pathlib import Path

from ortools.sat.python import cp_model

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from codex.partial_order_hints import audit_augmented_dag, load_partial_order_hints
from codex.solve_full_dag import load_hints, validate_solution
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


def select_arcs(ops: list[kernel._Op]) -> tuple[tuple[tuple[int, int], ...], int]:
    paths = [
        value for value in os.environ.get("PARTIAL_ORDER_HINTS", "").split(",") if value
    ]
    if not paths:
        raise ValueError("PARTIAL_ORDER_HINTS is required")
    partial = load_partial_order_hints(paths, ops)
    # Validate all supplied arcs before selecting a small subset.  This is
    # fail-closed against malformed or cyclic seed files.
    audit_augmented_dag(ops, partial)
    engines = {
        value for value in os.environ.get("PARTIAL_ARC_ENGINES", "").split(",") if value
    }
    unknown = engines - set(SLOT_LIMITS)
    if unknown:
        raise ValueError(f"unknown PARTIAL_ARC_ENGINES: {sorted(unknown)}")
    maximum = int(os.environ.get("PARTIAL_ARC_MAX", "512"))
    requested_stride = int(os.environ.get("PARTIAL_ARC_STRIDE", "0"))
    if requested_stride < 0:
        raise ValueError("PARTIAL_ARC_STRIDE must be non-negative")
    candidates = [
        edge
        for edge in partial.effective_edges
        if not engines or ops[edge[0]].engine in engines
    ]
    stride = requested_stride or max(
        1, (len(candidates) + max(1, maximum) - 1) // max(1, maximum)
    )
    selected = tuple(
        edge for rank, edge in enumerate(candidates) if rank % stride == 0
    )
    if maximum > 0:
        selected = selected[:maximum]
    audit = audit_augmented_dag(ops, replace(partial, effective_edges=selected))
    print(
        f"partial_seed paths={len(partial.paths)} raw={len(partial.resource_edges)} "
        f"effective={len(partial.effective_edges)} selected={len(selected)} "
        f"stride={stride} cap={maximum} augmented_span={audit.longest_path_span}",
        flush=True,
    )
    return selected, audit.longest_path_span


def augmented_data(
    ops: list[kernel._Op], arcs: tuple[tuple[int, int], ...]
) -> tuple[list[dict[int, int]], list[int], list[int]]:
    parents = [dict(op.parents) for op in ops]
    for parent, child in arcs:
        parents[child][parent] = max(parents[child].get(parent, -1), 1)
    children: list[list[tuple[int, int]]] = [[] for _ in ops]
    indegree = [0] * len(ops)
    for child, row in enumerate(parents):
        indegree[child] = len(row)
        for parent, lag in row.items():
            children[parent].append((child, lag))
    ready = [index for index, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    topological: list[int] = []
    while ready:
        parent = heapq.heappop(ready)
        topological.append(parent)
        for child, _ in children[parent]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(topological) != len(ops):
        raise AssertionError("validated partial arcs became cyclic")
    earliest = [0] * len(ops)
    for child in topological:
        for parent, lag in parents[child].items():
            earliest[child] = max(earliest[child], earliest[parent] + lag)
    tail = [0] * len(ops)
    for parent in reversed(topological):
        tail[parent] = max(
            (lag + tail[child] for child, lag in children[parent]), default=0
        )
    return parents, earliest, tail


def main() -> None:
    ops = build_ops()
    horizon = int(os.environ.get("TARGET", "959"))
    arcs, audited_span = select_arcs(ops)
    parents, earliest, tail = augmented_data(ops, arcs)
    counts = Counter(op.engine for op in ops)
    floors = {
        engine: (counts[engine] + capacity - 1) // capacity
        for engine, capacity in SLOT_LIMITS.items()
        if counts[engine]
    }
    dag_lb = max(earliest, default=-1) + 1
    print(
        f"strict_full_partial ops={len(ops)} raw_edges={sum(len(op.parents) for op in ops)} "
        f"partial_edges={len(arcs)} target={horizon} dag_lb={dag_lb} "
        f"audited_span={audited_span} floors={floors}",
        flush=True,
    )
    if dag_lb > horizon:
        print("status=INFEASIBLE reason=augmented_precedence_lower_bound", flush=True)
        return

    hints, _ = load_hints(os.environ.get("HINTS", "").split(","), ops)
    radius = int(os.environ.get("HINT_RADIUS", "-1"))
    model = cp_model.CpModel()
    starts: list[cp_model.IntVar] = []
    bounds: list[tuple[int, int]] = []
    for index in range(len(ops)):
        lower = earliest[index]
        upper = horizon - 1 - tail[index]
        if radius >= 0 and index in hints:
            lower = max(lower, hints[index] - radius)
            upper = min(upper, hints[index] + radius)
        if lower > upper:
            print(
                f"status=INFEASIBLE reason=empty_domain i={index} "
                f"bounds={lower}:{upper}",
                flush=True,
            )
            return
        bounds.append((lower, upper))
        starts.append(model.new_int_var(lower, upper, f"s{index}"))
    for child, row in enumerate(parents):
        for parent, lag in row.items():
            model.add(starts[child] >= starts[parent] + lag)
    microslot_engines = {
        value
        for value in os.environ.get("MICROSLOT_ENGINES", "").split(",")
        if value
    }
    all_different_engines = {
        value
        for value in os.environ.get("ALL_DIFFERENT_ENGINES", "").split(",")
        if value
    }
    unknown_modes = (microslot_engines | all_different_engines) - set(SLOT_LIMITS)
    if unknown_modes:
        raise ValueError(f"unknown capacity-mode engines: {sorted(unknown_modes)}")
    if microslot_engines & all_different_engines:
        raise ValueError("choose one capacity form per engine")
    capacity_forms: dict[str, str] = {}
    for engine, capacity in SLOT_LIMITS.items():
        indices = [
            index for index, op in enumerate(ops) if op.engine == engine
        ]
        if engine in microslot_engines:
            microslots = []
            for index in indices:
                lower, upper = bounds[index]
                lane = model.new_int_var(0, capacity - 1, f"lane_{engine}_{index}")
                microslot = model.new_int_var(
                    capacity * lower,
                    capacity * upper + capacity - 1,
                    f"microslot_{engine}_{index}",
                )
                model.add(microslot == capacity * starts[index] + lane)
                microslots.append(microslot)
            model.add_all_different(microslots)
            capacity_forms[engine] = "microslot_all_different"
        elif engine in all_different_engines:
            if capacity != 1:
                raise ValueError(
                    f"ALL_DIFFERENT_ENGINES requires capacity one ({engine})"
                )
            model.add_all_different([starts[index] for index in indices])
            capacity_forms[engine] = "all_different"
        else:
            intervals = [
                model.new_fixed_size_interval_var(starts[index], 1, f"i{index}")
                for index in indices
            ]
            model.add_cumulative(intervals, [1] * len(intervals), capacity)
            capacity_forms[engine] = "cumulative"
    for index, cycle in hints.items():
        lower, upper = bounds[index]
        model.add_hint(starts[index], min(upper, max(lower, cycle)))

    objective = None
    if bool(int(os.environ.get("OPTIMIZE", "0"))):
        objective = model.new_int_var(dag_lb - 1, horizon - 1, "makespan")
        for index in range(len(ops)):
            model.add(objective >= starts[index] + tail[index])
        model.minimize(objective)
    validation_error = model.validate()
    if validation_error:
        raise ValueError(validation_error)
    print(
        f"model vars={len(model.proto.variables)} constraints={len(model.proto.constraints)} "
        f"hinted={len(hints)} hint_radius={radius} capacity_forms={capacity_forms}",
        flush=True,
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(os.environ.get("TIME_LIMIT", "60"))
    solver.parameters.num_workers = int(os.environ.get("WORKERS", "4"))
    solver.parameters.random_seed = int(os.environ.get("RANDOM_SEED", "1"))
    solver.parameters.randomize_search = bool(int(os.environ.get("RANDOMIZE", "1")))
    solver.parameters.repair_hint = bool(int(os.environ.get("REPAIR_HINT", "1")))
    solver.parameters.log_search_progress = bool(int(os.environ.get("LOG", "0")))
    status = solver.solve(model)
    print(
        f"status={solver.status_name(status)} wall={solver.wall_time:.3f}s "
        f"conflicts={solver.num_conflicts} branches={solver.num_branches}",
        flush=True,
    )
    if objective is not None:
        print(
            f"objective={solver.objective_value + 1:g} "
            f"best_bound={solver.best_objective_bound + 1:g}",
            flush=True,
        )
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return
    result = [solver.value(start) for start in starts]
    # Raw DAG/capacity validation is deliberately repeated.  Partial arcs are
    # separately checked as they are additional constraints, not a substitute.
    validate_solution(ops, result, horizon)
    for parent, child in arcs:
        if result[child] < result[parent] + 1:
            raise AssertionError(f"partial arc violated {parent}->{child}")
    output = Path(os.environ.get("OUT", "/tmp/aopt-full-partial.json"))
    output.write_text(
        json.dumps(
            {
                "makespan": max(result) + 1,
                "cycles": result,
                "partial_edges": len(arcs),
                "strict_full_dag": True,
            }
        )
    )
    print(f"validated_makespan={max(result) + 1} output={output}", flush=True)


if __name__ == "__main__":
    main()
