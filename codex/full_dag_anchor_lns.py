"""Exact whole-DAG LNS that releases cross-engine anchors around one deletion.

This is a stricter, more flexible sibling of ``repair_schedule.py``.  Given a
legal full schedule, it removes one calendar cycle and fixes most operations to
their naturally shifted time.  Operations in a time window are free, then the
neighborhood is expanded through cross-engine dependencies and every edge made
invalid by the deletion.  The resulting CP-SAT model retains every effective
DAG edge and every engine capacity, so a feasible output is a globally valid
schedule rather than a projection repair.

It is useful when a pure time-window repair gets stuck because the window
boundary freezes the producer or consumer that carries a phase conflict.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import replace
import json
import os
from pathlib import Path

from ortools.sat.python import cp_model

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from codex.partial_order_hints import audit_augmented_dag, load_partial_order_hints
from codex.solve_full_dag import topological_data, validate_solution
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


def parse_schedule(path: Path, ops: list[kernel._Op]) -> list[int]:
    payload = json.loads(path.read_text())
    raw = payload["cycles"]
    if not isinstance(raw, list) or len(raw) != len(ops):
        raise ValueError(f"{path}: expected a full cycles list for {len(ops)} ops")
    return [int(value) for value in raw]


def selected_partial_arcs(
    ops: list[kernel._Op],
) -> tuple[tuple[tuple[int, int], ...], int | None]:
    """Load a fail-closed subset of explicit partial resource-order arcs."""
    raw_paths = tuple(
        value for value in os.environ.get("PARTIAL_ORDER_HINTS", "").split(",") if value
    )
    if not raw_paths:
        return (), None
    partial = load_partial_order_hints(raw_paths, ops)
    # Validate the source before filtering: a typo must not disappear merely
    # because it lands beyond a cap or stride.
    audit_augmented_dag(ops, partial)
    engines = {
        value
        for value in os.environ.get("PARTIAL_ARC_ENGINES", "").split(",")
        if value
    }
    unknown = engines - set(SLOT_LIMITS)
    if unknown:
        raise ValueError(f"unknown PARTIAL_ARC_ENGINES: {sorted(unknown)}")
    # A full retained order can contain tens of thousands of arcs.  Default
    # to a small, evenly sampled batch so an ordinary LNS invocation remains
    # a neighborhood search.  Set PARTIAL_ARC_MAX=0 and STRIDE=1 explicitly
    # only for the intentional full hardening experiment.
    requested_stride = int(os.environ.get("PARTIAL_ARC_STRIDE", "0"))
    if requested_stride < 0:
        raise ValueError("PARTIAL_ARC_STRIDE must be non-negative")
    maximum = int(os.environ.get("PARTIAL_ARC_MAX", "512"))
    candidates = [
        edge
        for edge in partial.effective_edges
        if not engines or ops[edge[0]].engine in engines
    ]
    stride = requested_stride
    if stride == 0:
        stride = max(1, (len(candidates) + max(1, maximum) - 1) // max(1, maximum))
    selected = tuple(
        edge for position, edge in enumerate(candidates) if position % stride == 0
    )
    if maximum > 0:
        selected = selected[:maximum]
    selected_audit = audit_augmented_dag(
        ops, replace(partial, effective_edges=selected)
    )
    print(
        f"partial_order_hints paths={len(partial.paths)} raw={len(partial.resource_edges)} "
        f"effective={len(partial.effective_edges)} selected={len(selected)} "
        f"stride={stride} requested_stride={requested_stride} cap={maximum} "
        f"engines={sorted(engines)} "
        f"augmented_span={selected_audit.longest_path_span}",
        flush=True,
    )
    return selected, selected_audit.longest_path_span


def neighborhood(
    ops: list[kernel._Op],
    children: list[list[tuple[int, int]]],
    base: list[int],
    deleted: int,
    window: int,
    depth: int,
    pad: int,
    partial_edges: tuple[tuple[int, int], ...],
    partial_mode: str,
) -> set[int]:
    """Select a time window plus nearby cross-engine anchor operations."""
    local = {
        index
        for index, cycle in enumerate(base)
        if deleted - window <= cycle <= deleted + window
    }
    # A positive-lag edge crossing the deleted cycle becomes invalid after
    # shifting only its child side.  Always free both endpoints; this is a
    # correctness closure, not a heuristic.
    changed = True
    while changed:
        changed = False
        for child, op in enumerate(ops):
            for parent, lag in op.parents.items():
                if base[child] < base[parent] + lag and (
                    parent not in local or child not in local
                ):
                    local.update((parent, child))
                    changed = True

    if partial_mode not in {"none", "violated", "near", "all"}:
        raise ValueError(
            "PARTIAL_ANCHOR_MODE must be none, violated, near, or all"
        )
    partial_lower = deleted - window - pad
    partial_upper = deleted + window + pad
    for parent, child in partial_edges:
        violated = base[child] < base[parent] + 1
        near = (
            partial_lower <= base[parent] <= partial_upper
            or partial_lower <= base[child] <= partial_upper
        )
        if partial_mode == "all" or (
            partial_mode == "near" and (near or violated)
        ) or (partial_mode == "violated" and violated):
            local.update((parent, child))

    frontier = deque((index, 0) for index in sorted(local))
    visited = set(local)
    lower = deleted - window - pad
    upper = deleted + window + pad
    while frontier:
        index, distance = frontier.popleft()
        if distance >= depth:
            continue
        related = [parent for parent in ops[index].parents]
        related.extend(child for child, _ in children[index])
        for other in related:
            if other in visited or ops[other].engine == ops[index].engine:
                continue
            if not lower <= base[other] <= upper:
                continue
            visited.add(other)
            local.add(other)
            frontier.append((other, distance + 1))
    return local


def main() -> None:
    schedule_path = Path(os.environ["SCHEDULE_IN"])
    ops = build_ops()
    _, children, earliest, tail = topological_data(ops)
    old = parse_schedule(schedule_path, ops)
    old_horizon = max(old) + 1
    validate_solution(ops, old, old_horizon)
    deleted = int(os.environ.get("DELETE", str(old_horizon - 2)))
    if not 0 <= deleted < old_horizon:
        raise ValueError(f"DELETE={deleted} outside 0:{old_horizon - 1}")
    horizon = old_horizon - 1
    base = [cycle if cycle < deleted else cycle - 1 for cycle in old]
    window = int(os.environ.get("WINDOW", "48"))
    depth = int(os.environ.get("CROSS_ENGINE_DEPTH", "2"))
    pad = int(os.environ.get("ANCHOR_PAD", "24"))
    move = int(os.environ.get("MOVE_RADIUS", str(window + pad)))
    partial_edges, partial_span = selected_partial_arcs(ops)
    partial_mode = os.environ.get("PARTIAL_ANCHOR_MODE", "violated")
    local = neighborhood(
        ops,
        children,
        base,
        deleted,
        window,
        depth,
        pad,
        partial_edges,
        partial_mode,
    )
    fixed = set(range(len(ops))) - local
    print(
        f"strict_anchor_lns old={old_horizon} target={horizon} delete={deleted} "
        f"window={window} depth={depth} pad={pad} move={move} "
        f"local={len(local)} fixed={len(fixed)} partial_arcs={len(partial_edges)} "
        f"partial_mode={partial_mode} partial_span={partial_span}",
        flush=True,
    )

    model = cp_model.CpModel()
    starts: dict[int, cp_model.IntVar] = {}
    bounds: dict[int, tuple[int, int]] = {}
    for index in sorted(local):
        lower = max(earliest[index], base[index] - move, 0)
        upper = min(horizon - 1 - tail[index], base[index] + move)
        if lower > upper:
            print(
                f"status=INFEASIBLE reason=empty_local_domain i={index} "
                f"base={base[index]} bounds={lower}:{upper}",
                flush=True,
            )
            return
        starts[index] = model.new_int_var(lower, upper, f"s{index}")
        bounds[index] = (lower, upper)

    # Every original edge remains present.  Fixed/fixed violations cannot be
    # repaired by this neighborhood and are reported rather than hidden.
    for child, op in enumerate(ops):
        child_local = child in local
        for parent, lag in op.parents.items():
            parent_local = parent in local
            if child_local and parent_local:
                model.add(starts[child] >= starts[parent] + lag)
            elif child_local:
                model.add(starts[child] >= base[parent] + lag)
            elif parent_local:
                model.add(base[child] >= starts[parent] + lag)
            elif base[child] < base[parent] + lag:
                print(
                    f"status=INFEASIBLE reason=fixed_edge {parent}->{child} "
                    f"lag={lag} cycles={base[parent]}->{base[child]}",
                    flush=True,
                )
                return

    for parent, child in partial_edges:
        parent_local = parent in local
        child_local = child in local
        if parent_local and child_local:
            model.add(starts[child] >= starts[parent] + 1)
        elif child_local:
            model.add(starts[child] >= base[parent] + 1)
        elif parent_local:
            model.add(base[child] >= starts[parent] + 1)
        elif base[child] < base[parent] + 1:
            print(
                f"status=INFEASIBLE reason=fixed_partial_arc {parent}->{child} "
                f"cycles={base[parent]}->{base[child]}",
                flush=True,
            )
            return

    fixed_use: dict[str, Counter[int]] = defaultdict(Counter)
    for index in fixed:
        fixed_use[ops[index].engine][base[index]] += 1
    for engine, capacity in SLOT_LIMITS.items():
        intervals: list[cp_model.IntervalVar] = []
        demands: list[int] = []
        for index in local:
            if ops[index].engine == engine:
                intervals.append(model.new_fixed_size_interval_var(starts[index], 1, f"i{index}"))
                demands.append(1)
        for cycle, demand in fixed_use[engine].items():
            if demand > capacity:
                raise AssertionError(f"fixed capacity overflow {engine}@{cycle}")
            intervals.append(model.new_fixed_size_interval_var(cycle, 1, f"f_{engine}_{cycle}"))
            demands.append(demand)
        if intervals:
            model.add_cumulative(intervals, demands, capacity)
    for index, start in starts.items():
        lower, upper = bounds[index]
        model.add_hint(start, min(upper, max(lower, base[index])))

    validation_error = model.validate()
    if validation_error:
        raise ValueError(validation_error)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(os.environ.get("TIME_LIMIT", "10"))
    solver.parameters.num_workers = int(os.environ.get("WORKERS", "4"))
    solver.parameters.random_seed = int(os.environ.get("RANDOM_SEED", "1"))
    solver.parameters.randomize_search = bool(int(os.environ.get("RANDOMIZE", "1")))
    solver.parameters.repair_hint = bool(int(os.environ.get("REPAIR_HINT", "0")))
    solver.parameters.log_search_progress = bool(int(os.environ.get("LOG", "0")))
    status = solver.solve(model)
    print(
        f"status={solver.status_name(status)} wall={solver.wall_time:.3f}s "
        f"conflicts={solver.num_conflicts} branches={solver.num_branches}",
        flush=True,
    )
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return
    result = base.copy()
    for index, start in starts.items():
        result[index] = solver.value(start)
    validate_solution(ops, result, horizon)
    output = Path(os.environ.get("OUT", "/tmp/aopt-anchor-lns.json"))
    output.write_text(
        json.dumps(
            {
                "makespan": max(result) + 1,
                "cycles": result,
                "deleted": deleted,
                "local": sorted(local),
                "strict_full_dag": True,
            }
        )
    )
    print(f"validated_makespan={max(result) + 1} output={output}", flush=True)


if __name__ == "__main__":
    main()
