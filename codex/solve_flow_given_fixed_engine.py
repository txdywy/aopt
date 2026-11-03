"""Solve one unit-capacity engine against a fixed calendar of another engine.

The full DAG contains many omitted engines between FLOW and LOAD.  If LOAD
cycles are fixed, those operations become exact release/deadline anchors.  A
small projection keeps only FLOW-to-FLOW paths and the longest paths that do
not cross FLOW, so CP-SAT solves the resulting 941-job all-different problem
without inventing a relaxed schedule for the fixed engine.
"""

from __future__ import annotations

from collections import Counter
import heapq
import json
import os
from pathlib import Path

from ortools.sat.python import cp_model

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from problem import SLOT_LIMITS


def topo_data(ops):
    children = [[] for _ in ops]
    indegree = [len(op.parents) for op in ops]
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            children[parent].append((child, lag))
    ready = [i for i, d in enumerate(indegree) if d == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        parent = heapq.heappop(ready)
        order.append(parent)
        for child, _ in children[parent]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(order) != len(ops):
        raise ValueError("kernel DAG is cyclic")
    return order, children


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
    horizon = int(os.environ.get("TARGET", "959"))
    solve_engine = os.environ.get("SOLVE_ENGINE", "flow")
    fixed_engine = os.environ.get("FIXED_ENGINE", "load")
    if solve_engine == fixed_engine:
        raise ValueError("SOLVE_ENGINE and FIXED_ENGINE must differ")
    fixed_payload = json.loads(Path(os.environ["FIXED_HINT"]).read_text())
    fixed = {
        int(index): int(cycle)
        for index, cycle in fixed_payload["cycles"].items()
    }
    fixed_set = {i for i, op in enumerate(ops) if op.engine == fixed_engine}
    solve_set = {i for i, op in enumerate(ops) if op.engine == solve_engine}
    if set(fixed) != fixed_set:
        raise ValueError("FIXED_HINT does not cover the fixed engine")
    capacity = SLOT_LIMITS[solve_engine]
    if max(Counter(fixed.values()).values(), default=0) > SLOT_LIMITS[fixed_engine]:
        raise ValueError("fixed calendar exceeds its engine capacity")

    order, children = topo_data(ops)
    solve_order = [i for i in order if i in solve_set]

    # Longest root/fixed-engine path to every node without crossing a solve
    # engine operation.  A fixed operation is an exact anchor, not a variable.
    release = [-1] * len(ops)
    for node in order:
        if node in fixed_set:
            release[node] = fixed[node]
            continue
        best = 0 if not ops[node].parents else -1
        for parent, lag in ops[node].parents.items():
            if parent in solve_set:
                continue
            if release[parent] >= 0:
                best = max(best, release[parent] + lag)
        release[node] = best

    # Reverse longest path to a fixed anchor/terminal without crossing solve.
    deadline = [horizon - 1] * len(ops)
    for node in reversed(order):
        if node in fixed_set:
            deadline[node] = fixed[node]
            continue
        # A path ending at a solve-engine child is intentionally omitted: the
        # child remains variable, so it must not manufacture a false deadline
        # for this node.  Start from the unconstrained horizon and tighten
        # only through fixed/non-solve descendants.
        best = horizon - 1
        for child, lag in children[node]:
            if child in solve_set:
                continue
            if child in fixed_set:
                best = min(best, fixed[child] - lag)
            elif deadline[child] >= 0:
                best = min(best, deadline[child] - lag)
        deadline[node] = best

    # Exact weighted projection between consecutive solve-engine operations.
    frontier = [{} for _ in ops]
    projected = {}
    for child in order:
        incoming = {}
        for parent, lag in ops[child].parents.items():
            sources = ({parent: 0} if parent in solve_set else frontier[parent])
            for source, distance in sources.items():
                incoming[source] = max(incoming.get(source, -1), distance + lag)
        if child in solve_set:
            projected[child] = incoming
            frontier[child] = {child: 0}
        else:
            frontier[child] = incoming

    bounds = {}
    for i in solve_order:
        lower = max(0, release[i])
        upper = min(horizon - 1, deadline[i])
        if lower > upper:
            raise ValueError(f"empty fixed-engine window for {i}: {lower}:{upper}")
        bounds[i] = [lower, upper]

    model = cp_model.CpModel()
    starts = {
        i: model.new_int_var(bounds[i][0], bounds[i][1], f"s{i}")
        for i in solve_order
    }
    for child, parents in projected.items():
        for parent, lag in parents.items():
            model.add(starts[child] >= starts[parent] + lag)
    if bool(int(os.environ.get("NO_CAPACITY", "0"))):
        print("capacity_constraint=disabled_diagnostic", flush=True)
    elif capacity == 1:
        model.add_all_different([starts[i] for i in solve_order])
    else:
        intervals = [
            model.new_fixed_size_interval_var(starts[i], 1, f"i{i}")
            for i in solve_order
        ]
        model.add_cumulative(intervals, [1] * len(intervals), capacity)

    if "SOLVE_HINT" in os.environ:
        payload = json.loads(Path(os.environ["SOLVE_HINT"]).read_text())
        cycles = {int(i): int(c) for i, c in payload["cycles"].items()}
        if set(cycles) != solve_set:
            raise ValueError("SOLVE_HINT does not cover solve engine")
        for i in solve_order:
            model.add_hint(starts[i], min(bounds[i][1], max(bounds[i][0], cycles[i])))

    print(
        f"solve_engine={solve_engine} fixed_engine={fixed_engine} "
        f"jobs={len(solve_order)} projected_edges={sum(map(len, projected.values()))} "
        f"domain_values={sum(hi-lo+1 for lo, hi in bounds.values())}",
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
        f"status={solver.status_name(status)} wall={solver.wall_time:g} "
        f"branches={solver.num_branches} conflicts={solver.num_conflicts}",
        flush=True,
    )
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return
    result = {i: solver.value(starts[i]) for i in solve_order}
    usage = Counter(result.values())
    print(
        f"first={min(result.values())} last={max(result.values())} "
        f"holes={horizon * capacity - len(result)} "
        f"max_usage={max(usage.values(), default=0)}",
        flush=True,
    )
    out = Path(os.environ.get("OUT", "/tmp/aopt-fixed-engine-solve.json"))
    out.write_text(
        json.dumps(
            {
                "engine": solve_engine,
                "fixed_engine": fixed_engine,
                "horizon": horizon,
                "cycles": {str(i): result[i] for i in solve_order},
                "fixed_hint": os.environ["FIXED_HINT"],
            }
        )
    )
    print(f"output={out}", flush=True)


if __name__ == "__main__":
    main()
