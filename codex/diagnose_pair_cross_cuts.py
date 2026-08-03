"""Find a small semantic class of VALU/Flow edges that blocks H959.

This is an offline compiler diagnostic.  It keeps the projected VALU/Flow
graph and both exact physical calendars, but makes selected *cross-engine*
precedence classes optional.  The resulting cut classes are not a legal
schedule; they identify which compiler lowering must remove, retime, or keep
on one engine to make the two saturated calendars compatible.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import heapq
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


def projected_pair(ops: list[kernel._Op], engines: tuple[str, ...]):
    """Build the same weighted projection used by solve_projected_engines."""
    children: list[list[tuple[int, int]]] = [[] for _ in ops]
    indegree = [0] * len(ops)
    for child, op in enumerate(ops):
        indegree[child] = len(op.parents)
        for parent, lag in op.parents.items():
            children[parent].append((child, lag))
    ready = [i for i, degree in enumerate(indegree) if not degree]
    heapq.heapify(ready)
    topological: list[int] = []
    while ready:
        parent = heapq.heappop(ready)
        topological.append(parent)
        for child, _ in children[parent]:
            indegree[child] -= 1
            if not indegree[child]:
                heapq.heappush(ready, child)
    if len(topological) != len(ops):
        raise ValueError("kernel DAG is cyclic")
    position = {index: rank for rank, index in enumerate(topological)}
    selected_set = {
        i for i in topological if ops[i].engine in engines
    }
    frontier: list[dict[int, int]] = [{} for _ in ops]
    projected: dict[int, dict[int, int]] = {}
    for child in topological:
        incoming: dict[int, int] = {}
        for parent, lag in ops[child].parents.items():
            sources = ({parent: 0} if parent in selected_set else frontier[parent])
            for source, distance in sources.items():
                incoming[source] = max(incoming.get(source, -1), distance + lag)
        if child in selected_set:
            projected[child] = incoming
            frontier[child] = {child: 0}
        else:
            frontier[child] = incoming

    reduced: dict[int, dict[int, int]] = {}
    ancestor_distance: dict[int, dict[int, int]] = {}
    for child in topological:
        if child not in selected_set:
            continue
        kept: dict[int, int] = {}
        implied: dict[int, int] = {}
        for parent, lag in sorted(
            projected[child].items(),
            key=lambda item: position[item[0]],
            reverse=True,
        ):
            if implied.get(parent, -1) >= lag:
                continue
            kept[parent] = lag
            implied[parent] = max(implied.get(parent, -1), distance := lag)
            for ancestor, distance in ancestor_distance[parent].items():
                implied[ancestor] = max(implied.get(ancestor, -1), distance + lag)
        reduced[child] = kept
        ancestor_distance[child] = implied
    selected = [i for i in topological if i in selected_set]
    return selected, reduced


def edge_label(parent: int, child: int, lag: int, ops: list[kernel._Op], mode: str):
    p, c = ops[parent], ops[child]
    if mode == "tag":
        return (p.engine, p.tag, c.engine, c.tag)
    if mode == "tag_round":
        return (p.engine, p.tag, c.engine, c.tag, c.round)
    if mode == "tag_group":
        return (p.engine, p.tag, c.engine, c.tag, c.group)
    if mode == "group_round":
        return (p.engine, p.group, p.round, c.engine, c.group, c.round)
    raise ValueError("GROUPING must be tag, tag_round, tag_group, or group_round")


def main() -> None:
    ops = build_ops()
    horizon = int(os.environ.get("TARGET", "959"))
    engines = tuple(value for value in os.environ.get("ENGINES", "valu,flow").split(",") if value)
    if len(engines) != 2 or any(engine not in SLOT_LIMITS for engine in engines):
        raise ValueError("ENGINES must name exactly two physical engines")
    selected, projected = projected_pair(ops, engines)
    selected_set = set(selected)
    # Full projected earliest/latest bounds retain every hard projected edge.
    children: dict[int, list[tuple[int, int]]] = {i: [] for i in selected}
    indegree = {i: 0 for i in selected}
    for child, parents in projected.items():
        for parent, lag in parents.items():
            children[parent].append((child, lag))
            indegree[child] += 1
    ready = [i for i in selected if not indegree[i]]
    heapq.heapify(ready)
    topo: list[int] = []
    while ready:
        parent = heapq.heappop(ready)
        topo.append(parent)
        for child, _ in children[parent]:
            indegree[child] -= 1
            if not indegree[child]:
                heapq.heappush(ready, child)
    earliest = {i: 0 for i in selected}
    for child in topo:
        earliest[child] = max(
            (earliest[parent] + lag for parent, lag in projected[child].items()),
            default=0,
        )
    tail = {i: 0 for i in selected}
    for parent in reversed(topo):
        tail[parent] = max((lag + tail[child] for child, lag in children[parent]), default=0)
    if any(earliest[i] > horizon - 1 - tail[i] for i in selected):
        raise ValueError("projected DAG itself is impossible at target")

    grouping = os.environ.get("GROUPING", "tag")
    edge_classes: dict[tuple[object, ...], list[tuple[int, int, int]]] = defaultdict(list)
    hard_edges: list[tuple[int, int, int]] = []
    for child in selected:
        for parent, lag in projected[child].items():
            if ops[parent].engine == ops[child].engine:
                hard_edges.append((parent, child, lag))
            else:
                edge_classes[edge_label(parent, child, lag, ops, grouping)].append((parent, child, lag))
    ordered_classes = sorted(edge_classes, key=str)
    print(
        f"pair_cross_cuts ops={len(selected)} edges={sum(map(len, projected.values()))} "
        f"hard_same={len(hard_edges)} cross={sum(map(len, edge_classes.values()))} "
        f"classes={len(ordered_classes)} grouping={grouping} horizon={horizon}",
        flush=True,
    )
    model = cp_model.CpModel()
    starts = {
        i: model.new_int_var(earliest[i], horizon - 1 - tail[i], f"s{i}")
        for i in selected
    }
    for parent, child, lag in hard_edges:
        model.add(starts[child] >= starts[parent] + lag)
    cuts = {
        key: model.new_bool_var(f"cut_{rank}")
        for rank, key in enumerate(ordered_classes)
    }
    for key, edges in edge_classes.items():
        literal = cuts[key]
        for parent, child, lag in edges:
            model.add(starts[child] >= starts[parent] + lag).only_enforce_if(literal.Not())
    for engine in engines:
        capacity = SLOT_LIMITS[engine]
        indices = [i for i in selected if ops[i].engine == engine]
        # AllDifferent is a useful exact diagnostic, but its lane variables
        # are expensive for a 7k-job pair.  Cumulative is equally exact for
        # unit-duration jobs and is selectable for low-memory UNSAT-core
        # searches.
        if os.environ.get("CAPACITY_MODE", "microslot") == "cumulative":
            intervals = [
                model.new_fixed_size_interval_var(starts[i], 1, f"i{i}")
                for i in indices
            ]
            model.add_cumulative(intervals, [1] * len(intervals), capacity)
        else:
            microslots = []
            for i in indices:
                lane = model.new_int_var(0, capacity - 1, f"lane_{engine}_{i}")
                micro = model.new_int_var(
                    0, capacity * horizon + len(selected), f"micro_{engine}_{i}"
                )
                model.add(micro == capacity * starts[i] + lane)
                microslots.append(micro)
            model.add_all_different(microslots)
    assumptions = bool(int(os.environ.get("ASSUMPTIONS", "0")))
    if assumptions:
        # ``keep`` is true in the strict graph.  An UNSAT core is a compact
        # certificate of the semantic cross-engine classes that cannot all
        # remain hard; it is much easier to solve than a weighted Max-SAT
        # objective over forty phase classes.
        model.add_assumptions([literal.Not() for literal in cuts.values()])
    else:
        # First minimize number of semantic classes, then number of cut edges.
        model.minimize(
            sum(cuts.values()) * 10000
            + sum(len(edge_classes[key]) * cuts[key] for key in ordered_classes)
        )
        for literal in cuts.values():
            model.add_hint(literal, 1)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(os.environ.get("TIME_LIMIT", "120"))
    solver.parameters.num_workers = int(os.environ.get("WORKERS", "8"))
    solver.parameters.random_seed = int(os.environ.get("RANDOM_SEED", "1"))
    solver.parameters.randomize_search = bool(int(os.environ.get("RANDOMIZE", "0")))
    solver.parameters.repair_hint = bool(int(os.environ.get("REPAIR_HINT", "1")))
    status = solver.solve(model)
    print(
        f"status={solver.status_name(status)} wall={solver.wall_time:.3f}s "
        f"objective={solver.objective_value:g} bound={solver.best_objective_bound:g} "
        f"assumptions={int(assumptions)}",
        flush=True,
    )
    if assumptions and status == cp_model.INFEASIBLE:
        by_index = {literal.index: key for key, literal in cuts.items()}
        raw_core = solver.sufficient_assumptions_for_infeasibility()
        print(f"raw_assumption_core={list(raw_core)}", flush=True)
        core = [
            by_index[abs(index)]
            for index in raw_core
            if abs(index) in by_index
        ]
        print(f"cross_class_core={len(core)}", flush=True)
        for key in core:
            print(f"  core_class={key} edges={len(edge_classes[key])}", flush=True)
        return
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return
    chosen = [key for key in ordered_classes if solver.value(cuts[key])]
    print(f"cut_classes={len(chosen)} cut_edges={sum(len(edge_classes[key]) for key in chosen)}", flush=True)
    for key in chosen:
        print(f"  class={key} edges={len(edge_classes[key])}", flush=True)
        for parent, child, lag in edge_classes[key][:8]:
            print(
                f"    {parent}:{ops[parent].tag}:g{ops[parent].group}:r{ops[parent].round} "
                f"->{child}:{ops[child].tag}:g{ops[child].group}:r{ops[child].round} lag={lag}",
                flush=True,
            )


if __name__ == "__main__":
    main()
