"""Permute isomorphic per-group engine calendars as one atomic move.

The 959-cycle engine projections are almost fully packed.  Moving one job at
a time therefore has a very small useful neighborhood, while exchanging the
complete calendar of two structurally identical groups preserves resource
usage exactly.  This program finds the best such permutation with CP-SAT.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import heapq
import json
import os
from pathlib import Path

from ortools.sat.python import cp_model

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from problem import SLOT_LIMITS


def freeze(value: object) -> object:
    if isinstance(value, list):
        return tuple(freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(freeze(item) for item in value)
    return value


def local_ids(
    ops: list[kernel._Op], engine: str
) -> tuple[dict[int, tuple[object, ...]], dict[int, tuple[object, ...]]]:
    """Return per-operation ids and an isomorphism signature per group."""
    mode = os.environ.get("GROUP_CALENDAR_SIGNATURE", "exact")
    occurrences: Counter[tuple[object, ...]] = Counter()
    ids: dict[int, tuple[object, ...]] = {}
    members: dict[int, list[tuple[object, ...]]] = defaultdict(list)
    for index, op in enumerate(ops):
        if op.engine != engine or op.group < 0:
            continue
        if mode == "rank":
            core = ("rank",)
        elif mode == "round":
            core = ("round", op.round)
        elif op.identity:
            core = ("identity", op.round, freeze(op.identity))
        else:
            opcode = op.slot[0] if op.slot else None
            core = ("opcode", opcode, op.tag, op.round)
        occurrence_key = (op.group, core)
        occurrence = occurrences[occurrence_key]
        occurrences[occurrence_key] += 1
        local_id = (core, occurrence)
        ids[index] = local_id
        members[op.group].append(local_id)
    if mode == "rank":
        signatures = {group: (len(values),) for group, values in members.items()}
    else:
        signatures = {
            group: tuple(sorted(values, key=repr))
            for group, values in members.items()
        }
    return ids, signatures


def project_graph(
    ops: list[kernel._Op], engines: tuple[str, ...]
) -> tuple[list[int], dict[int, dict[int, int]]]:
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
    selected = [i for i in topological if ops[i].engine in engines]
    selected_set = set(selected)
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
    for child in selected:
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
            implied[parent] = max(implied.get(parent, -1), lag)
            for ancestor, distance in ancestor_distance[parent].items():
                implied[ancestor] = max(
                    implied.get(ancestor, -1), distance + lag
                )
        reduced[child] = kept
        ancestor_distance[child] = implied
    return selected, reduced


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
    engines = tuple(os.environ.get("ENGINES", "alu,valu").split(","))
    selected, projected = project_graph(ops, engines)
    engine = os.environ.get("PERMUTE_ENGINE", "valu")
    cap = int(os.environ.get("MAXIMUM", "36"))
    target_cap = int(os.environ.get("TARGET_MAXIMUM", str(cap - 1)))
    horizon = int(os.environ.get("TARGET", "959"))

    cycles: dict[int, int] = {}
    source_paths: dict[str, Path] = {}
    for path_text in os.environ["PROJECTED_HINTS"].split(","):
        path = Path(path_text)
        payload = json.loads(path.read_text())
        payload_engine = payload["engine"]
        source_paths[payload_engine] = path
        cycles.update((int(i), int(c)) for i, c in payload["cycles"].items())
    if set(cycles) != set(selected):
        raise ValueError("projected hints do not cover the selected operations")

    ids, signatures = local_ids(ops, engine)
    by_signature: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for group, signature in signatures.items():
        by_signature[signature].append(group)
    classes = [sorted(groups) for groups in by_signature.values()]
    print(
        f"engine={engine} classes={sorted(map(len, classes), reverse=True)}",
        flush=True,
    )

    index_by_group_id = {
        (ops[index].group, local_id): index for index, local_id in ids.items()
    }

    def mapped_index(target_index: int, source_group: int) -> int:
        return index_by_group_id[(source_group, ids[target_index])]

    model = cp_model.CpModel()
    assignment: dict[tuple[int, int], cp_model.IntVar] = {}
    compatible_sources: dict[int, list[int]] = {}
    for groups in classes:
        for target_group in groups:
            compatible_sources[target_group] = groups
            for source_group in groups:
                assignment[target_group, source_group] = model.new_bool_var(
                    f"g{target_group}_from_{source_group}"
                )
            model.add_exactly_one(
                assignment[target_group, source_group]
                for source_group in groups
            )
        for source_group in groups:
            model.add_exactly_one(
                assignment[target_group, source_group]
                for target_group in groups
            )

    # Same-engine projected precedences are the validity constraints.  Each
    # forbidden source-pair becomes a compact implication cut in the group
    # assignment model.
    forbidden: set[tuple[int, int, int, int]] = set()
    unary_forbidden: set[tuple[int, int]] = set()
    for child, parents in projected.items():
        if ops[child].engine != engine:
            continue
        child_group = ops[child].group
        for parent, lag in parents.items():
            if ops[parent].engine != engine:
                continue
            parent_group = ops[parent].group
            if parent_group < 0 and child_group < 0:
                if cycles[child] < cycles[parent] + lag:
                    raise ValueError("input engine hint is internally infeasible")
                continue
            if parent_group < 0:
                for child_source in compatible_sources[child_group]:
                    mapped_child = mapped_index(child, child_source)
                    if cycles[mapped_child] < cycles[parent] + lag:
                        unary_forbidden.add((child_group, child_source))
                continue
            if child_group < 0:
                for parent_source in compatible_sources[parent_group]:
                    mapped_parent = mapped_index(parent, parent_source)
                    if cycles[child] < cycles[mapped_parent] + lag:
                        unary_forbidden.add((parent_group, parent_source))
                continue
            for parent_source in compatible_sources[parent_group]:
                mapped_parent = mapped_index(parent, parent_source)
                for child_source in compatible_sources[child_group]:
                    mapped_child = mapped_index(child, child_source)
                    if cycles[mapped_child] < cycles[mapped_parent] + lag:
                        forbidden.add(
                            (
                                parent_group,
                                parent_source,
                                child_group,
                                child_source,
                            )
                        )
    for target_group, source_group in unary_forbidden:
        model.add(assignment[target_group, source_group] == 0)
    for parent_group, parent_source, child_group, child_source in forbidden:
        left = assignment[parent_group, parent_source]
        right = assignment[child_group, child_source]
        if left is right:
            model.add(left == 0)
        else:
            model.add(left + right <= 1)
    print(
        f"unary_forbidden={len(unary_forbidden)} pair_forbidden={len(forbidden)}",
        flush=True,
    )

    costs: dict[tuple[int, int], tuple[int, int]] = {}
    constant_excess = 0
    constant_sum = 0
    for child, parents in projected.items():
        for parent, lag in parents.items():
            if ops[parent].engine == ops[child].engine:
                continue
            permuted = child if ops[child].engine == engine else parent
            group = ops[permuted].group
            if group < 0:
                amount = max(0, cycles[parent] + lag - cycles[child])
                constant_excess += max(0, amount - target_cap)
                constant_sum += amount
                continue
            for source_group in compatible_sources[group]:
                mapped = mapped_index(permuted, source_group)
                parent_cycle = cycles[mapped] if permuted == parent else cycles[parent]
                child_cycle = cycles[mapped] if permuted == child else cycles[child]
                amount = max(0, parent_cycle + lag - child_cycle)
                if amount > cap:
                    unary_forbidden.add((group, source_group))
                old_excess, old_sum = costs.get((group, source_group), (0, 0))
                costs[group, source_group] = (
                    old_excess + max(0, amount - target_cap),
                    old_sum + amount,
                )
    for target_group, source_group in unary_forbidden:
        model.add(assignment[target_group, source_group] == 0)
    secondary_scale = sum(value[1] for value in costs.values()) + 1
    model.minimize(
        sum(
            assignment[key] * (excess * secondary_scale + violation_sum)
            for key, (excess, violation_sum) in costs.items()
        )
    )
    for group, sources in compatible_sources.items():
        if group in sources:
            model.add_hint(assignment[group, group], 1)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(os.environ.get("TIME_LIMIT", "300"))
    solver.parameters.num_workers = int(os.environ.get("WORKERS", "8"))
    status = solver.solve(model)
    print(
        f"status={solver.status_name(status)} objective={solver.objective_value}",
        flush=True,
    )
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return

    permutation = {
        target_group: next(
            source_group
            for source_group in sources
            if solver.value(assignment[target_group, source_group])
        )
        for target_group, sources in compatible_sources.items()
    }
    result = dict(cycles)
    for index, local_id in ids.items():
        source_group = permutation[ops[index].group]
        result[index] = cycles[index_by_group_id[source_group, local_id]]

    usage = Counter((ops[i].engine, result[i]) for i in selected)
    if any(demand > SLOT_LIMITS[e] for (e, _), demand in usage.items()):
        raise AssertionError("calendar permutation changed resource feasibility")
    violations = []
    for child, parents in projected.items():
        for parent, lag in parents.items():
            amount = result[parent] + lag - result[child]
            if ops[parent].engine == ops[child].engine and amount > 0:
                raise AssertionError((parent, child, lag, amount))
            if ops[parent].engine != ops[child].engine and amount > 0:
                violations.append(amount)
    metrics = (
        max(violations, default=0),
        sum(violations),
        len(violations),
    )
    print(
        f"metrics={metrics} excess={sum(max(0, v - target_cap) for v in violations)} "
        f"permutation={json.dumps(permutation, sort_keys=True)}",
        flush=True,
    )

    output_prefix = Path(os.environ.get("OUT_PREFIX", "/tmp/aopt-group-permute"))
    for output_engine in engines:
        output = Path(f"{output_prefix}-{output_engine}.json")
        output.write_text(
            json.dumps(
                {
                    "engine": output_engine,
                    "horizon": horizon,
                    "cycles": {
                        str(i): result[i]
                        for i in selected
                        if ops[i].engine == output_engine
                    },
                }
            )
        )
        print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
