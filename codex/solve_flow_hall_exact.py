"""Solve FLOW order directly against one or more LOAD Hall cuts.

For a fixed horizon and a Hall window ``[0, right]``, a load can be placed
after the window iff every closest downstream FLOW operation starts strictly
after the load-to-FLOW path distance plus ``right``.  This gives a compact
exact model with only the FLOW start variables and one Boolean per
load/cut, instead of jointly time-indexing all FLOW and LOAD operations.
"""

from __future__ import annotations

from collections import Counter
import heapq
import json
import os
from pathlib import Path

import numpy as np
from ortools.sat.python import cp_model

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from problem import SLOT_LIMITS


def topological_order(
    ops: list[kernel._Op],
    parents: list[dict[int, int]] | None = None,
) -> tuple[list[int], list[list[tuple[int, int]]]]:
    actual_parents = parents or [dict(op.parents) for op in ops]
    children: list[list[tuple[int, int]]] = [[] for _ in ops]
    indegree = [len(item) for item in actual_parents]
    for child, child_parents in enumerate(actual_parents):
        for parent, lag in child_parents.items():
            children[parent].append((child, lag))
    ready = [i for i, degree in enumerate(indegree) if not degree]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        parent = heapq.heappop(ready)
        order.append(parent)
        for child, _ in children[parent]:
            indegree[child] -= 1
            if not indegree[child]:
                heapq.heappush(ready, child)
    if len(order) != len(ops):
        raise ValueError("operation graph is cyclic")
    return order, children


def full_windows(
    ops: list[kernel._Op],
    order: list[int],
    children: list[list[tuple[int, int]]],
    horizon: int,
    parents: list[dict[int, int]] | None = None,
) -> tuple[list[int], list[int]]:
    actual_parents = parents or [dict(op.parents) for op in ops]
    earliest = [0] * len(ops)
    for child in order:
        earliest[child] = max(
            (
                earliest[parent] + lag
                for parent, lag in actual_parents[child].items()
            ),
            default=0,
        )
    latest = [horizon - 1] * len(ops)
    for parent in reversed(order):
        latest[parent] = min(
            (
                latest[child] - lag
                for child, lag in children[parent]
            ),
            default=horizon - 1,
        )
    return earliest, latest


def worst_hall(
    ops: list[kernel._Op],
    earliest: list[int],
    latest: list[int],
    horizon: int,
) -> tuple[int, int, int, int]:
    loads = [i for i, op in enumerate(ops) if op.engine == "load"]
    matrix = np.zeros((horizon, horizon), dtype=np.int16)
    np.add.at(
        matrix,
        (
            np.fromiter((earliest[i] for i in loads), dtype=np.int32),
            np.fromiter((latest[i] for i in loads), dtype=np.int32),
        ),
        1,
    )
    contained = np.cumsum(
        np.cumsum(matrix[::-1], axis=0, dtype=np.int32)[::-1],
        axis=1,
        dtype=np.int32,
    )
    best = (-10**9, 0, 0, 0)
    capacity = SLOT_LIMITS["load"]
    for left in range(horizon):
        overloads = contained[left, left:] - capacity * np.arange(
            1, horizon - left + 1, dtype=np.int32
        )
        offset = int(np.argmax(overloads))
        right = left + offset
        candidate = (
            int(overloads[offset]),
            left,
            right,
            int(contained[left, right]),
        )
        if candidate[0] > best[0]:
            best = candidate
    return best


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
    rights = tuple(
        int(value)
        for value in os.environ.get("RIGHTS", "853").split(",")
        if value
    )
    if not rights or any(right < 0 or right >= horizon for right in rights):
        raise ValueError("RIGHTS must contain cycles inside the horizon")

    topological, children = topological_order(ops)
    position = {index: rank for rank, index in enumerate(topological)}
    earliest, latest = full_windows(
        ops, topological, children, horizon
    )
    if any(left > right for left, right in zip(earliest, latest)):
        raise ValueError("bare DAG does not fit the requested horizon")

    flow = [i for i in topological if ops[i].engine == "flow"]
    flow_set = set(flow)
    loads = [i for i in topological if ops[i].engine == "load"]

    # Project all paths between consecutive FLOW operations.
    frontier: list[dict[int, int]] = [{} for _ in ops]
    projected: dict[int, dict[int, int]] = {}
    for child in topological:
        incoming: dict[int, int] = {}
        for parent, lag in ops[child].parents.items():
            sources = ({parent: 0} if parent in flow_set else frontier[parent])
            for source, distance in sources.items():
                incoming[source] = max(
                    incoming.get(source, -1), distance + lag
                )
        if child in flow_set:
            projected[child] = incoming
            frontier[child] = {child: 0}
        else:
            frontier[child] = incoming

    reduced: dict[int, dict[int, int]] = {}
    ancestor_distance: dict[int, dict[int, int]] = {}
    for child in flow:
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
    projected = reduced

    # Reverse projection: for every non-FLOW operation, retain the closest
    # downstream FLOW operations and the longest path to a terminal that does
    # not cross FLOW.  The latter is an order-independent deadline.
    downstream_flow: list[dict[int, int]] = [{} for _ in ops]
    terminal_tail = [-1] * len(ops)
    for parent in reversed(topological):
        if parent in flow_set:
            downstream_flow[parent] = {parent: 0}
            terminal_tail[parent] = -1
            continue
        outgoing: dict[int, int] = {}
        terminal = 0 if not children[parent] else -1
        for child, lag in children[parent]:
            sources = (
                {child: 0}
                if child in flow_set
                else downstream_flow[child]
            )
            for target, distance in sources.items():
                outgoing[target] = max(
                    outgoing.get(target, -1), lag + distance
                )
            if child not in flow_set and terminal_tail[child] >= 0:
                terminal = max(terminal, lag + terminal_tail[child])
        downstream_flow[parent] = outgoing
        terminal_tail[parent] = terminal

    hint: dict[int, int] = {}
    if "FLOW_HINT" in os.environ:
        payload = json.loads(Path(os.environ["FLOW_HINT"]).read_text())
        hint = {
            int(index): int(cycle)
            for index, cycle in payload["cycles"].items()
        }
        if set(hint) != flow_set:
            raise ValueError("FLOW_HINT does not match this FLOW graph")

    domain_radius = int(os.environ.get("DOMAIN_RADIUS", "-1"))
    bounds: dict[int, tuple[int, int]] = {}
    for i in flow:
        lower, upper = earliest[i], latest[i]
        if domain_radius >= 0 and i in hint:
            center = min(upper, max(lower, hint[i]))
            lower = max(lower, center - domain_radius)
            upper = min(upper, center + domain_radius)
        bounds[i] = (lower, upper)

    model = cp_model.CpModel()
    starts = {
        i: model.new_int_var(bounds[i][0], bounds[i][1], f"s{i}")
        for i in flow
    }
    for child, child_parents in projected.items():
        for parent, lag in child_parents.items():
            model.add(starts[child] >= starts[parent] + lag)
    model.add_all_different(starts.values())
    if hint:
        hint_order = sorted(flow, key=lambda i: (hint[i], i))
        fixed_prefix = int(os.environ.get("FIX_HINT_PREFIX", "0"))
        fixed_prefix = min(len(hint_order), max(0, fixed_prefix))
        for previous, current in zip(
            hint_order[:fixed_prefix],
            hint_order[1:fixed_prefix],
        ):
            model.add(starts[current] >= starts[previous] + 1)
        if bool(int(os.environ.get("FIX_HINT_PREFIX_CYCLES", "0"))):
            for i in hint_order[:fixed_prefix]:
                if not bounds[i][0] <= hint[i] <= bounds[i][1]:
                    raise ValueError("fixed hint cycle lies outside its domain")
                model.add(starts[i] == hint[i])
        print(
            f"domain_radius={domain_radius} fixed_hint_prefix={fixed_prefix}",
            flush=True,
        )

    escape_variables: dict[tuple[int, int], cp_model.IntVar] = {}
    required_by_right: dict[int, int] = {}
    for right in rights:
        required = max(
            0,
            len(loads) - SLOT_LIMITS["load"] * (right + 1),
        )
        required_by_right[right] = required
        variables = []
        for load in loads:
            escape = model.new_bool_var(f"late_{load}_{right}")
            escape_variables[load, right] = escape
            impossible = (
                terminal_tail[load] >= 0
                and horizon - 1 - terminal_tail[load] <= right
            )
            thresholds = []
            for target, distance in downstream_flow[load].items():
                threshold = right + distance + 1
                if threshold > bounds[target][1]:
                    impossible = True
                    break
                thresholds.append((target, threshold))
            if impossible:
                model.add(escape == 0)
            else:
                for target, threshold in thresholds:
                    model.add(starts[target] >= threshold).only_enforce_if(
                        escape
                    )
            variables.append(escape)
        if bool(int(os.environ.get("ENFORCE_CUTS", "1"))):
            model.add(sum(variables) >= required)
        print(
            f"cut=0:{right} required_late={required} "
            f"current_capacity={SLOT_LIMITS['load'] * (right + 1)}",
            flush=True,
        )

    if bool(int(os.environ.get("MAXIMIZE_ESCAPES", "1"))):
        # Smaller rights are harder and receive the larger lexicographic-like
        # weight.  The solved schedule is still checked against every Hall
        # window afterwards.
        objective = []
        for rank, right in enumerate(sorted(rights)):
            weight = len(rights) - rank
            objective.extend(
                weight * escape_variables[load, right]
                for load in loads
            )
        model.maximize(sum(objective))

    for i, start in starts.items():
        if i in hint:
            model.add_hint(
                start,
                min(bounds[i][1], max(bounds[i][0], hint[i])),
            )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(
        os.environ.get("TIME_LIMIT", "300")
    )
    solver.parameters.num_workers = int(os.environ.get("WORKERS", "8"))
    solver.parameters.random_seed = int(os.environ.get("RANDOM_SEED", "1"))
    solver.parameters.randomize_search = bool(
        int(os.environ.get("RANDOMIZE", "1"))
    )
    solver.parameters.repair_hint = bool(
        int(os.environ.get("REPAIR_HINT", "0"))
    )
    solver.parameters.log_search_progress = bool(
        int(os.environ.get("LOG", "0"))
    )
    status = solver.solve(model)
    print(f"status={solver.status_name(status)}", flush=True)
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return

    result = {i: solver.value(starts[i]) for i in flow}
    for right in rights:
        realized = sum(
            solver.value(escape_variables[load, right])
            for load in loads
        )
        print(
            f"cut=0:{right} modeled_late={realized} "
            f"required={required_by_right[right]}",
            flush=True,
        )

    # Convert the chosen start times to the canonical unary FLOW order, then
    # recompute exact full-DAG windows.  This removes any dependence on CP's
    # arbitrary idle-cycle placement.
    flow_order = sorted(flow, key=lambda i: (result[i], i))
    ordered_parents = [dict(op.parents) for op in ops]
    for previous, current in zip(flow_order, flow_order[1:]):
        ordered_parents[current][previous] = max(
            ordered_parents[current].get(previous, 0), 1
        )
    ordered_topological, ordered_children = topological_order(
        ops, ordered_parents
    )
    ordered_earliest, ordered_latest = full_windows(
        ops,
        ordered_topological,
        ordered_children,
        horizon,
        ordered_parents,
    )
    if any(
        left > right
        for left, right in zip(ordered_earliest, ordered_latest)
    ):
        raise AssertionError("solved FLOW order does not fit the horizon")
    overload, left, right, trapped = worst_hall(
        ops, ordered_earliest, ordered_latest, horizon
    )
    span = max(ordered_earliest) + 1
    usage = Counter(ordered_earliest[i] for i in flow)
    if max(usage.values(), default=0) > 1:
        raise AssertionError("FLOW capacity overflow")
    print(
        f"canonical_span={span} hall_overload={overload} "
        f"window={left}:{right} trapped={trapped}",
        flush=True,
    )

    output = Path(
        os.environ.get("OUT", "/tmp/aopt-flow-hall-exact.json")
    )
    output.write_text(
        json.dumps(
            {
                "engine": "flow",
                "horizon": horizon,
                "cycles": {
                    str(i): ordered_earliest[i] for i in flow
                },
                "order": flow_order,
                "hall_overload": overload,
                "hall_window": [left, right],
            }
        )
    )
    print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
