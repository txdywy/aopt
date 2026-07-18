"""Solve FLOW order directly against one or more LOAD Hall cuts.

For a fixed horizon and a Hall window ``[left, right]``, a load can escape on
the left iff every closest upstream FLOW operation finishes its path to the
load before ``left``.  It can escape on the right iff every closest downstream
FLOW operation starts strictly after the load-to-FLOW path distance plus
``right``.  This gives a compact exact model with only the FLOW start
variables and a few Booleans per load/cut, instead of jointly time-indexing
all FLOW and LOAD operations.
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
    engine: str = "load",
) -> tuple[int, int, int, int]:
    loads = [i for i, op in enumerate(ops) if op.engine == engine]
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
    capacity = SLOT_LIMITS[engine]
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
    order_engine = os.environ.get("ORDER_ENGINE", "flow")
    hall_engine = os.environ.get("HALL_ENGINE", "load")
    if order_engine == hall_engine:
        raise ValueError("ORDER_ENGINE and HALL_ENGINE must differ")
    cuts = [
        (0, int(value))
        for value in os.environ.get("RIGHTS", "").split(",")
        if value
    ]
    cuts.extend(
        tuple(int(component) for component in value.split(":"))
        for value in os.environ.get("CUTS", "").split(",")
        if value
    )
    if not cuts:
        cuts = [(0, 853)]
    cuts = list(dict.fromkeys(cuts))
    if any(
        len(cut) != 2 or not 0 <= cut[0] <= cut[1] < horizon
        for cut in cuts
    ):
        raise ValueError("CUTS must contain left:right cycles in the horizon")

    topological, children = topological_order(ops)
    position = {index: rank for rank, index in enumerate(topological)}
    earliest, latest = full_windows(
        ops, topological, children, horizon
    )
    if any(left > right for left, right in zip(earliest, latest)):
        raise ValueError("bare DAG does not fit the requested horizon")

    flow = [i for i in topological if ops[i].engine == order_engine]
    flow_set = set(flow)
    loads = [i for i in topological if ops[i].engine == hall_engine]
    order_capacity = SLOT_LIMITS[order_engine]
    hall_capacity = SLOT_LIMITS[hall_engine]

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

    # The forward projection above also gives the closest upstream FLOW
    # sources for every non-FLOW operation.  Separately retain the longest
    # root-to-operation path that never crosses FLOW; it is the
    # order-independent part of an operation's release time.
    upstream_flow = frontier
    root_release = [-1] * len(ops)
    for child in topological:
        if child in flow_set:
            root_release[child] = -1
            continue
        release = 0 if not ops[child].parents else -1
        for parent, lag in ops[child].parents.items():
            if parent not in flow_set and root_release[parent] >= 0:
                release = max(release, root_release[parent] + lag)
        root_release[child] = release

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
    if order_capacity == 1:
        model.add_all_different(starts.values())
    else:
        intervals = [
            model.new_fixed_size_interval_var(starts[i], 1, f"i{i}")
            for i in flow
        ]
        model.add_cumulative(
            intervals, [1] * len(intervals), order_capacity
        )
        if bool(int(os.environ.get("SATURATE_ORDER_MICROSLOTS", "0"))):
            microslots = []
            for i in flow:
                lane = model.new_int_var(
                    0, order_capacity - 1, f"lane_{i}"
                )
                microslot = model.new_int_var(
                    order_capacity * bounds[i][0],
                    order_capacity * bounds[i][1] + order_capacity - 1,
                    f"microslot_{i}",
                )
                model.add(
                    microslot == order_capacity * starts[i] + lane
                )
                microslots.append(microslot)
            for hole in range(order_capacity * horizon - len(flow)):
                microslots.append(
                    model.new_int_var(
                        0,
                        order_capacity * horizon - 1,
                        f"microslot_hole_{hole}",
                    )
                )
            model.add_all_different(microslots)
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

    # A cut's left-escape predicate depends only on ``(load, left)`` and its
    # right-escape predicate only on ``(load, right)``.  Cutting-plane runs
    # deliberately accumulate many overlapping windows, so sharing these
    # literals removes hundreds of thousands of duplicate reified path
    # constraints from the later exact models.
    early_variables: dict[
        tuple[int, int], tuple[cp_model.IntVar, int | None]
    ] = {}
    late_variables: dict[
        tuple[int, int], tuple[cp_model.IntVar, int | None]
    ] = {}
    constants = {
        value: model.new_constant(value)
        for value in (0, 1)
    }

    def early_variable(
        load: int, left: int
    ) -> tuple[cp_model.IntVar, int | None]:
        key = (load, left)
        if key in early_variables:
            return early_variables[key]
        impossible = (
            root_release[load] >= 0
            and root_release[load] >= left
        )
        thresholds = []
        for source, distance in upstream_flow[load].items():
            threshold = left - 1 - distance
            if threshold < bounds[source][0]:
                impossible = True
                break
            thresholds.append((source, threshold))
        if impossible:
            result = (constants[0], 0)
        elif not thresholds:
            result = (constants[1], 1)
        else:
            variable = model.new_bool_var(f"early_{load}_{left}")
            for source, threshold in thresholds:
                model.add(starts[source] <= threshold).only_enforce_if(
                    variable
                )
            result = (variable, None)
        early_variables[key] = result
        return result

    def late_variable(
        load: int, right: int
    ) -> tuple[cp_model.IntVar, int | None]:
        key = (load, right)
        if key in late_variables:
            return late_variables[key]
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
            result = (constants[0], 0)
        elif not thresholds:
            result = (constants[1], 1)
        else:
            variable = model.new_bool_var(f"late_{load}_{right}")
            for target, threshold in thresholds:
                model.add(starts[target] >= threshold).only_enforce_if(
                    variable
                )
            result = (variable, None)
        late_variables[key] = result
        return result

    escape_variables: dict[tuple[int, int, int], cp_model.IntVar] = {}
    required_by_cut: dict[tuple[int, int], int] = {}
    overload_allowance = int(os.environ.get("HALL_OVERLOAD_ALLOWANCE", "0"))
    margin_cap = int(os.environ.get("MAXIMIZE_MARGIN_CAP", "0"))
    margin_variables: list[cp_model.IntVar] = []
    for left, right in cuts:
        required = max(
            0,
            len(loads) - hall_capacity * (right - left + 1),
        )
        enforced_required = max(0, required - overload_allowance)
        required_by_cut[left, right] = enforced_required
        variables = []
        for load in loads:
            early, early_fixed = early_variable(load, left)
            late, late_fixed = late_variable(load, right)
            if early_fixed == 1 or late_fixed == 1:
                escape = constants[1]
            elif early_fixed == 0 and late_fixed == 0:
                escape = constants[0]
            elif early_fixed == 0:
                escape = late
            elif late_fixed == 0:
                escape = early
            else:
                escape = model.new_bool_var(
                    f"escape_{load}_{left}_{right}"
                )
                # Exact OR channeling prevents a wide-window load from
                # counting twice when it can escape on both sides.
                model.add(escape >= early)
                model.add(escape >= late)
                model.add(escape <= early + late)
            escape_variables[load, left, right] = escape
            variables.append(escape)
        if bool(int(os.environ.get("ENFORCE_CUTS", "1"))):
            model.add(sum(variables) >= enforced_required)
        if margin_cap > 0:
            # A capped margin objective is much better behaved than the raw
            # total number of escapes: already-easy cuts stop contributing
            # after ``margin_cap``, so CP-SAT spends its effort pulling every
            # near-binding Hall cut away from the feasibility boundary.
            margin = model.new_int_var(
                0, margin_cap, f"margin_{left}_{right}"
            )
            model.add(margin <= sum(variables) - enforced_required)
            margin_variables.append(margin)
        print(
            f"cut={left}:{right} required_escape={enforced_required} "
            f"zero_overload_escape={required} "
            f"current_capacity={hall_capacity * (right - left + 1)}",
            flush=True,
        )
    print(
        f"shared_predicates early={len(early_variables)} "
        f"late={len(late_variables)} escapes={len(escape_variables)}",
        flush=True,
    )

    if margin_variables:
        model.maximize(sum(margin_variables))
    elif bool(int(os.environ.get("MAXIMIZE_ESCAPES", "1"))):
        # Shorter windows are harder and receive the larger
        # lexicographic-like weight.  The solved schedule is still checked
        # against every Hall window afterwards.
        objective = []
        ordered_cuts = sorted(cuts, key=lambda cut: cut[1] - cut[0])
        for rank, (left, right) in enumerate(ordered_cuts):
            weight = len(cuts) - rank
            objective.extend(
                weight * escape_variables[load, left, right]
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
    print(
        f"status={solver.status_name(status)} "
        f"objective={solver.objective_value:g} "
        f"bound={solver.best_objective_bound:g} "
        f"wall={solver.wall_time:g}",
        flush=True,
    )
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return

    result = {i: solver.value(starts[i]) for i in flow}
    for left, right in cuts:
        realized = sum(
            solver.value(escape_variables[load, left, right])
            for load in loads
        )
        print(
            f"cut={left}:{right} modeled_escape={realized} "
            f"required={required_by_cut[left, right]}",
            flush=True,
        )

    # Keep CP-SAT's exact FLOW placement as well as its order.  The horizon
    # has a small but valuable number of FLOW holes; left-compacting those
    # holes can destroy a late escape that the exact Hall model intentionally
    # created.  Fixed FLOW cycles are legitimate resource decisions, so
    # propagate them through the remaining DAG to obtain exact LOAD windows.
    exact_earliest = [0] * len(ops)
    for child in topological:
        release = max(
            (
                exact_earliest[parent] + lag
                for parent, lag in ops[child].parents.items()
            ),
            default=0,
        )
        if child in flow_set:
            exact_earliest[child] = result[child]
            if release > result[child]:
                raise AssertionError("fixed FLOW cycle precedes its release")
        else:
            exact_earliest[child] = release
    exact_latest = [horizon - 1] * len(ops)
    for parent in reversed(topological):
        deadline = min(
            (
                exact_latest[child] - lag
                for child, lag in children[parent]
            ),
            default=horizon - 1,
        )
        if parent in flow_set:
            exact_latest[parent] = result[parent]
            if result[parent] > deadline:
                raise AssertionError("fixed FLOW cycle exceeds its deadline")
        else:
            exact_latest[parent] = deadline
    if any(
        left > right
        for left, right in zip(exact_earliest, exact_latest)
    ):
        raise AssertionError("fixed FLOW cycles make the DAG infeasible")
    exact_overload, exact_left, exact_right, exact_trapped = worst_hall(
        ops, exact_earliest, exact_latest, horizon, hall_engine
    )
    exact_span = max(exact_earliest) + 1
    print(
        f"exact_span={exact_span} hall_overload={exact_overload} "
        f"window={exact_left}:{exact_right} trapped={exact_trapped}",
        flush=True,
    )

    # Convert the chosen start times to the canonical unary FLOW order, then
    # recompute exact full-DAG windows.  This removes any dependence on CP's
    # arbitrary idle-cycle placement.
    flow_order = sorted(flow, key=lambda i: (result[i], i))
    ordered_parents = [dict(op.parents) for op in ops]
    for previous, current in zip(
        flow_order, flow_order[order_capacity:]
    ):
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
        ops, ordered_earliest, ordered_latest, horizon, hall_engine
    )
    span = max(ordered_earliest) + 1
    usage = Counter(ordered_earliest[i] for i in flow)
    if max(usage.values(), default=0) > order_capacity:
        raise AssertionError(f"{order_engine} capacity overflow")
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
                "engine": order_engine,
                "hall_engine": hall_engine,
                "horizon": horizon,
                "cycles": {
                    str(i): result[i] for i in flow
                },
                "canonical_cycles": {
                    str(i): ordered_earliest[i] for i in flow
                },
                "order": flow_order,
                "hall_overload": exact_overload,
                "hall_window": [exact_left, exact_right],
                "canonical_hall_overload": overload,
                "canonical_hall_window": [left, right],
            }
        )
    )
    print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
