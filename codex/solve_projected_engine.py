"""Solve one VLIW engine after projecting away all other instructions.

The full kernel has about twenty thousand operations, but only the operations
issued by the selected engine consume its capacity.  Every path through the
other engines is summarized as a weighted edge between consecutive selected
operations.  This keeps all true/anti dependency timing constraints while
shrinking the CP-SAT model by roughly an order of magnitude.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import heapq
import json
import os
from pathlib import Path
import random

from ortools.sat.python import cp_model

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from problem import SLOT_LIMITS


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

    # Condition this projected solve on already chosen resource orders.  A
    # capacity-k engine order is represented by unit-lag edges from item i to
    # item i+k.  Solving engines sequentially with these edges avoids the
    # dependency cycles that arise when independently optimal orders are
    # combined after the fact.
    if "ORDER_HINTS" in os.environ or "PARTIAL_ORDER_HINTS" in os.environ:
        conditioned_parents = [dict(op.parents) for op in ops]
        conditioned_engines: set[str] = set()
        for raw_path in os.environ.get("ORDER_HINTS", "").split(","):
            if not raw_path:
                continue
            path = Path(raw_path)
            payload = json.loads(path.read_text())
            fixed_engine = payload["engine"]
            if fixed_engine in conditioned_engines:
                raise ValueError(f"duplicate conditioned engine: {fixed_engine}")
            conditioned_engines.add(fixed_engine)
            fixed_cycles = {
                int(index): int(cycle)
                for index, cycle in payload["cycles"].items()
            }
            fixed_selected = {
                i for i, op in enumerate(ops) if op.engine == fixed_engine
            }
            if set(fixed_cycles) != fixed_selected:
                raise ValueError(
                    f"{path} does not cover every {fixed_engine} operation"
                )
            fixed_capacity = SLOT_LIMITS[fixed_engine]
            sequence = sorted(
                fixed_selected,
                key=lambda index: (fixed_cycles[index], index),
            )
            for previous, current in zip(sequence, sequence[fixed_capacity:]):
                if fixed_cycles[previous] >= fixed_cycles[current]:
                    raise ValueError(
                        f"{path} exceeds {fixed_engine} capacity"
                    )
                conditioned_parents[current][previous] = max(
                    conditioned_parents[current].get(previous, 0), 1
                )
        for raw_path in os.environ.get("PARTIAL_ORDER_HINTS", "").split(","):
            if not raw_path:
                continue
            path = Path(raw_path)
            payload = json.loads(path.read_text())
            fixed_engine = payload["engine"]
            fixed_cycles = {
                int(index): int(cycle)
                for index, cycle in payload["cycles"].items()
            }
            matched = {
                int(index) for index in payload.get("matched_indices", ())
            }
            drop_groups = {
                int(value)
                for value in os.environ.get(
                    "PARTIAL_ORDER_DROP_GROUPS", ""
                ).split(",")
                if value
            }
            if drop_groups:
                matched = {
                    index
                    for index in matched
                    if ops[index].group not in drop_groups
                }
            drop_group_rounds = {
                tuple(int(component) for component in value.split(":"))
                for value in os.environ.get(
                    "PARTIAL_ORDER_DROP_GROUP_ROUNDS", ""
                ).split(",")
                if value
            }
            if drop_group_rounds:
                matched = {
                    index
                    for index in matched
                    if (ops[index].group, ops[index].round)
                    not in drop_group_rounds
                }
            if not matched:
                raise ValueError(f"{path} has no matched partial order")
            if any(
                index not in fixed_cycles or ops[index].engine != fixed_engine
                for index in matched
            ):
                raise ValueError(f"{path} has invalid matched indices")
            fixed_capacity = SLOT_LIMITS[fixed_engine]
            sequence = sorted(
                matched, key=lambda index: (fixed_cycles[index], index)
            )
            for previous, current in zip(
                sequence, sequence[fixed_capacity:]
            ):
                conditioned_parents[current][previous] = max(
                    conditioned_parents[current].get(previous, 0), 1
                )
        ops = [
            replace(op, parents=conditioned_parents[i])
            for i, op in enumerate(ops)
        ]

    engine = os.environ.get("ENGINE", "load")
    horizon = int(os.environ.get("TARGET", "959"))
    capacity = SLOT_LIMITS[engine]

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
        raise ValueError("conditioned resource orders introduce a cycle")
    topological_position = {
        index: position for position, index in enumerate(topological)
    }
    selected = [i for i in topological if ops[i].engine == engine]
    selected_set = set(selected)

    earliest = [0] * len(ops)
    for child in topological:
        for parent, lag in ops[child].parents.items():
            earliest[child] = max(earliest[child], earliest[parent] + lag)
    tail = [0] * len(ops)
    for parent in reversed(topological):
        tail[parent] = max(
            (lag + tail[child] for child, lag in children[parent]),
            default=0,
        )

    # frontier[node] maps each closest selected ancestor to the longest path
    # from it to node.  A selected node resets the frontier after its incoming
    # projected edges have been recorded, so transitive selected ancestors do
    # not bloat the model.
    frontier: list[dict[int, int]] = [{} for _ in ops]
    projected: dict[int, dict[int, int]] = {}
    for child in topological:
        op = ops[child]
        incoming: dict[int, int] = {}
        for parent, lag in op.parents.items():
            sources = ({parent: 0} if parent in selected_set else frontier[parent])
            for source, distance in sources.items():
                incoming[source] = max(
                    incoming.get(source, -1), distance + lag
                )
        if child in selected_set:
            projected[child] = incoming
            frontier[child] = {child: 0}
        else:
            frontier[child] = incoming

    # Weighted transitive reduction.  The raw frontier is exact but wide DAG
    # merges can repeat the same old load as a predecessor of hundreds of
    # later loads.  Process candidate parents from newest to oldest; an edge
    # is redundant when a kept newer parent already implies at least its lag.
    reduced: dict[int, dict[int, int]] = {}
    ancestor_distance: dict[int, dict[int, int]] = {}
    for child in selected:
        kept: dict[int, int] = {}
        implied: dict[int, int] = {}
        for parent, lag in sorted(
            projected[child].items(),
            key=lambda item: topological_position[item[0]],
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

    source_cycles = None
    if "HINT" in os.environ:
        source_cycles = json.loads(Path(os.environ["HINT"]).read_text())["cycles"]
        if len(source_cycles) != len(ops):
            raise ValueError("hint does not match projected DAG")
    projected_hint = None
    if "PROJECTED_HINT" in os.environ:
        payload = json.loads(Path(os.environ["PROJECTED_HINT"]).read_text())
        projected_hint = {int(i): cycle for i, cycle in payload["cycles"].items()}
        if set(projected_hint) != selected_set:
            raise ValueError("projected hint does not match selected engine")

    projected_edge_count = sum(map(len, projected.values()))
    print(
        f"engine={engine} jobs={len(selected)} "
        f"projected_edges={projected_edge_count}",
        flush=True,
    )
    if bool(int(os.environ.get("ANALYZE_HALL", "0"))):
        latest = [horizon - 1 - value for value in tail]
        best_overload = -10**9
        best_window = (0, horizon - 1)
        for left in range(horizon):
            histogram = [0] * horizon
            for i in selected:
                if earliest[i] >= left and 0 <= latest[i] < horizon:
                    histogram[latest[i]] += 1
            contained = 0
            for right in range(left, horizon):
                contained += histogram[right]
                overload = contained - capacity * (right - left + 1)
                if overload > best_overload:
                    best_overload = overload
                    best_window = (left, right)
        left, right = best_window
        if "ANALYZE_HALL_FORCE_WINDOW" in os.environ:
            left, right = (
                int(value)
                for value in os.environ["ANALYZE_HALL_FORCE_WINDOW"].split(":")
            )
            best_overload = sum(
                earliest[i] >= left and latest[i] <= right for i in selected
            ) - capacity * (right - left + 1)
        trapped = [
            i for i in selected
            if earliest[i] >= left and latest[i] <= right
        ]
        print(
            f"hall_overload={best_overload} window={left}:{right} "
            f"jobs={len(trapped)} capacity={capacity * (right - left + 1)}",
            flush=True,
        )
        for label, counts in (
            ("tag", Counter(ops[i].tag for i in trapped)),
            ("group", Counter(ops[i].group for i in trapped)),
            ("round", Counter(ops[i].round for i in trapped)),
        ):
            print(
                f"hall_{label}s="
                + ",".join(
                    f"{value}:{count}"
                    for value, count in counts.most_common(32)
                ),
                flush=True,
            )
        trapped_set = set(trapped)
        detail_tags = frozenset(
            value
            for value in os.environ.get("ANALYZE_HALL_DETAIL_TAGS", "").split(",")
            if value
        )
        print_tags = frozenset(
            value
            for value in os.environ.get("ANALYZE_HALL_PRINT_TAGS", "").split(",")
            if value
        )
        for i in selected:
            op = ops[i]
            if op.tag in print_tags:
                print(
                    f"hall_job i={i} earliest={earliest[i]} latest={latest[i]} "
                    f"g={op.group} r={op.round} {op.tag}",
                    flush=True,
                )
        for i in trapped:
            op = ops[i]
            if op.tag in detail_tags:
                parent_text = ";".join(
                    f"{parent}:{ops[parent].tag}@{earliest[parent]}+{lag}"
                    for parent, lag in op.parents.items()
                )
                print(
                    f"hall_trapped i={i} earliest={earliest[i]} "
                    f"latest={latest[i]} g={op.group} r={op.round} {op.tag} "
                    f"parents={parent_text}",
                    flush=True,
                )
        excluded = [i for i in selected if i not in trapped_set]
        if len(excluded) <= int(os.environ.get("ANALYZE_HALL_EXCLUDED_MAX", "64")):
            for i in excluded:
                op = ops[i]
                parent_suffix = ""
                if op.tag in detail_tags:
                    parent_suffix = " parents=" + ";".join(
                        f"{parent}:{ops[parent].tag}@{earliest[parent]}+{lag}"
                        for parent, lag in op.parents.items()
                    )
                print(
                    f"hall_excluded i={i} earliest={earliest[i]} "
                    f"latest={latest[i]} g={op.group} r={op.round} {op.tag}"
                    f"{parent_suffix}",
                    flush=True,
                )
    if bool(int(os.environ.get("ANALYZE_ONLY", "0"))):
        return

    if bool(int(os.environ.get("ANALYZE_PROJECTED_HINT", "0"))):
        if projected_hint is None:
            raise ValueError("projected analysis requires PROJECTED_HINT")
        order = sorted(selected, key=lambda i: (projected_hint[i], i))
        ordered_parents = {i: dict(projected[i]) for i in selected}
        edge_kind = {
            (parent, child): "dag"
            for child, parents in projected.items()
            for parent in parents
        }
        for position in range(capacity, len(order)):
            parent = order[position - capacity]
            child = order[position]
            if projected_hint[parent] >= projected_hint[child]:
                raise AssertionError(
                    (parent, child, projected_hint[parent])
                )
            if ordered_parents[child].get(parent, 0) < 1:
                ordered_parents[child][parent] = 1
                edge_kind[parent, child] = "resource"
        ordered_earliest = {i: earliest[i] for i in selected}
        reason: dict[int, int] = {}
        for child in order:
            for parent, lag in ordered_parents[child].items():
                candidate = ordered_earliest[parent] + lag
                if candidate > ordered_earliest[child]:
                    ordered_earliest[child] = candidate
                    reason[child] = parent
        endpoint = max(
            selected,
            key=lambda i: ordered_earliest[i] + tail[i],
        )
        chain = []
        node = endpoint
        while True:
            chain.append(node)
            if node not in reason:
                break
            node = reason[node]
        chain.reverse()
        print(
            f"projected_hint_span={span if (span := max(projected_hint.values()) + 1) else 0} "
            f"ordered_lb={ordered_earliest[endpoint] + tail[endpoint] + 1} "
            f"endpoint={endpoint} chain={len(chain)}",
            flush=True,
        )
        for index in chain[-int(os.environ.get("ANALYZE_TAIL", "120")):]:
            op = ops[index]
            parent = reason.get(index, -1)
            print(
                f"i={index:5d} c={projected_hint[index]:3d} "
                f"e={ordered_earliest[index]:3d} tail={tail[index]:3d} "
                f"via={edge_kind.get((parent, index), 'root'):8s} "
                f"g={op.group:2d} r={op.round:2d} {op.tag}",
                flush=True,
            )
        return

    if bool(int(os.environ.get("BUBBLE_DELETE_REPAIR", "0"))):
        if projected_hint is None:
            raise ValueError("bubble delete repair requires PROJECTED_HINT")
        current = dict(projected_hint)

        def bubble_span(cycles: dict[int, int]) -> int:
            return max(
                max(earliest),
                max(cycles[i] + tail[i] for i in selected),
            ) + 1

        def bubble_check(cycles: dict[int, int], span: int) -> None:
            usage = Counter(cycles.values())
            if any(
                cycle < 0 or count > capacity
                for cycle, count in usage.items()
            ):
                raise AssertionError("invalid bubble engine usage")
            for i in selected:
                if cycles[i] < earliest[i] or cycles[i] + tail[i] >= span:
                    raise AssertionError(
                        (i, cycles[i], earliest[i], tail[i], span)
                    )
                for parent, lag in projected[i].items():
                    if cycles[i] < cycles[parent] + lag:
                        raise AssertionError(
                            (parent, i, lag, cycles[parent], cycles[i])
                        )

        current_span = bubble_span(current)
        bubble_check(current, current_span)
        output = Path(
            os.environ.get(
                "OUT", f"/tmp/aopt-{engine}-projected-bubble-repair.json"
            )
        )
        target_span = horizon
        per_candidate_limit = float(
            os.environ.get("BUBBLE_TIME_LIMIT", "30")
        )
        movement = int(os.environ.get("BUBBLE_EXTRA_MOVEMENT", "0"))
        requested_cycles = tuple(
            int(value)
            for value in os.environ.get("BUBBLE_DELETE_CYCLES", "").split(",")
            if value
        )
        print(
            f"bubble_start={current_span} target={target_span} "
            f"extra_movement={movement}",
            flush=True,
        )

        while current_span > target_span:
            usage = Counter(current.values())
            if requested_cycles:
                candidates = tuple(
                    cycle for cycle in requested_cycles
                    if 0 < cycle < current_span
                )
            else:
                # Prefer genuine holes, then half-full cycles.  Distance from
                # the middle only breaks ties and keeps both setup and
                # deadline bubbles in play.
                candidates = tuple(
                    cycle
                    for _, _, cycle in sorted(
                        (
                            usage[cycle],
                            abs(2 * cycle - current_span),
                            cycle,
                        )
                        for cycle in range(1, current_span)
                    )
                )
            candidate_limit = int(
                os.environ.get("BUBBLE_CANDIDATES", "64")
            )
            repaired = None
            for attempt, deleted in enumerate(candidates[:candidate_limit], 1):
                new_span = current_span - 1
                model = cp_model.CpModel()
                starts = {}
                impossible = False
                for i in selected:
                    old = current[i]
                    if old < deleted:
                        lower = upper = old
                    else:
                        base = old - 1
                        lower = max(earliest[i], base - movement)
                        upper = min(
                            new_span - 1 - tail[i],
                            old + movement,
                        )
                    if lower > upper:
                        impossible = True
                        break
                    starts[i] = model.new_int_var(lower, upper, f"s{i}")
                if impossible:
                    continue
                for child, parents in projected.items():
                    for parent, lag in parents.items():
                        model.add(starts[child] >= starts[parent] + lag)
                intervals = [
                    model.new_fixed_size_interval_var(starts[i], 1, f"i{i}")
                    for i in selected
                ]
                model.add_cumulative(intervals, [1] * len(intervals), capacity)
                for i, start in starts.items():
                    old = current[i]
                    preferred = old if old < deleted else old - 1
                    model.add_hint(start, preferred)
                solver = cp_model.CpSolver()
                solver.parameters.max_time_in_seconds = per_candidate_limit
                solver.parameters.num_workers = int(
                    os.environ.get("WORKERS", "8")
                )
                solver.parameters.random_seed = (
                    int(os.environ.get("RANDOM_SEED", "1"))
                    + deleted
                    + current_span
                )
                solver.parameters.randomize_search = bool(
                    int(os.environ.get("RANDOMIZE", "1"))
                )
                status = solver.solve(model)
                print(
                    f"bubble_try span={current_span} delete={deleted} "
                    f"attempt={attempt} status={solver.status_name(status)}",
                    flush=True,
                )
                if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
                    continue
                repaired = {
                    i: solver.value(starts[i]) for i in selected
                }
                bubble_check(repaired, new_span)
                print(
                    f"bubble_success={current_span}->{new_span} "
                    f"cycle={deleted} attempt={attempt}",
                    flush=True,
                )
                break
            if repaired is None:
                print(f"bubble_stuck={current_span}", flush=True)
                break
            current = repaired
            current_span -= 1
            output.write_text(
                json.dumps(
                    {
                        "engine": engine,
                        "horizon": current_span,
                        "cycles": {
                            str(i): cycle for i, cycle in current.items()
                        },
                    }
                )
            )
        print(f"bubble_final={current_span} output={output}", flush=True)
        return

    if bool(int(os.environ.get("DELETE_REPAIR", "0"))):
        if projected_hint is None:
            raise ValueError("delete repair requires PROJECTED_HINT")
        current = dict(projected_hint)
        repair_target = horizon
        repair_radii = tuple(
            int(value)
            for value in os.environ.get(
                "DELETE_RADII", "24,40,64,96,144"
            ).split(",")
            if value
        )
        candidates_per_radius = int(
            os.environ.get("DELETE_CANDIDATES", "24")
        )
        per_candidate_limit = float(
            os.environ.get("DELETE_TIME_LIMIT", "4")
        )

        def span_of(cycles: dict[int, int]) -> int:
            return max(
                max(earliest),
                max(cycles[i] + tail[i] for i in selected),
            ) + 1

        def check(cycles: dict[int, int], span: int) -> None:
            usage = Counter(cycles.values())
            if any(
                cycle < 0 or count > capacity
                for cycle, count in usage.items()
            ):
                raise AssertionError("invalid repaired engine usage")
            for i in selected:
                if cycles[i] < earliest[i] or cycles[i] + tail[i] >= span:
                    raise AssertionError(
                        (i, cycles[i], earliest[i], tail[i], span)
                    )
                for parent, lag in projected[i].items():
                    if cycles[i] < cycles[parent] + lag:
                        raise AssertionError(
                            (parent, i, lag, cycles[parent], cycles[i])
                        )

        current_span = span_of(current)
        check(current, current_span)
        print(
            f"delete_start={current_span} target={repair_target}",
            flush=True,
        )

        def ranked_deletions(
            cycles: dict[int, int], span: int, radius: int
        ) -> list[int]:
            usage = Counter(cycles.values())
            prefix = [0] * (span + 1)
            for cycle in range(span):
                prefix[cycle + 1] = prefix[cycle] + usage[cycle]
            lower = max(1, int(os.environ.get("DELETE_MIN", "1")))
            upper = min(
                span - 1,
                int(os.environ.get("DELETE_MAX", str(span - 1))),
            )
            ranked: list[tuple[int, int, int]] = []
            for deleted in range(lower, upper):
                new_span = span - 1
                lo = max(0, deleted - radius)
                hi = min(new_span - 1, deleted + radius)
                # The compressed neighborhood receives old cycles through
                # hi+1.  Reject windows whose aggregate work cannot fit even
                # before considering precedences.
                old_lo = lo
                old_hi = min(span - 1, hi + 1)
                jobs = prefix[old_hi + 1] - prefix[old_lo]
                room = capacity * (hi - lo + 1) - jobs
                if room < 0:
                    continue
                ranked.append((usage[deleted], -room, deleted))
            return [
                deleted
                for _, _, deleted in sorted(ranked)[:candidates_per_radius]
            ]

        def delete_one(
            cycles: dict[int, int], span: int, deleted: int, radius: int
        ) -> dict[int, int] | None:
            new_span = span - 1
            lo = max(0, deleted - radius)
            hi = min(new_span - 1, deleted + radius)
            base = {
                i: cycle if cycle < deleted else cycle - 1
                for i, cycle in cycles.items()
            }
            local = [i for i in selected if lo <= base[i] <= hi]
            local_set = set(local)
            if any(
                base[i] < earliest[i]
                or base[i] + tail[i] >= new_span
                for i in selected
                if i not in local_set
            ):
                return None

            movement_radius = int(
                os.environ.get("DELETE_MOVEMENT_RADIUS", "0")
            )
            model = cp_model.CpModel()
            bounds = {
                i: (
                    max(
                        lo,
                        earliest[i],
                        base[i] - movement_radius
                        if movement_radius
                        else lo,
                    ),
                    min(
                        hi,
                        new_span - 1 - tail[i],
                        base[i] + movement_radius
                        if movement_radius
                        else hi,
                    ),
                )
                for i in local
            }
            if any(lower > upper for lower, upper in bounds.values()):
                return None
            starts = {
                i: model.new_int_var(lower, upper, f"s{i}")
                for i, (lower, upper) in bounds.items()
            }
            for child, parents in projected.items():
                child_local = child in local_set
                for parent, lag in parents.items():
                    parent_local = parent in local_set
                    if child_local and parent_local:
                        model.add(starts[child] >= starts[parent] + lag)
                    elif child_local:
                        model.add(starts[child] >= base[parent] + lag)
                    elif parent_local:
                        model.add(base[child] >= starts[parent] + lag)
                    elif base[child] < base[parent] + lag:
                        return None

            fixed_usage = Counter(
                base[i]
                for i in selected
                if i not in local_set and lo <= base[i] <= hi
            )
            if any(count > capacity for count in fixed_usage.values()):
                return None
            intervals = [
                model.new_fixed_size_interval_var(starts[i], 1, f"i{i}")
                for i in local
            ]
            demands = [1] * len(intervals)
            for cycle, demand in fixed_usage.items():
                intervals.append(
                    model.new_fixed_size_interval_var(
                        cycle, 1, f"fixed_{cycle}"
                    )
                )
                demands.append(demand)
            model.add_cumulative(intervals, demands, capacity)
            for i, start in starts.items():
                lower, upper = bounds[i]
                model.add_hint(start, min(upper, max(lower, base[i])))

            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = per_candidate_limit
            solver.parameters.num_workers = int(
                os.environ.get("WORKERS", "8")
            )
            solver.parameters.random_seed = (
                int(os.environ.get("RANDOM_SEED", "1"))
                + deleted
                + radius
            )
            solver.parameters.randomize_search = bool(
                int(os.environ.get("RANDOMIZE", "1"))
            )
            status = solver.solve(model)
            if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
                return None
            result = dict(base)
            for i, start in starts.items():
                result[i] = solver.value(start)
            check(result, new_span)
            return result

        output = Path(
            os.environ.get(
                "OUT", f"/tmp/aopt-{engine}-projected-delete-repair.json"
            )
        )
        while current_span > repair_target:
            repaired = None
            attempts = 0
            for radius in repair_radii:
                candidates = ranked_deletions(current, current_span, radius)
                print(
                    f"delete_span={current_span} radius={radius} "
                    f"candidates={candidates}",
                    flush=True,
                )
                for deleted in candidates:
                    attempts += 1
                    repaired = delete_one(
                        current, current_span, deleted, radius
                    )
                    if repaired is not None:
                        print(
                            f"delete_success={current_span}->{current_span - 1} "
                            f"cycle={deleted} radius={radius} attempts={attempts}",
                            flush=True,
                        )
                        break
                if repaired is not None:
                    break
            if repaired is None:
                print(
                    f"delete_stuck={current_span} attempts={attempts}",
                    flush=True,
                )
                break
            current = repaired
            current_span -= 1
            output.write_text(
                json.dumps(
                    {
                        "engine": engine,
                        "horizon": current_span,
                        "cycles": {
                            str(i): cycle for i, cycle in current.items()
                        },
                    }
                )
            )
        print(f"delete_final={current_span} output={output}", flush=True)
        return

    if "REPAIR_CUTOFF" in os.environ:
        if projected_hint is None:
            raise ValueError("projected repair requires PROJECTED_HINT")
        repair_cutoff = int(os.environ["REPAIR_CUTOFF"])
        hint_shift = int(os.environ.get("HINT_SHIFT", "0"))
        incumbent = {
            i: projected_hint[i] - hint_shift for i in selected
        }
        local_set = {
            i for i in selected if incumbent[i] < repair_cutoff
        }
        fixed_set = selected_set - local_set
        fixed_usage = Counter(incumbent[i] for i in fixed_set)
        if any(
            cycle < 0 or cycle >= horizon or count > capacity
            for cycle, count in fixed_usage.items()
        ):
            raise ValueError("fixed projected repair schedule is out of range")

        repair_model = cp_model.CpModel()
        bounds: dict[int, list[int]] = {
            i: [earliest[i], horizon - 1 - tail[i]] for i in local_set
        }
        for child, parents in projected.items():
            for parent, lag in parents.items():
                if child in local_set and parent in fixed_set:
                    bounds[child][0] = max(
                        bounds[child][0], incumbent[parent] + lag
                    )
                elif child in fixed_set and parent in local_set:
                    bounds[parent][1] = min(
                        bounds[parent][1], incumbent[child] - lag
                    )
                elif child in fixed_set and parent in fixed_set:
                    if incumbent[child] < incumbent[parent] + lag:
                        raise ValueError("fixed projected edge is invalid")
        if any(lower > upper for lower, upper in bounds.values()):
            print("repair_status=INFEASIBLE_BOUNDARY", flush=True)
            return
        repair_starts = {
            i: repair_model.new_int_var(lower, upper, f"s{i}")
            for i, (lower, upper) in bounds.items()
        }
        for child, parents in projected.items():
            if child not in local_set:
                continue
            for parent, lag in parents.items():
                if parent in local_set:
                    repair_model.add(
                        repair_starts[child] >= repair_starts[parent] + lag
                    )
        repair_intervals = [
            repair_model.new_fixed_size_interval_var(
                repair_starts[i], 1, f"i{i}"
            )
            for i in local_set
        ]
        repair_demands = [1] * len(repair_intervals)
        for cycle, demand in fixed_usage.items():
            repair_intervals.append(
                repair_model.new_fixed_size_interval_var(
                    cycle, 1, f"fixed_{cycle}"
                )
            )
            repair_demands.append(demand)
        repair_model.add_cumulative(
            repair_intervals, repair_demands, capacity
        )
        for i, start in repair_starts.items():
            lower, upper = bounds[i]
            repair_model.add_hint(
                start, min(upper, max(lower, incumbent[i]))
            )
        repair_solver = cp_model.CpSolver()
        repair_solver.parameters.max_time_in_seconds = float(
            os.environ.get("TIME_LIMIT", "60")
        )
        repair_solver.parameters.num_workers = int(os.environ.get("WORKERS", "8"))
        repair_status = repair_solver.solve(repair_model)
        print(
            f"repair_jobs={len(local_set)} cutoff={repair_cutoff} "
            f"status={repair_solver.status_name(repair_status)}",
            flush=True,
        )
        if repair_status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            return
        result = incumbent.copy()
        for i, start in repair_starts.items():
            result[i] = repair_solver.value(start)
        output = Path(
            os.environ.get("OUT", f"/tmp/aopt-{engine}-projected-repair.json")
        )
        output.write_text(
            json.dumps(
                {
                    "engine": engine,
                    "horizon": horizon,
                    "cycles": {str(i): cycle for i, cycle in result.items()},
                }
            )
        )
        print(f"output={output}", flush=True)
        return

    greedy_trials = int(os.environ.get("GREEDY_TRIALS", "0"))
    if greedy_trials:
        projected_children: dict[int, list[tuple[int, int]]] = {
            i: [] for i in selected
        }
        for child, parents in projected.items():
            for parent, lag in parents.items():
                projected_children[parent].append((child, lag))
        projected_height: dict[int, int] = {}
        projected_reach: dict[int, int] = {}
        for parent in reversed(selected):
            projected_height[parent] = max(
                (
                    lag + projected_height[child]
                    for child, lag in projected_children[parent]
                ),
                default=0,
            )
            projected_reach[parent] = min(
                1_000_000,
                sum(
                    1 + projected_reach[child]
                    for child, _ in projected_children[parent]
                ),
            )

        direction = os.environ.get("GREEDY_DIRECTION", "forward")
        backward_pair_bias = {
            (int(group), int(round_index)): int(bias)
            for group, round_index, bias in (
                item.split(":")
                for item in os.environ.get(
                    "BACKWARD_PAIR_BIASES", ""
                ).split(",")
                if item
            )
        }

        def greedy_forward(seed: int) -> tuple[int, dict[int, int]]:
            rng = random.Random(seed)
            # Random linear extensions around four useful compiler signals:
            # end-to-end deadline, selected-engine critical height, fanout,
            # and transitive selected-engine work unlocked.
            tail_weight = rng.choice((2, 4, 8, 16, 32, 64))
            height_weight = rng.choice((0, 1, 2, 4, 8, 16))
            reach_divisor = rng.choice((8, 16, 32, 64, 128, 256, 512))
            fanout_weight = rng.choice((0, 1, 2, 4, 8, 16))
            group_bias = rng.choice((-8, -4, -2, -1, 0, 1, 2, 4, 8))
            noise_amplitude = rng.choice((0, 1, 2, 4, 8, 16, 32))
            noise = {
                i: rng.randrange(-noise_amplitude, noise_amplitude + 1)
                for i in selected
            }

            indegree = {i: len(projected[i]) for i in selected}
            ready_at = {i: earliest[i] for i in selected}
            future: dict[int, list[int]] = {}
            for i in selected:
                if not indegree[i]:
                    future.setdefault(ready_at[i], []).append(i)
            ready: list[tuple[tuple[int, ...], int]] = []

            def push(i: int) -> None:
                op = ops[i]
                priority = (
                    tail_weight * tail[i]
                    + height_weight * projected_height[i]
                    + projected_reach[i] // reach_divisor
                    + fanout_weight * len(projected_children[i])
                    + group_bias * (op.group if op.group is not None else -1)
                    + noise[i],
                    tail[i],
                    projected_height[i],
                    projected_reach[i],
                    -i,
                )
                heapq.heappush(ready, (tuple(-x for x in priority), i))

            cycles: dict[int, int] = {}
            cycle = 0
            while len(cycles) < len(selected):
                for available in sorted(k for k in future if k <= cycle):
                    for i in future.pop(available):
                        push(i)
                used = 0
                while ready and used < capacity:
                    _, parent = heapq.heappop(ready)
                    cycles[parent] = cycle
                    used += 1
                    for child, lag in projected_children[parent]:
                        indegree[child] -= 1
                        ready_at[child] = max(ready_at[child], cycle + lag)
                        if not indegree[child]:
                            if ready_at[child] <= cycle:
                                push(child)
                            else:
                                future.setdefault(ready_at[child], []).append(child)
                if used:
                    cycle += 1
                elif future:
                    cycle = min(future)
                else:
                    raise AssertionError("projected graph is cyclic")
            score = max(
                max(earliest),
                max(cycles[i] + tail[i] for i in selected),
            ) + 1
            return score, cycles

        def greedy_backward(seed: int) -> tuple[int, dict[int, int]]:
            rng = random.Random(seed)
            early_weight = rng.choice((2, 4, 8, 16, 32, 64))
            ancestor_divisor = rng.choice((8, 16, 32, 64, 128, 256))
            fanin_weight = rng.choice((0, 1, 2, 4, 8, 16))
            group_bias = rng.choice((-8, -4, -2, -1, 0, 1, 2, 4, 8))
            noise_amplitude = rng.choice((0, 1, 2, 4, 8, 16, 32))
            dynamic_unlock = bool(int(os.environ.get("DYNAMIC_UNLOCK", "0")))
            unlock_weight = rng.choice((16, 32, 64, 128, 256, 512, 1024))
            noise = {
                i: rng.randrange(-noise_amplitude, noise_amplitude + 1)
                for i in selected
            }
            successor_count = {
                i: len(projected_children[i]) for i in selected
            }
            latest_at = {i: horizon - 1 - tail[i] for i in selected}
            ancestor_count = {
                i: len(ancestor_distance[i]) for i in selected
            }
            future: dict[int, list[int]] = {}
            for i in selected:
                if not successor_count[i]:
                    future.setdefault(latest_at[i], []).append(i)
            ready: list[tuple[tuple[int, ...], int]] = []
            ready_set: set[int] = set()

            def priority_for(i: int) -> tuple[int, ...]:
                op = ops[i]
                unlock_score = sum(
                    1 + ancestor_count[parent] // ancestor_divisor
                    for parent in projected[i]
                    if successor_count[parent] == 1
                )
                priority = (
                    early_weight * earliest[i]
                    + ancestor_count[i] // ancestor_divisor
                    + fanin_weight * len(projected[i])
                    + unlock_weight * unlock_score
                    + group_bias * (op.group if op.group is not None else -1)
                    + backward_pair_bias.get((op.group, op.round), 0)
                    + noise[i],
                    earliest[i],
                    ancestor_count[i],
                    len(projected[i]),
                    i,
                )
                return priority

            def push(i: int) -> None:
                if dynamic_unlock:
                    ready_set.add(i)
                    return
                priority = priority_for(i)
                heapq.heappush(ready, (tuple(-x for x in priority), i))

            cycles: dict[int, int] = {}
            cycle = horizon - 1
            while len(cycles) < len(selected):
                for available in sorted(
                    (k for k in future if k >= cycle), reverse=True
                ):
                    for i in future.pop(available):
                        push(i)
                used = 0
                while (ready_set if dynamic_unlock else ready) and used < capacity:
                    if dynamic_unlock:
                        child = max(ready_set, key=priority_for)
                        ready_set.remove(child)
                    else:
                        _, child = heapq.heappop(ready)
                    cycles[child] = cycle
                    used += 1
                    for parent, lag in projected[child].items():
                        successor_count[parent] -= 1
                        latest_at[parent] = min(
                            latest_at[parent], cycle - lag
                        )
                        if not successor_count[parent]:
                            if latest_at[parent] >= cycle:
                                push(parent)
                            else:
                                future.setdefault(latest_at[parent], []).append(parent)
                if used:
                    cycle -= 1
                elif future:
                    cycle = max(future)
                else:
                    raise AssertionError("projected graph is cyclic")
            # If a task crossed its source-side release, shifting the entire
            # backward schedule right by the maximum violation repairs all
            # releases and increases the required horizon by the same amount.
            violation = max(
                (earliest[i] - cycles[i] for i in selected), default=0
            )
            shift = max(0, violation)
            if shift:
                cycles = {i: value + shift for i, value in cycles.items()}
            return horizon + shift, cycles

        best: tuple[int, int, dict[int, int]] | None = None
        save_max = int(os.environ.get("SAVE_MAX", "-1"))
        save_prefix = os.environ.get("SAVE_PREFIX", "")
        accepted = 0
        for seed in range(
            int(os.environ.get("GREEDY_START", "0")),
            int(os.environ.get("GREEDY_START", "0")) + greedy_trials,
        ):
            score, cycles = (
                greedy_backward(seed)
                if direction == "backward"
                else greedy_forward(seed)
            )
            candidate = (score, seed, cycles)
            if save_prefix and score <= save_max:
                Path(f"{save_prefix}-{seed}.json").write_text(
                    json.dumps(
                        {
                            "engine": engine,
                            "horizon": score,
                            "seed": seed,
                            "cycles": {
                                str(i): cycle for i, cycle in cycles.items()
                            },
                        }
                    )
                )
                accepted += 1
            if best is None or candidate[:2] < best[:2]:
                best = candidate
                print(f"greedy score={score} seed={seed}", flush=True)
        assert best is not None
        output = Path(
            os.environ.get("OUT", f"/tmp/aopt-{engine}-projected-greedy.json")
        )
        output.write_text(
            json.dumps(
                {
                    "engine": engine,
                    "horizon": best[0],
                    "seed": best[1],
                    "cycles": {str(i): cycle for i, cycle in best[2].items()},
                }
            )
        )
        print(
            f"greedy_best={best[0]} seed={best[1]} accepted={accepted} "
            f"output={output}"
        )
        if direction == "backward":
            shift = best[0] - horizon
            raw_usage = Counter(cycle - shift for cycle in best[2].values())
            holes = [
                (cycle, capacity - raw_usage[cycle])
                for cycle in range(min(raw_usage), horizon)
                if raw_usage[cycle] < capacity
            ]
            print(
                f"backward_raw_start={min(raw_usage)} "
                f"idle_slots={sum(value for _, value in holes)} "
                f"holes={holes}"
            )
            if bool(int(os.environ.get("PRINT_HOLE_CONTEXT", "0"))):
                by_raw_cycle: dict[int, list[int]] = {}
                for i, scheduled_cycle in best[2].items():
                    by_raw_cycle.setdefault(scheduled_cycle - shift, []).append(i)
                for cycle in range(25, 70):
                    descriptions = [
                        f"i={i} g={ops[i].group} r={ops[i].round} {ops[i].tag}"
                        for i in by_raw_cycle.get(cycle, ())
                    ]
                    print(f"raw_cycle={cycle:3d} " + " | ".join(descriptions))
            critical = sorted(
                selected,
                key=lambda i: (
                    earliest[i] - (best[2][i] - shift),
                    earliest[i],
                    -i,
                ),
                reverse=True,
            )[:20]
            for i in critical:
                op = ops[i]
                print(
                    f"release_violation="
                    f"{earliest[i] - (best[2][i] - shift):3d} "
                    f"cycle={best[2][i]:3d} earliest={earliest[i]:3d} "
                    f"i={i:5d} g={op.group:2d} r={op.round:2d} {op.tag}"
                )
        return

    model = cp_model.CpModel()
    starts = {
        i: model.new_int_var(
            earliest[i], horizon - 1 - tail[i], f"s{i}"
        )
        for i in selected
    }
    for child in selected:
        for parent, lag in projected[child].items():
            model.add(starts[child] >= starts[parent] + lag)

    intervals = [
        model.new_fixed_size_interval_var(starts[i], 1, f"i{i}")
        for i in selected
    ]
    model.add_cumulative(intervals, [1] * len(intervals), capacity)
    if projected_hint is not None:
        projected_hint_shift = int(
            os.environ.get("PROJECTED_HINT_SHIFT", "0")
        )
        for i in selected:
            model.add_hint(
                starts[i],
                min(
                    horizon - 1 - tail[i],
                    max(earliest[i], projected_hint[i] - projected_hint_shift),
                ),
            )
    elif source_cycles is not None:
        for i in selected:
            model.add_hint(
                starts[i],
                min(horizon - 1 - tail[i], max(earliest[i], source_cycles[i])),
            )

    makespan = None
    if bool(int(os.environ.get("OPTIMIZE", "0"))):
        makespan = model.new_int_var(0, horizon - 1, "makespan")
        for i in selected:
            model.add(makespan >= starts[i] + tail[i])
        model.minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(os.environ.get("TIME_LIMIT", "60"))
    solver.parameters.num_workers = int(os.environ.get("WORKERS", "8"))
    solver.parameters.random_seed = int(os.environ.get("RANDOM_SEED", "1"))
    solver.parameters.randomize_search = bool(int(os.environ.get("RANDOMIZE", "0")))
    solver.parameters.log_search_progress = bool(int(os.environ.get("LOG", "0")))
    validation_error = model.validate()
    if validation_error:
        print(f"model_validation_error={validation_error}", flush=True)
    status = solver.solve(model)
    print(
        f"engine={engine} jobs={len(selected)} projected_edges="
        f"{projected_edge_count} floor="
        f"{(len(selected) + capacity - 1) // capacity} "
        f"status={solver.status_name(status)}",
        flush=True,
    )
    if makespan is not None:
        print(
            f"objective={solver.objective_value + 1} "
            f"best_bound={solver.best_objective_bound + 1}",
            flush=True,
        )
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return
    result = {str(i): solver.value(starts[i]) for i in selected}
    usage = Counter(result.values())
    if max(usage.values(), default=0) > capacity:
        raise AssertionError("projected engine capacity overflow")
    output = Path(os.environ.get("OUT", f"/tmp/aopt-{engine}-projected.json"))
    output.write_text(
        json.dumps(
            {
                "engine": engine,
                "horizon": horizon,
                "cycles": result,
            }
        )
    )
    print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
