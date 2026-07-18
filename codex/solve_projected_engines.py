"""Jointly schedule several VLIW engines after projecting away the rest.

Independent per-engine optima can encode mutually cyclic resource orders.  A
joint projection retains every path between operations on any selected engine
and lets CP-SAT choose compatible orders under all selected capacities at
once.  This is especially useful for the saturated flow/load pair while still
being much smaller than the complete roughly 20k-operation RCPSP.
"""

from __future__ import annotations

from collections import Counter
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

    engines = tuple(
        value for value in os.environ.get("ENGINES", "flow,load").split(",")
        if value
    )
    if not engines or len(set(engines)) != len(engines):
        raise ValueError("ENGINES must contain distinct engine names")
    horizon = int(os.environ.get("TARGET", "959"))

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

    selected = [i for i in topological if ops[i].engine in engines]
    selected_set = set(selected)
    frontier: list[dict[int, int]] = [{} for _ in ops]
    projected: dict[int, dict[int, int]] = {}
    for child in topological:
        incoming: dict[int, int] = {}
        for parent, lag in ops[child].parents.items():
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

    # Weighted transitive reduction in topological order.
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
    projected = reduced

    counts = Counter(ops[i].engine for i in selected)
    edge_count = sum(map(len, projected.values()))
    print(
        f"engines={engines} jobs={len(selected)} counts={dict(counts)} "
        f"projected_edges={edge_count}",
        flush=True,
    )

    greedy_trials = int(os.environ.get("GREEDY_TRIALS", "0"))
    if greedy_trials:
        projected_children: dict[int, list[tuple[int, int]]] = {
            i: [] for i in selected
        }
        for child, child_parents in projected.items():
            for parent, lag in child_parents.items():
                projected_children[parent].append((child, lag))
        ancestor_count = {
            i: len(ancestor_distance[i]) for i in selected
        }
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

        def greedy_forward(seed: int) -> tuple[int, dict[int, int]]:
            rng = random.Random(seed)
            tail_weight = rng.choice((2, 4, 8, 16, 32, 64, 128))
            height_weight = rng.choice((0, 1, 2, 4, 8, 16, 32))
            reach_divisor = rng.choice((8, 16, 32, 64, 128, 256, 512))
            fanout_weight = rng.choice((0, 1, 2, 4, 8, 16, 32))
            unlock_weight = rng.choice((0, 8, 16, 32, 64, 128, 256, 512))
            group_weight = rng.choice((-8, -4, -2, -1, 0, 1, 2, 4, 8))
            noise_amplitude = rng.choice((0, 1, 2, 4, 8, 16, 32))
            noise = {
                i: rng.randrange(-noise_amplitude, noise_amplitude + 1)
                for i in selected
            }
            engine_order = list(engines)
            rng.shuffle(engine_order)
            predecessor_count = {i: len(projected[i]) for i in selected}
            ready_at = {i: earliest[i] for i in selected}
            future: dict[int, list[int]] = {}
            ready: dict[str, list[tuple[tuple[int, ...], int]]] = {
                engine: [] for engine in engines
            }

            def priority(i: int) -> tuple[int, ...]:
                op = ops[i]
                unlock = sum(
                    1 + projected_reach[child] // reach_divisor
                    for child, _ in projected_children[i]
                    if predecessor_count[child] == 1
                )
                group = op.group if op.group is not None else -1
                return (
                    tail_weight * tail[i]
                    + height_weight * projected_height[i]
                    + projected_reach[i] // reach_divisor
                    + fanout_weight * len(projected_children[i])
                    + unlock_weight * unlock
                    + group_weight * group
                    + noise[i],
                    tail[i],
                    projected_height[i],
                    projected_reach[i],
                    -i,
                )

            def push(i: int) -> None:
                heapq.heappush(
                    ready[ops[i].engine],
                    (tuple(-value for value in priority(i)), i),
                )

            for i in selected:
                if not predecessor_count[i]:
                    future.setdefault(ready_at[i], []).append(i)
            cycles: dict[int, int] = {}
            cycle = 0
            while len(cycles) < len(selected):
                for available in sorted(value for value in future if value <= cycle):
                    for i in future.pop(available):
                        push(i)
                used = {engine: 0 for engine in engines}
                made_progress = True
                while made_progress:
                    made_progress = False
                    for engine in engine_order:
                        heap = ready[engine]
                        while heap and used[engine] < SLOT_LIMITS[engine]:
                            _, parent = heapq.heappop(heap)
                            cycles[parent] = cycle
                            used[engine] += 1
                            made_progress = True
                            for child, lag in projected_children[parent]:
                                predecessor_count[child] -= 1
                                ready_at[child] = max(
                                    ready_at[child], cycle + lag
                                )
                                if not predecessor_count[child]:
                                    if ready_at[child] <= cycle:
                                        push(child)
                                    else:
                                        future.setdefault(
                                            ready_at[child], []
                                        ).append(child)
                if any(used.values()):
                    cycle += 1
                elif future:
                    cycle = min(future)
                else:
                    raise AssertionError("projected graph is cyclic")
            score = max(cycles[i] + tail[i] for i in selected) + 1
            return score, cycles

        def greedy_backward(seed: int) -> tuple[int, dict[int, int]]:
            rng = random.Random(seed)
            early_weight = rng.choice((2, 4, 8, 16, 32, 64, 128))
            ancestor_divisor = rng.choice((8, 16, 32, 64, 128, 256, 512))
            unlock_reach_divisor = rng.choice(
                (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
            )
            fanin_weight = rng.choice((0, 1, 2, 4, 8, 16, 32))
            unlock_weight = rng.choice((0, 16, 32, 64, 128, 256, 512, 1024))
            group_weight = rng.choice((-8, -4, -2, -1, 0, 1, 2, 4, 8))
            tail_weight = rng.choice((0, 1, 2, 4, 8, 16))
            noise_amplitude = rng.choice((0, 1, 2, 4, 8, 16, 32))
            noise = {
                i: rng.randrange(-noise_amplitude, noise_amplitude + 1)
                for i in selected
            }
            engine_order = list(engines)
            rng.shuffle(engine_order)
            successor_count = {
                i: len(projected_children[i]) for i in selected
            }
            latest_at = {i: horizon - 1 - tail[i] for i in selected}
            future: dict[int, list[int]] = {}
            ready: dict[str, list[tuple[tuple[int, ...], int]]] = {
                engine: [] for engine in engines
            }

            def priority(i: int) -> tuple[int, ...]:
                op = ops[i]
                # In a backwards schedule the valuable action is often to
                # schedule the final remaining child of a high-fanout root,
                # making that root ready while there is still room near the
                # start of the horizon.  Ancestor count alone gets this
                # exactly backwards for constants and input loads: they have
                # few ancestors but gate a very large downstream cone.
                unlock = sum(
                    1
                    + ancestor_count[parent] // ancestor_divisor
                    + projected_reach[parent] // unlock_reach_divisor
                    for parent in projected[i]
                    if successor_count[parent] == 1
                )
                group = op.group if op.group is not None else -1
                return (
                    early_weight * earliest[i]
                    + ancestor_count[i] // ancestor_divisor
                    + fanin_weight * len(projected[i])
                    + unlock_weight * unlock
                    + group_weight * group
                    + tail_weight * tail[i]
                    + noise[i],
                    earliest[i],
                    ancestor_count[i],
                    len(projected[i]),
                    i,
                )

            def push(i: int) -> None:
                heapq.heappush(
                    ready[ops[i].engine],
                    (tuple(-value for value in priority(i)), i),
                )

            for i in selected:
                if not successor_count[i]:
                    future.setdefault(latest_at[i], []).append(i)

            cycles: dict[int, int] = {}
            cycle = horizon - 1
            while len(cycles) < len(selected):
                for available in sorted(
                    (value for value in future if value >= cycle),
                    reverse=True,
                ):
                    for i in future.pop(available):
                        push(i)
                used = {engine: 0 for engine in engines}
                made_progress = True
                while made_progress:
                    made_progress = False
                    for engine in engine_order:
                        heap = ready[engine]
                        while heap and used[engine] < SLOT_LIMITS[engine]:
                            _, child = heapq.heappop(heap)
                            cycles[child] = cycle
                            used[engine] += 1
                            made_progress = True
                            for parent, lag in projected[child].items():
                                successor_count[parent] -= 1
                                latest_at[parent] = min(
                                    latest_at[parent], cycle - lag
                                )
                                if not successor_count[parent]:
                                    if latest_at[parent] >= cycle:
                                        push(parent)
                                    else:
                                        future.setdefault(
                                            latest_at[parent], []
                                        ).append(parent)
                if any(used.values()):
                    cycle -= 1
                elif future:
                    cycle = max(future)
                else:
                    raise AssertionError("projected graph is cyclic")
            violation = max(
                (earliest[i] - cycles[i] for i in selected), default=0
            )
            shift = max(0, violation)
            if shift:
                cycles = {i: value + shift for i, value in cycles.items()}
            return horizon + shift, cycles

        best: tuple[int, int, dict[int, int]] | None = None
        direction = os.environ.get("GREEDY_DIRECTION", "backward")
        start_seed = int(os.environ.get("GREEDY_START", "0"))
        save_max = int(os.environ.get("SAVE_MAX", "-1"))
        save_prefix = os.environ.get("SAVE_PREFIX", "")
        accepted = 0
        for seed in range(start_seed, start_seed + greedy_trials):
            score, cycles = (
                greedy_forward(seed)
                if direction == "forward"
                else greedy_backward(seed)
            )
            candidate = (score, seed, cycles)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
                print(f"greedy score={score} seed={seed}", flush=True)
            if save_prefix and score <= save_max:
                for engine in engines:
                    output = Path(f"{save_prefix}-{seed}-{engine}.json")
                    output.write_text(
                        json.dumps(
                            {
                                "engine": engine,
                                "horizon": score,
                                "seed": seed,
                                "cycles": {
                                    str(i): cycles[i]
                                    for i in selected
                                    if ops[i].engine == engine
                                },
                            }
                        )
                    )
                accepted += 1
        assert best is not None
        score, seed, cycles = best
        prefix = Path(os.environ.get("OUT_PREFIX", "/tmp/aopt-joint-greedy"))
        for engine in engines:
            output = Path(f"{prefix}-{engine}.json")
            output.write_text(
                json.dumps(
                    {
                        "engine": engine,
                        "horizon": score,
                        "seed": seed,
                        "cycles": {
                            str(i): cycles[i]
                            for i in selected
                            if ops[i].engine == engine
                        },
                    }
                )
            )
        print(
            f"greedy_best={score} seed={seed} accepted={accepted} "
            f"output_prefix={prefix}",
            flush=True,
        )
        shift = score - horizon if direction == "backward" else 0
        for engine in engines:
            raw_usage = Counter(
                cycles[i] - shift
                for i in selected
                if ops[i].engine == engine
            )
            print(
                f"greedy_{engine}_raw_start={min(raw_usage)} "
                f"idle_slots={sum(SLOT_LIMITS[engine] - raw_usage[cycle] for cycle in range(min(raw_usage), horizon))}",
                flush=True,
            )
        critical = sorted(
            selected,
            key=lambda i: (
                earliest[i] - (cycles[i] - shift),
                earliest[i],
                -i,
            ),
            reverse=True,
        )[:24]
        for i in critical:
            op = ops[i]
            print(
                f"release_violation={earliest[i] - (cycles[i] - shift):3d} "
                f"raw={cycles[i] - shift:4d} earliest={earliest[i]:3d} "
                f"{op.engine:5s} i={i:5d} g={op.group:2d} "
                f"r={op.round:2d} {op.tag}",
                flush=True,
            )
        return

    hint_payloads = []
    hint_targets: dict[int, int] = {}
    for raw_path in os.environ.get("PROJECTED_HINTS", "").split(","):
        if not raw_path:
            continue
        payload = json.loads(Path(raw_path).read_text())
        hint_payloads.append(payload)
        hint_engine = payload["engine"]
        if hint_engine not in engines:
            continue
        hint_cycles = {
            int(index): int(cycle)
            for index, cycle in payload["cycles"].items()
        }
        source_horizon = int(
            payload.get("horizon", max(hint_cycles.values(), default=0) + 1)
        )
        for i, cycle in hint_cycles.items():
            hint_targets[i] = (
                round(cycle * (horizon - 1) / max(1, source_horizon - 1))
                if bool(int(os.environ.get("SCALE_HINTS", "0")))
                else cycle
            )

    domain_radius = int(os.environ.get("DOMAIN_RADIUS", "-1"))
    domain_radius_engines = frozenset(
        value
        for value in os.environ.get(
            "DOMAIN_RADIUS_ENGINES", ",".join(engines)
        ).split(",")
        if value
    )
    bounds: dict[int, tuple[int, int]] = {}
    for i in selected:
        natural_lower = earliest[i]
        natural_upper = horizon - 1 - tail[i]
        if (
            domain_radius >= 0
            and i in hint_targets
            and ops[i].engine in domain_radius_engines
        ):
            center = min(
                natural_upper, max(natural_lower, hint_targets[i])
            )
            lower = max(natural_lower, center - domain_radius)
            upper = min(natural_upper, center + domain_radius)
        else:
            lower, upper = natural_lower, natural_upper
        bounds[i] = (lower, upper)

    model = cp_model.CpModel()
    starts = {
        i: model.new_int_var(bounds[i][0], bounds[i][1], f"s{i}")
        for i in selected
    }
    saturated_windows: dict[str, tuple[int, int]] = {}
    for raw_window in os.environ.get(
        "SATURATED_HALL_WINDOWS", ""
    ).split(","):
        if not raw_window:
            continue
        engine, raw_left, raw_right = raw_window.split(":")
        if engine not in engines:
            raise ValueError(
                f"saturated Hall engine {engine!r} is not selected"
            )
        left, right = int(raw_left), int(raw_right)
        if not 0 <= left <= right < horizon:
            raise ValueError(
                f"invalid saturated Hall window {engine}:{left}:{right}"
            )
        saturated_windows[engine] = (left, right)
        trapped = {
            i
            for i in selected
            if ops[i].engine == engine
            and earliest[i] >= left
            and horizon - 1 - tail[i] <= right
        }
        required = SLOT_LIMITS[engine] * (right - left + 1)
        if len(trapped) != required:
            raise ValueError(
                f"{engine} Hall window is not saturated: "
                f"jobs={len(trapped)} capacity={required}"
            )
        for i in selected:
            if ops[i].engine != engine or i in trapped:
                continue
            lower, upper = bounds[i]
            if upper < left or lower > right:
                continue
            can_be_early = lower <= left - 1
            can_be_late = upper >= right + 1
            if can_be_early and can_be_late:
                early_side = model.new_bool_var(f"hall_early_{engine}_{i}")
                model.add(starts[i] <= left - 1).only_enforce_if(
                    early_side
                )
                model.add(starts[i] >= right + 1).only_enforce_if(
                    early_side.negated()
                )
            elif can_be_early:
                model.add(starts[i] <= left - 1)
            elif can_be_late:
                model.add(starts[i] >= right + 1)
            else:
                model.add_bool_or([])
        print(
            f"saturated_hall_window engine={engine} "
            f"window={left}:{right} trapped={len(trapped)}",
            flush=True,
        )
    domain_sizes = {
        engine: sum(
            bounds[i][1] - bounds[i][0] + 1
            for i in selected
            if ops[i].engine == engine
        )
        for engine in engines
    }
    print(f"domain_sizes={domain_sizes}", flush=True)
    for child in selected:
        for parent, lag in projected[child].items():
            model.add(starts[child] >= starts[parent] + lag)
    fixed_hint_orders = frozenset(
        value
        for value in os.environ.get("FIX_HINT_ORDERS", "").split(",")
        if value
    )
    for payload in hint_payloads:
        hint_engine = payload["engine"]
        if hint_engine not in fixed_hint_orders:
            continue
        capacity = SLOT_LIMITS[hint_engine]
        ordered = sorted(
            (
                (int(cycle), int(index))
                for index, cycle in payload["cycles"].items()
                if int(index) in starts
            )
        )
        cycle_counts = Counter(cycle for cycle, _ in ordered)
        if max(cycle_counts.values(), default=0) > capacity:
            raise ValueError(f"hint for {hint_engine} exceeds capacity")
        for (_, parent), (_, child) in zip(ordered, ordered[capacity:]):
            model.add(starts[child] >= starts[parent] + 1)
        print(
            f"fixed_hint_order engine={hint_engine} jobs={len(ordered)}",
            flush=True,
        )
    fixed_hint_cycles = frozenset(
        value
        for value in os.environ.get("FIX_HINT_CYCLES", "").split(",")
        if value
    )
    for payload in hint_payloads:
        hint_engine = payload["engine"]
        if hint_engine not in fixed_hint_cycles:
            continue
        fixed = 0
        for raw_index in payload["cycles"]:
            index = int(raw_index)
            if index not in starts:
                continue
            cycle = hint_targets[index]
            lower, upper = bounds[index]
            if not lower <= cycle <= upper:
                raise ValueError(
                    f"fixed {hint_engine} hint cycle outside domain: "
                    f"i={index} cycle={cycle} domain={lower}:{upper}"
                )
            model.add(starts[index] == cycle)
            fixed += 1
        print(
            f"fixed_hint_cycles engine={hint_engine} jobs={fixed}",
            flush=True,
        )
    time_indexed_engines = frozenset(
        value
        for value in os.environ.get("TIME_INDEXED_ENGINES", "").split(",")
        if value
    )
    assignments: dict[tuple[int, int], cp_model.IntVar] = {}
    microslot_engines = frozenset(
        value
        for value in os.environ.get("MICROSLOT_ENGINES", "").split(",")
        if value
    )
    for engine in engines:
        engine_ops = [i for i in selected if ops[i].engine == engine]
        if engine in microslot_engines:
            capacity = SLOT_LIMITS[engine]
            microslots = []
            for i in engine_ops:
                lower, upper = bounds[i]
                if capacity == 1:
                    microslots.append(starts[i])
                    continue
                lane = model.new_int_var(
                    0, capacity - 1, f"lane_{engine}_{i}"
                )
                microslot = model.new_int_var(
                    capacity * lower,
                    capacity * upper + capacity - 1,
                    f"microslot_{engine}_{i}",
                )
                model.add(microslot == capacity * starts[i] + lane)
                microslots.append(microslot)
            # Near-saturated engines are much easier to propagate as a full
            # permutation.  The unrestricted hole variables name the few
            # unused microslots; with exactly one variable per domain value,
            # AllDifferent then proves that jobs plus holes cover the entire
            # horizon instead of merely being pairwise distinct.
            if bool(int(os.environ.get("SATURATE_MICROSLOTS", "0"))):
                hole_count = capacity * horizon - len(engine_ops)
                if hole_count < 0:
                    model.add_bool_or([])
                for hole in range(hole_count):
                    microslots.append(
                        model.new_int_var(
                            0,
                            capacity * horizon - 1,
                            f"microslot_hole_{engine}_{hole}",
                        )
                    )
            model.add_all_different(microslots)
        if engine in time_indexed_engines:
            for i in engine_ops:
                choices = []
                lower, upper = bounds[i]
                for cycle in range(lower, upper + 1):
                    choice = model.new_bool_var(f"x{i}_{cycle}")
                    assignments[i, cycle] = choice
                    choices.append(choice)
                model.add_exactly_one(choices)
                model.add(
                    starts[i]
                    == sum(
                        cycle * assignments[i, cycle]
                        for cycle in range(lower, upper + 1)
                    )
                )
            for cycle in range(horizon):
                choices = [
                    assignments[i, cycle]
                    for i in engine_ops
                    if (i, cycle) in assignments
                ]
                if bool(int(os.environ.get("SATURATE_TIME_INDEXED", "1"))):
                    hole = model.new_int_var(
                        0, SLOT_LIMITS[engine], f"hole_{engine}_{cycle}"
                    )
                    model.add(sum(choices) + hole == SLOT_LIMITS[engine])
                else:
                    model.add(sum(choices) <= SLOT_LIMITS[engine])
        else:
            intervals = [
                model.new_fixed_size_interval_var(starts[i], 1, f"i{i}")
                for i in engine_ops
            ]
            model.add_cumulative(
                intervals,
                [1] * len(intervals),
                SLOT_LIMITS[engine],
            )

    # Hints may come from incompatible independent schedules; they remain
    # useful as soft phase suggestions and never constrain the joint solve.
    for payload in hint_payloads:
        hint_engine = payload["engine"]
        hint_cycles = {
            int(index): int(cycle)
            for index, cycle in payload["cycles"].items()
        }
        if hint_engine not in engines:
            continue
        for i in hint_cycles:
            if i in starts:
                lower, upper = bounds[i]
                clamped = min(upper, max(lower, hint_targets[i]))
                model.add_hint(starts[i], clamped)
                for candidate in range(lower, upper + 1):
                    if (i, candidate) in assignments:
                        model.add_hint(
                            assignments[i, candidate],
                            int(candidate == clamped),
                        )

    makespan = None
    if bool(int(os.environ.get("OPTIMIZE", "0"))):
        makespan = model.new_int_var(0, horizon - 1, "makespan")
        for i in selected:
            model.add(makespan >= starts[i] + tail[i])
        model.minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(os.environ.get("TIME_LIMIT", "300"))
    solver.parameters.num_workers = int(os.environ.get("WORKERS", "8"))
    solver.parameters.random_seed = int(os.environ.get("RANDOM_SEED", "1"))
    solver.parameters.randomize_search = bool(int(os.environ.get("RANDOMIZE", "0")))
    solver.parameters.repair_hint = bool(int(os.environ.get("REPAIR_HINT", "1")))
    solver.parameters.hint_conflict_limit = 500_000
    solver.parameters.log_search_progress = bool(int(os.environ.get("LOG", "0")))
    status = solver.solve(model)
    print(f"status={solver.status_name(status)}", flush=True)
    if makespan is not None:
        print(
            f"objective={solver.objective_value + 1:g} "
            f"best_bound={solver.best_objective_bound + 1:g}",
            flush=True,
        )
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return

    result = {i: solver.value(starts[i]) for i in selected}
    prefix = Path(os.environ.get("OUT_PREFIX", "/tmp/aopt-joint"))
    for engine in engines:
        engine_cycles = {
            str(i): result[i] for i in selected if ops[i].engine == engine
        }
        usage = Counter(engine_cycles.values())
        if max(usage.values(), default=0) > SLOT_LIMITS[engine]:
            raise AssertionError(f"{engine} capacity overflow")
        output = Path(f"{prefix}-{engine}.json")
        output.write_text(
            json.dumps(
                {"engine": engine, "horizon": horizon, "cycles": engine_cycles}
            )
        )
        print(
            f"engine={engine} first={min(usage)} last={max(usage)} "
            f"holes={horizon * SLOT_LIMITS[engine] - len(engine_cycles)} "
            f"output={output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
