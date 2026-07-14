"""Solve one VLIW engine after projecting away all other instructions.

The full kernel has about twenty thousand operations, but only the operations
issued by the selected engine consume its capacity.  Every path through the
other engines is summarized as a weighted edge between consecutive selected
operations.  This keeps all true/anti dependency timing constraints while
shrinking the CP-SAT model by roughly an order of magnitude.
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
    engine = os.environ.get("ENGINE", "load")
    horizon = int(os.environ.get("TARGET", "959"))
    capacity = SLOT_LIMITS[engine]
    selected = [i for i, op in enumerate(ops) if op.engine == engine]
    selected_set = set(selected)

    earliest = [0] * len(ops)
    children: list[list[tuple[int, int]]] = [[] for _ in ops]
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            earliest[child] = max(earliest[child], earliest[parent] + lag)
            children[parent].append((child, lag))
    tail = [0] * len(ops)
    for parent in reversed(range(len(ops))):
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
    for child, op in enumerate(ops):
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
            projected[child].items(), reverse=True
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
    if bool(int(os.environ.get("ANALYZE_ONLY", "0"))):
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
        print(f"greedy_best={best[0]} seed={best[1]} output={output}")
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
