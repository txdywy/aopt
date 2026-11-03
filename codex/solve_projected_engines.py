"""Jointly schedule several VLIW engines after projecting away the rest.

Independent per-engine optima can encode mutually cyclic resource orders.  A
joint projection retains every path between operations on any selected engine
and lets CP-SAT choose compatible orders under all selected capacities at
once.  This is especially useful for the saturated flow/load pair while still
being much smaller than the complete roughly 20k-operation RCPSP.
"""

from __future__ import annotations

from collections import Counter, deque
import heapq
import json
import os
from pathlib import Path
import random

from ortools.sat.python import cp_model

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from codex.partial_order_hints import (
    audit_augmented_dag,
    load_partial_order_hints,
)
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

    requested_engines = tuple(
        value for value in os.environ.get("ENGINES", "flow,load").split(",")
        if value
    )
    if not requested_engines or len(set(requested_engines)) != len(requested_engines):
        raise ValueError("ENGINES must contain distinct engine names")
    # Pair projections otherwise erase setup-only producers on an omitted
    # engine (for example the VALU constant broadcasts consume LOAD/ALU
    # setup values).  In anchor mode retain those group=-1 operations and
    # their engine-local capacity, while still projecting away all dynamic
    # work on omitted engines.  This is a diagnostic bridge to a full solve,
    # not a relaxation: setup resources are real and remain constrained.
    if os.environ.get("INCLUDE_SETUP_ANCHORS", "0") == "1":
        setup_engines = tuple(
            sorted({op.engine for op in ops if op.group is None or op.group < 0})
        )
        engines = tuple(dict.fromkeys((*requested_engines, *setup_engines)))
    else:
        engines = requested_engines
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

    include_setup = os.environ.get("INCLUDE_SETUP_ANCHORS", "0") == "1"
    selected = [
        i for i in topological
        if ops[i].engine in requested_engines
        or (include_setup and (ops[i].group is None or ops[i].group < 0))
    ]
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

    # Preserve the unreduced same-engine projection for the alternating
    # relaxed model below.  Its cross-engine edges are intentionally soft, so
    # a global transitive reduction is not sound for a same-engine edge whose
    # only proof goes through one of those soft edges.  The strict model still
    # uses the compact global reduction, where every projected edge is hard.
    raw_same_engine_projected = {
        child: {
            parent: lag
            for parent, lag in parents.items()
            if ops[parent].engine == ops[child].engine
        }
        for child, parents in projected.items()
    }
    raw_same_engine_direct_edges = tuple(
        (parent, child, lag)
        for child in selected
        for parent, lag in ops[child].parents.items()
        if parent in selected_set
        and ops[parent].engine == ops[child].engine
    )

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

    # A cycle calendar is only a soft teacher: independently solved engine
    # calendars may be mutually incompatible.  For joint search, accept only
    # explicit resource arcs from the audited partial-order format and add
    # those arcs as hard same-engine precedences.  This is deliberately done
    # after the weighted reduction above so that a retained arc cannot be
    # accidentally discarded as transitively implied by a path that crosses
    # an omitted engine.
    partial_paths = tuple(
        value
        for value in os.environ.get("PARTIAL_ORDER_HINTS", "").split(",")
        if value
    )
    if partial_paths:
        partial = load_partial_order_hints(partial_paths, ops)
        audit = audit_augmented_dag(ops, partial)
        for parent, child in partial.effective_edges:
            if parent not in selected_set or child not in selected_set:
                raise ValueError(
                    "partial-order endpoint is not in the selected projection"
                )
            projected[child][parent] = max(
                projected[child].get(parent, -1), 1
            )
        print(
            f"partial_order_hints paths={len(partial.paths)} "
            f"edges={len(partial.resource_edges)} "
            f"effective={len(partial.effective_edges)} "
            f"redundant={partial.direct_dag_redundant} "
            f"augmented_span={audit.longest_path_span}",
            flush=True,
        )

    counts = Counter(ops[i].engine for i in selected)
    edge_count = sum(map(len, projected.values()))
    hard_same_edge_count = sum(
        map(len, raw_same_engine_projected.values())
    )
    print(
        f"engines={engines} jobs={len(selected)} counts={dict(counts)} "
        f"projected_edges={edge_count} hard_same_edges="
        f"{hard_same_edge_count}",
        flush=True,
    )

    greedy_trials = int(os.environ.get("GREEDY_TRIALS", "0"))
    if greedy_trials:
        # Optional per-engine 959 schedules act as teacher permutations.  An
        # independently feasible engine order is far more informative than a
        # generic tail/fanout score on these almost completely saturated
        # machines.  The joint greedy is still free to interleave the two
        # teachers differently when cross-engine edges require it.
        greedy_guide: dict[int, int] = {}
        for raw_path in os.environ.get("GREEDY_GUIDE_HINTS", "").split(","):
            if not raw_path:
                continue
            payload = json.loads(Path(raw_path).read_text())
            if payload.get("engine") not in engines:
                continue
            greedy_guide.update(
                (int(index), int(cycle))
                for index, cycle in payload["cycles"].items()
            )
        if greedy_guide and set(greedy_guide) != selected_set:
            raise ValueError(
                "GREEDY_GUIDE_HINTS must cover every selected operation"
            )
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
            guide_weight = rng.choice(
                (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
            )
            if "GREEDY_GUIDE_WEIGHT" in os.environ:
                guide_weight = int(os.environ["GREEDY_GUIDE_WEIGHT"])
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
                    guide_weight * (horizon - greedy_guide.get(i, horizon))
                    + tail_weight * tail[i]
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
            earliest_bucket = int(
                os.environ.get("GREEDY_EARLIEST_BUCKET", "0")
            )
            release_bucket = int(
                os.environ.get("GREEDY_RELEASE_BUCKET", "0")
            )
            phase_bucket = int(
                os.environ.get("GREEDY_PHASE_BUCKET", "0")
            )
            early_weight = rng.choice((2, 4, 8, 16, 32, 64, 128))
            ancestor_divisor = rng.choice((8, 16, 32, 64, 128, 256, 512))
            # Expose the reach divisor as a fixed research knob too.  The
            # winning schedules in a long stochastic sweep are otherwise
            # difficult to replay because this draw happens before the
            # per-operation noise stream and engine-order shuffle.
            unlock_reach_divisor = int(
                os.environ.get(
                    "GREEDY_FIXED_UNLOCK_REACH",
                    str(
                        rng.choice(
                            (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
                        )
                    ),
                )
            )
            fanin_weight = rng.choice((0, 1, 2, 4, 8, 16, 32))
            unlock_weight = rng.choice((0, 16, 32, 64, 128, 256, 512, 1024))
            group_weight = rng.choice((-8, -4, -2, -1, 0, 1, 2, 4, 8))
            tail_weight = rng.choice((0, 1, 2, 4, 8, 16))
            phase_weight = rng.choice(
                (-128, -64, -32, -16, -8, -4, 0, 4, 8, 16, 32, 64, 128)
            )
            if "GREEDY_PHASE_WEIGHT" in os.environ:
                phase_weight = int(os.environ["GREEDY_PHASE_WEIGHT"])
            guide_weight = rng.choice(
                (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
            )
            if "GREEDY_GUIDE_WEIGHT" in os.environ:
                guide_weight = int(os.environ["GREEDY_GUIDE_WEIGHT"])
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
                unlock_release = max(
                    (
                        earliest[parent]
                        for parent in projected[i]
                        if successor_count[parent] == 1
                    ),
                    default=0,
                )
                release_pressure = max(earliest[i], unlock_release)
                phase_pressure = (
                    max(kernel.FULL_ROUND_OFFSETS) + 1
                    if op.group is None or op.group < 0
                    else max(kernel.FULL_ROUND_OFFSETS)
                    - kernel.FULL_ROUND_OFFSETS[op.group]
                )
                group = op.group if op.group is not None else -1
                return (
                    (
                        phase_pressure // phase_bucket
                        if phase_bucket > 0
                        else 0
                    ),
                    (
                        release_pressure // release_bucket
                        if release_bucket > 0
                        else 0
                    ),
                    (
                        earliest[i] // earliest_bucket
                        if earliest_bucket > 0
                        else 0
                    ),
                    guide_weight * greedy_guide.get(i, 0)
                    + early_weight * earliest[i]
                    + ancestor_count[i] // ancestor_divisor
                    + fanin_weight * len(projected[i])
                    + unlock_weight * unlock
                    + group_weight * group
                    + phase_weight * phase_pressure
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
    hint_orders: dict[str, list[int]] = {}
    for raw_path in os.environ.get("PROJECTED_HINTS", "").split(","):
        if not raw_path:
            continue
        payload = json.loads(Path(raw_path).read_text())
        hint_payloads.append(payload)
        hint_engine = payload["engine"]
        if hint_engine not in engines:
            continue
        if "order" in payload:
            hint_orders[hint_engine] = [int(i) for i in payload["order"]]
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

    dump_order_problem = os.environ.get("DUMP_ORDER_PROBLEM", "")
    if dump_order_problem:
        if set(hint_targets) != selected_set:
            raise ValueError(
                "order-problem dump requires a joint hint for every job"
            )
        compact = {job: rank for rank, job in enumerate(selected)}
        lines = [
            f"{len(selected)} {sum(map(len, projected.values()))} "
            f"{len(engines)}"
        ]
        engine_number = {engine: rank for rank, engine in enumerate(engines)}
        for job in selected:
            lines.append(
                f"{job} {engine_number[ops[job].engine]} {earliest[job]} "
                f"{tail[job]} {hint_targets[job]}"
            )
        for child in selected:
            for parent, lag in projected[child].items():
                lines.append(
                    f"{compact[parent]} {compact[child]} {lag}"
                )
        for engine in engines:
            order = hint_orders.get(engine)
            if order is None:
                order = sorted(
                    (job for job in selected if ops[job].engine == engine),
                    key=lambda job: (hint_targets[job], position[job]),
                )
            expected = {job for job in selected if ops[job].engine == engine}
            if set(order) != expected or len(order) != len(expected):
                raise ValueError(f"invalid explicit order for {engine}")
            lines.append(
                f"{engine} {SLOT_LIMITS[engine]} {len(order)} "
                + " ".join(str(compact[job]) for job in order)
            )
        Path(dump_order_problem).write_text("\n".join(lines) + "\n")
        print(
            f"dumped_order_problem={dump_order_problem} jobs={len(selected)}",
            flush=True,
        )
        return

    alternating_rounds = int(os.environ.get("ALTERNATING_REPAIR", "0"))
    if alternating_rounds:
        if set(hint_targets) != selected_set:
            raise ValueError(
                "alternating repair requires one 959 hint per selected job"
            )
        current = dict(hint_targets)
        per_engine_limit = float(
            os.environ.get("ALTERNATING_TIME_LIMIT", "30")
        )
        proximity_weight = int(
            os.environ.get("ALTERNATING_PROXIMITY_WEIGHT", "1")
        )
        output_prefix = Path(
            os.environ.get("OUT_PREFIX", "/tmp/aopt-alternating")
        )

        def cross_metrics(cycles: dict[int, int]) -> tuple[int, int, int]:
            violations = []
            for child in selected:
                for parent, lag in projected[child].items():
                    if ops[parent].engine == ops[child].engine:
                        continue
                    amount = cycles[parent] + lag - cycles[child]
                    if amount > 0:
                        violations.append(amount)
            return (
                max(violations, default=0),
                sum(violations),
                len(violations),
            )

        def raw_same_engine_direct_violations(
            cycles: dict[int, int],
        ) -> list[tuple[int, int, int, int]]:
            return [
                (cycles[parent] + lag - cycles[child], parent, child, lag)
                for parent, child, lag in raw_same_engine_direct_edges
                if cycles[parent] + lag > cycles[child]
            ]

        def raw_same_engine_projected_violations(
            cycles: dict[int, int],
        ) -> list[tuple[int, int, int, int]]:
            return [
                (cycles[parent] + lag - cycles[child], parent, child, lag)
                for child, parents in raw_same_engine_projected.items()
                for parent, lag in parents.items()
                if cycles[parent] + lag > cycles[child]
            ]

        def audit_raw_same_engine_direct(
            cycles: dict[int, int], label: str
        ) -> bool:
            violations = raw_same_engine_direct_violations(cycles)
            print(
                f"raw_same_engine_direct_audit label={label} "
                f"violations={len(violations)}",
                flush=True,
            )
            limit = int(
                os.environ.get("RAW_SAME_ENGINE_AUDIT_LIMIT", "8")
            )
            for amount, parent, child, lag in sorted(
                violations, reverse=True
            )[:limit]:
                parent_op = ops[parent]
                child_op = ops[child]
                print(
                    f"raw_same_engine_direct_violation amount={amount} "
                    f"lag={lag} parent={parent}@{cycles[parent]} "
                    f"{parent_op.engine}:g{parent_op.group}:"
                    f"r{parent_op.round}:{parent_op.tag} -> "
                    f"child={child}@{cycles[child]} "
                    f"{child_op.engine}:g{child_op.group}:"
                    f"r{child_op.round}:{child_op.tag}",
                    flush=True,
                )
            return not violations

        def write_alternating(cycles: dict[int, int]) -> None:
            if not audit_raw_same_engine_direct(cycles, "output"):
                raise ValueError(
                    "refusing to write a relaxed projected schedule that "
                    "violates an original same-engine DAG edge"
                )
            for engine in engines:
                Path(f"{output_prefix}-{engine}.json").write_text(
                    json.dumps(
                        {
                            "engine": engine,
                            "horizon": horizon,
                            "cycles": {
                                str(i): cycles[i]
                                for i in selected
                                if ops[i].engine == engine
                            },
                        }
                    )
                )

        print(
            f"alternating_start={cross_metrics(current)} horizon={horizon}",
            flush=True,
        )
        audit_raw_same_engine_direct(current, "input")
        if bool(int(os.environ.get("PRINT_CROSS_HISTOGRAM", "0"))):
            histogram = Counter()
            critical_edges = []
            for child in selected:
                for parent, lag in projected[child].items():
                    if ops[parent].engine == ops[child].engine:
                        continue
                    amount = current[parent] + lag - current[child]
                    if amount > 0:
                        histogram[amount] += 1
                        critical_edges.append((amount, parent, child, lag))
            print(
                "cross_histogram="
                + json.dumps(dict(sorted(histogram.items()))),
                flush=True,
            )
            if critical_edges:
                peak = max(amount for amount, *_ in critical_edges)
                group_histogram: Counter[int] = Counter()
                for amount, parent, child, _ in critical_edges:
                    if amount != peak:
                        continue
                    for group in {ops[parent].group, ops[child].group}:
                        if group >= 0:
                            group_histogram[group] += 1
                print(
                    "cross_peak_groups="
                    + json.dumps(
                        dict(group_histogram.most_common()), sort_keys=False
                    ),
                    flush=True,
                )
                direction_histogram: Counter[str] = Counter()
                tag_histogram: Counter[str] = Counter()
                pair_tag_histogram: Counter[str] = Counter()
                window_histogram: Counter[str] = Counter()
                for amount, parent, child, _ in critical_edges:
                    if amount != peak:
                        continue
                    parent_op = ops[parent]
                    child_op = ops[child]
                    direction_histogram[
                        f"{parent_op.engine}->{child_op.engine}"
                    ] += 1
                    tag_histogram[f"{parent_op.tag}->{child_op.tag}"] += 1
                    group = (
                        parent_op.group
                        if parent_op.group >= 0
                        else child_op.group
                    )
                    rnd = (
                        parent_op.round
                        if parent_op.round >= 0
                        else child_op.round
                    )
                    pair_tag_histogram[
                        f"g{group}:r{rnd}:"
                        f"{parent_op.tag}->{child_op.tag}"
                    ] += 1
                    window_histogram[
                        f"{parent_op.engine}:{current[parent] // 100}xx->"
                        f"{child_op.engine}:{current[child] // 100}xx"
                    ] += 1
                print(
                    "cross_peak_directions="
                    + json.dumps(dict(direction_histogram.most_common())),
                    flush=True,
                )
                print(
                    "cross_peak_tags="
                    + json.dumps(dict(tag_histogram.most_common(24))),
                    flush=True,
                )
                print(
                    "cross_peak_pair_tags="
                    + json.dumps(dict(pair_tag_histogram.most_common(256))),
                    flush=True,
                )
                print(
                    "cross_peak_windows="
                    + json.dumps(dict(window_histogram.most_common())),
                    flush=True,
                )
            if bool(int(os.environ.get("PRINT_CROSS_CRITICAL", "0"))):
                limit = int(os.environ.get("PRINT_CROSS_CRITICAL_LIMIT", "64"))
                for amount, parent, child, lag in sorted(
                    critical_edges, reverse=True
                )[:limit]:
                    parent_op = ops[parent]
                    child_op = ops[child]
                    print(
                        f"cross_edge amount={amount:3d} lag={lag:2d} "
                        f"parent={parent:5d}@{current[parent]:3d} "
                        f"{parent_op.engine}:g{parent_op.group}:r{parent_op.round}:"
                        f"{parent_op.tag} -> child={child:5d}@{current[child]:3d} "
                        f"{child_op.engine}:g{child_op.group}:r{child_op.round}:"
                        f"{child_op.tag}",
                        flush=True,
                    )
            return

        # Coordinate descent can stop at a state where neither engine can
        # improve while the other is fixed, even though a simultaneous local
        # exchange is feasible.  The joint mode releases a time-neighborhood
        # around both endpoints of every edge above the requested cap and
        # solves the two resource calendars together.  Jobs outside that
        # critical block remain fixed, keeping the model much smaller than a
        # cold full-engine solve.
        if bool(int(os.environ.get("ALTERNATING_JOINT", "0"))):
            joint_drop = int(
                os.environ.get("ALTERNATING_FORCE_MAX_DROP", "1")
            )
            joint_keep_max = bool(
                int(os.environ.get("ALTERNATING_KEEP_MAX", "0"))
            )
            joint_current_max = cross_metrics(current)[0]
            joint_cap = max(0, joint_current_max - joint_drop)
            joint_radius = int(
                os.environ.get("ALTERNATING_LOCAL_RADIUS", "4")
            )
            # A time window selects the operations that may participate in an
            # ejection chain, but their natural earliest/latest windows can
            # still span almost the entire 959-cycle horizon.  That creates a
            # needlessly huge Boolean calendar model and lets CP-SAT solve a
            # different global reordering problem.  Optionally bound each
            # selected operation around the incumbent issue cycle instead.
            # A negative value retains the historic unbounded behavior.
            joint_move_radius = int(
                os.environ.get("ALTERNATING_JOINT_MOVE_RADIUS", "-1")
            )
            joint_feasibility = bool(
                int(os.environ.get("ALTERNATING_JOINT_FEASIBILITY", "0"))
            )
            joint_soft_cap = bool(
                int(os.environ.get("ALTERNATING_JOINT_SOFT_CAP", "0"))
            )
            joint_secondary_sum = bool(
                int(os.environ.get("ALTERNATING_JOINT_SECONDARY_SUM", "0"))
            )
            raw_windows = os.environ.get("ALTERNATING_JOINT_WINDOWS", "")
            windows = [
                (engine, int(left), int(right))
                for engine, left, right in (
                    item.split(":")
                    for item in raw_windows.split(",")
                    if item
                )
            ]
            anchor_windows_only = bool(
                int(os.environ.get("ALTERNATING_ANCHOR_WINDOWS_ONLY", "0"))
            )

            def in_joint_window(index: int) -> bool:
                return any(
                    ops[index].engine == engine
                    and left <= current[index] <= right
                    for engine, left, right in windows
                )

            anchors = set()
            for child in selected:
                for parent, lag in projected[child].items():
                    if ops[parent].engine == ops[child].engine:
                        continue
                    amount = current[parent] + lag - current[child]
                    if (
                        amount < joint_cap
                        if joint_keep_max
                        else amount <= joint_cap
                    ):
                        continue
                    # In staged soft-cap repair, leave peak edges outside
                    # the selected time phase as a fixed shell.  They retain
                    # the current hard maximum while this solve spends its
                    # capacity on one coherent wave rather than dragging all
                    # unrelated critical endpoints into the CP model.
                    if (
                        anchor_windows_only
                        and windows
                        and not (in_joint_window(parent) or in_joint_window(child))
                    ):
                        continue
                    anchors.add(parent)
                    anchors.add(child)
            # A globally reduced same-engine edge may have been justified by
            # a path containing ALU<->VALU edges.  Those edges are soft in
            # this mode, so release every endpoint of an unreduced hard
            # same-engine violation regardless of the selected cross window.
            hard_same_violations = raw_same_engine_projected_violations(
                current
            )
            if hard_same_violations:
                for _, parent, child, _ in hard_same_violations:
                    anchors.add(parent)
                    anchors.add(child)
                print(
                    f"alternating_joint_hard_same_violations="
                    f"{len(hard_same_violations)}",
                    flush=True,
                )
            if bool(
                int(os.environ.get("ALTERNATING_ANCHOR_RESOURCE_OVERFLOW", "0"))
            ):
                resource_usage = Counter(
                    (ops[i].engine, current[i]) for i in selected
                )
                overflow_cycles = {
                    (engine, cycle)
                    for (engine, cycle), demand in resource_usage.items()
                    if demand > SLOT_LIMITS[engine]
                }
                overflow_jobs = {
                    i
                    for i in selected
                    if (ops[i].engine, current[i]) in overflow_cycles
                }
                anchors.update(overflow_jobs)
                print(
                    f"alternating_resource_overflow cycles="
                    f"{len(overflow_cycles)} jobs={len(overflow_jobs)}",
                    flush=True,
                )
            anchor_cycles = [current[i] for i in anchors]
            if raw_windows:
                repair_local = set(anchors)
                repair_local.update(
                    i for i in selected if in_joint_window(i)
                )
            elif joint_radius:
                repair_local = {
                    i
                    for i in selected
                    if i in anchors
                    or any(
                        abs(current[i] - cycle) <= joint_radius
                        for cycle in anchor_cycles
                    )
                }
            else:
                repair_local = set(selected)
            print(
                f"alternating_joint cap={joint_cap} anchors={len(anchors)} "
                f"jobs={len(repair_local)} radius={joint_radius} "
                f"move_radius={joint_move_radius} "
                f"windows={raw_windows or '-'}",
                flush=True,
            )
            model = cp_model.CpModel()
            explicit_domains = bool(
                int(os.environ.get("ALTERNATING_EXPLICIT_DOMAINS", "0"))
            )
            fixed_usage = Counter(
                (ops[i].engine, current[i])
                for i in selected
                if i not in repair_local
            )
            available_cycles = {
                engine: [
                    cycle
                    for cycle in range(horizon)
                    if fixed_usage[engine, cycle] < SLOT_LIMITS[engine]
                ]
                for engine in engines
            }
            starts = {}
            for i in selected:
                lower = earliest[i]
                upper = horizon - 1 - tail[i]
                if i not in repair_local:
                    lower = current[i]
                    upper = current[i]
                elif joint_move_radius >= 0:
                    lower = max(lower, current[i] - joint_move_radius)
                    upper = min(upper, current[i] + joint_move_radius)
                if explicit_domains and i in repair_local:
                    domain = [
                        cycle
                        for cycle in available_cycles[ops[i].engine]
                        if lower <= cycle <= upper
                    ]
                    if not domain:
                        raise ValueError(
                            f"empty residual-resource domain for job {i}"
                        )
                    starts[i] = model.new_int_var_from_domain(
                        cp_model.Domain.from_values(domain), f"s{i}"
                    )
                else:
                    starts[i] = model.new_int_var(lower, upper, f"s{i}")
            for child, parents in raw_same_engine_projected.items():
                for parent, lag in parents.items():
                    model.add(starts[child] >= starts[parent] + lag)
            for engine in engines:
                local = [
                    i
                    for i in selected
                    if ops[i].engine == engine
                    and (not explicit_domains or i in repair_local)
                ]
                intervals = [
                    model.new_fixed_size_interval_var(starts[i], 1, f"i{i}")
                    for i in local
                ]
                demands = [1] * len(intervals)
                if explicit_domains:
                    for cycle in range(horizon):
                        demand = fixed_usage[engine, cycle]
                        if not demand:
                            continue
                        intervals.append(
                            model.new_fixed_size_interval_var(
                                cycle, 1, f"fixed_{engine}_{cycle}"
                            )
                        )
                        demands.append(demand)
                model.add_cumulative(intervals, demands, SLOT_LIMITS[engine])
            violations = []
            excesses = []
            fixed_violation_sum = 0
            for child in selected:
                for parent, lag in projected[child].items():
                    if ops[parent].engine == ops[child].engine:
                        continue
                    if parent not in repair_local and child not in repair_local:
                        amount = max(
                            0, current[parent] + lag - current[child]
                        )
                        fixed_cap = (
                            joint_current_max if joint_soft_cap else joint_cap
                        )
                        if amount > fixed_cap:
                            raise ValueError(
                                "joint critical block omitted a capped edge"
                            )
                        fixed_violation_sum += amount
                        continue
                    if not joint_soft_cap:
                        model.add(
                            starts[parent] + lag - starts[child] <= joint_cap
                        )
                    else:
                        # Minimize excess above the next cap without allowing
                        # the optimizer to concentrate that excess into a
                        # taller peak.  This makes successive soft-cap solves
                        # monotone in the true maximum violation.
                        model.add(
                            starts[parent] + lag - starts[child]
                            <= joint_current_max
                        )
                    violation = model.new_int_var(
                        0,
                        horizon if joint_soft_cap else joint_cap,
                        f"jv{parent}_{child}",
                    )
                    model.add(
                        violation
                        >= starts[parent] + lag - starts[child]
                    )
                    violations.append(violation)
                    if joint_soft_cap:
                        excess = model.new_int_var(
                            0, horizon, f"je{parent}_{child}"
                        )
                        model.add(
                            excess
                            >= starts[parent] + lag - starts[child] - joint_cap
                        )
                        excesses.append(excess)
            if joint_soft_cap:
                if joint_secondary_sum:
                    # Strict lexicographic scalarization: changing the primary
                    # excess by one dominates the complete range of the
                    # secondary violation sum.  Including the fixed-shell
                    # constant also makes objective values comparable across
                    # different neighborhood radii.
                    primary_scale = horizon * len(violations) + 1
                    model.minimize(
                        sum(excesses) * primary_scale
                        + sum(violations)
                        + fixed_violation_sum
                    )
                else:
                    model.minimize(sum(excesses))
            elif not joint_feasibility:
                model.minimize(sum(violations))
            for i in selected:
                model.add_hint(starts[i], current[i])
            joint_solver = cp_model.CpSolver()
            joint_solver.parameters.max_time_in_seconds = float(
                os.environ.get("ALTERNATING_JOINT_TIME_LIMIT", "300")
            )
            joint_solver.parameters.num_workers = int(
                os.environ.get("WORKERS", "8")
            )
            joint_solver.parameters.random_seed = int(
                os.environ.get("RANDOM_SEED", "1")
            )
            if bool(
                int(os.environ.get("ALTERNATING_JOINT_LNS_ONLY", "0"))
            ):
                joint_solver.parameters.use_lns_only = True
            joint_status = joint_solver.solve(model)
            print(
                f"alternating_joint status="
                f"{joint_solver.status_name(joint_status)} objective="
                f"{joint_solver.objective_value} fixed_sum="
                f"{fixed_violation_sum}",
                flush=True,
            )
            if joint_status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
                current.update(
                    {i: joint_solver.value(starts[i]) for i in selected}
                )
                print(
                    f"alternating_joint_metrics max_sum_count="
                    f"{cross_metrics(current)}",
                    flush=True,
                )
                write_alternating(current)
            print(
                f"alternating_final={cross_metrics(current)} "
                f"output_prefix={output_prefix}",
                flush=True,
            )
            return

        for iteration in range(alternating_rounds):
            changed = False
            for engine_number, engine in enumerate(engines):
                local = [i for i in selected if ops[i].engine == engine]
                local_set = set(local)
                model = cp_model.CpModel()
                forced_drop = int(
                    os.environ.get("ALTERNATING_FORCE_MAX_DROP", "0")
                )
                keep_max = bool(
                    int(os.environ.get("ALTERNATING_KEEP_MAX", "0"))
                )
                force_cap = forced_drop > 0 or keep_max
                violation_cap = None
                if force_cap:
                    violation_cap = max(
                        0, cross_metrics(current)[0] - forced_drop
                    )
                repair_local = set(local)
                local_radius = int(
                    os.environ.get("ALTERNATING_LOCAL_RADIUS", "0")
                )
                if local_radius and violation_cap is not None:
                    anchors = set()
                    for child in selected:
                        for parent, lag in projected[child].items():
                            if ops[parent].engine == ops[child].engine:
                                continue
                            amount = current[parent] + lag - current[child]
                            # A forced descent must repair every edge above
                            # the target cap.  Plateau cleanup instead frees
                            # the edges currently sitting exactly on the cap,
                            # so the sum objective can rearrange the critical
                            # block without increasing the maximum.
                            if (
                                amount < violation_cap
                                if keep_max
                                else amount <= violation_cap
                            ):
                                continue
                            if parent in local_set:
                                anchors.add(parent)
                            if child in local_set:
                                anchors.add(child)
                    for _, parent, child, _ in (
                        raw_same_engine_projected_violations(current)
                    ):
                        if parent in local_set:
                            anchors.add(parent)
                        if child in local_set:
                            anchors.add(child)
                    anchor_cycles = [current[i] for i in anchors]
                    repair_local = {
                        i
                        for i in local
                        if i in anchors
                        or any(
                            abs(current[i] - cycle) <= local_radius
                            for cycle in anchor_cycles
                        )
                    }
                    print(
                        f"alternating_local engine={engine} anchors="
                        f"{len(anchors)} jobs={len(repair_local)} "
                        f"radius={local_radius}",
                        flush=True,
                    )
                direct_lower = {i: earliest[i] for i in local}
                direct_upper = {
                    i: horizon - 1 - tail[i] for i in local
                }
                if violation_cap is not None:
                    for child in selected:
                        for parent, lag in projected[child].items():
                            if ops[parent].engine == ops[child].engine:
                                continue
                            parent_local = parent in local_set
                            child_local = child in local_set
                            if parent_local == child_local:
                                continue
                            if child_local:
                                direct_lower[child] = max(
                                    direct_lower[child],
                                    current[parent] + lag - violation_cap,
                                )
                            else:
                                direct_upper[parent] = min(
                                    direct_upper[parent],
                                    current[child] - lag + violation_cap,
                                )
                if local_radius:
                    for i in local_set - repair_local:
                        direct_lower[i] = current[i]
                        direct_upper[i] = current[i]
                if any(direct_lower[i] > direct_upper[i] for i in local):
                    print(
                        f"alternating iteration={iteration} engine={engine} "
                        "status=INFEASIBLE_DIRECT_BOUNDS",
                        flush=True,
                    )
                    continue
                starts = {
                    i: model.new_int_var(
                        direct_lower[i], direct_upper[i], f"s{i}"
                    )
                    for i in local
                }
                for child in local:
                    for parent, lag in raw_same_engine_projected[child].items():
                        if parent in local_set:
                            model.add(starts[child] >= starts[parent] + lag)
                intervals = [
                    model.new_fixed_size_interval_var(starts[i], 1, f"i{i}")
                    for i in local
                ]
                model.add_cumulative(
                    intervals, [1] * len(intervals), SLOT_LIMITS[engine]
                )

                violations = []
                for child in selected:
                    for parent, lag in projected[child].items():
                        if ops[parent].engine == ops[child].engine:
                            continue
                        parent_local = parent in local_set
                        child_local = child in local_set
                        if parent_local == child_local:
                            continue
                        violation = model.new_int_var(0, horizon, f"v{parent}_{child}")
                        if child_local:
                            model.add(
                                violation
                                >= current[parent] + lag - starts[child]
                            )
                        else:
                            model.add(
                                violation
                                >= starts[parent] + lag - current[child]
                            )
                        violations.append(violation)
                maximum = model.new_int_var(0, horizon, "maximum_violation")
                model.add_max_equality(maximum, violations)
                if force_cap:
                    assert violation_cap is not None
                    model.add(maximum <= violation_cap)
                deviations = []
                if proximity_weight:
                    for i in local:
                        deviation = model.new_int_var(0, horizon, f"d{i}")
                        model.add_abs_equality(deviation, starts[i] - current[i])
                        deviations.append(deviation)
                # First drive down the worst broken dependence, then the
                # total broken distance, retaining the incumbent as a weak
                # proximal term to prevent two-engine oscillation.
                if force_cap:
                    model.minimize(
                        sum(violations) * 100
                        + proximity_weight * sum(deviations)
                    )
                else:
                    model.minimize(
                        maximum * 10_000_000
                        + sum(violations) * 100
                        + proximity_weight * sum(deviations)
                    )
                for i in local:
                    model.add_hint(starts[i], current[i])
                repair_solver = cp_model.CpSolver()
                repair_solver.parameters.max_time_in_seconds = float(
                    os.environ.get(
                        f"ALTERNATING_TIME_LIMIT_{engine.upper()}",
                        str(per_engine_limit),
                    )
                )
                repair_solver.parameters.num_workers = int(
                    os.environ.get("WORKERS", "8")
                )
                repair_solver.parameters.random_seed = (
                    int(os.environ.get("RANDOM_SEED", "1"))
                    + iteration * 101
                    + engine_number
                )
                repair_status = repair_solver.solve(model)
                print(
                    f"alternating iteration={iteration} engine={engine} "
                    f"status={repair_solver.status_name(repair_status)} "
                    f"objective={repair_solver.objective_value}",
                    flush=True,
                )
                if repair_status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
                    continue
                updated = {
                    i: repair_solver.value(starts[i]) for i in local
                }
                changed |= any(updated[i] != current[i] for i in local)
                current.update(updated)
                metrics = cross_metrics(current)
                print(
                    f"alternating_metrics iteration={iteration} "
                    f"engine={engine} max_sum_count={metrics}",
                    flush=True,
                )
                write_alternating(current)
                if metrics == (0, 0, 0):
                    print("alternating_feasible=1", flush=True)
                    return
            if not changed:
                break
        print(
            f"alternating_final={cross_metrics(current)} "
            f"output_prefix={output_prefix}",
            flush=True,
        )
        return

    # Diagnose why independently optimal engine schedules cannot simply be
    # superimposed.  Sorting each engine by its hint cycle and connecting
    # item k to k+capacity produces capacity-safe virtual lane chains.  Any
    # SCC in the union of those chains and the real dependence graph is an
    # explicit resource-order cycle; its internal resource edges are the
    # precise candidates a critical-block search must reverse.
    if bool(int(os.environ.get("ANALYZE_ORDER_CYCLES", "0"))):
        if set(hint_targets) != selected_set:
            raise ValueError(
                "order-cycle analysis requires a hint for every selected job"
            )
        compact = {job: rank for rank, job in enumerate(selected)}
        adjacency: list[list[int]] = [[] for _ in selected]
        reverse: list[list[int]] = [[] for _ in selected]
        edge_kind: dict[tuple[int, int], str] = {}

        def add_edge(parent: int, child: int, kind: str) -> None:
            left = compact[parent]
            right = compact[child]
            adjacency[left].append(right)
            reverse[right].append(left)
            edge_kind[(left, right)] = kind

        for child in selected:
            for parent in projected[child]:
                add_edge(parent, child, "dep")
        for engine in engines:
            capacity = SLOT_LIMITS[engine]
            expected = {i for i in selected if ops[i].engine == engine}
            order = hint_orders.get(engine)
            if order is None:
                order = sorted(
                    expected,
                    key=lambda i: (hint_targets[i], position[i]),
                )
            if len(order) != len(expected) or set(order) != expected:
                raise ValueError(f"invalid explicit order for {engine}")
            for rank in range(len(order) - capacity):
                add_edge(order[rank], order[rank + capacity], engine)

        visited = [False] * len(selected)
        finish: list[int] = []
        for root in range(len(selected)):
            if visited[root]:
                continue
            visited[root] = True
            stack: list[tuple[int, int]] = [(root, 0)]
            while stack:
                node, offset = stack[-1]
                if offset < len(adjacency[node]):
                    child = adjacency[node][offset]
                    stack[-1] = (node, offset + 1)
                    if not visited[child]:
                        visited[child] = True
                        stack.append((child, 0))
                    continue
                finish.append(node)
                stack.pop()

        component = [-1] * len(selected)
        components: list[list[int]] = []
        for root in reversed(finish):
            if component[root] >= 0:
                continue
            number = len(components)
            members: list[int] = []
            component[root] = number
            stack = [root]
            while stack:
                node = stack.pop()
                members.append(node)
                for parent in reverse[node]:
                    if component[parent] < 0:
                        component[parent] = number
                        stack.append(parent)
            components.append(members)

        cyclic = [members for members in components if len(members) > 1]
        internal = Counter()
        examples: list[tuple[int, str, int, int]] = []
        for (left, right), kind in edge_kind.items():
            if component[left] != component[right]:
                continue
            if len(components[component[left]]) <= 1:
                continue
            internal[kind] += 1
            if kind != "dep":
                examples.append(
                    (
                        len(components[component[left]]),
                        kind,
                        selected[left],
                        selected[right],
                    )
                )
        print(
            f"order_sccs={len(cyclic)} cyclic_jobs="
            f"{sum(map(len, cyclic))} largest="
            f"{max(map(len, cyclic), default=0)} internal_edges="
            f"{dict(internal)}",
            flush=True,
        )
        for size, kind, parent, child in sorted(examples, reverse=True)[:64]:
            parent_op = ops[parent]
            child_op = ops[child]
            print(
                f"cycle_resource size={size:5d} engine={kind:5s} "
                f"{hint_targets[parent]:4d}->{hint_targets[child]:4d} "
                f"p={parent:5d} g={parent_op.group!s:>2s} "
                f"r={parent_op.round!s:>2s} {parent_op.tag} | "
                f"c={child:5d} g={child_op.group!s:>2s} "
                f"r={child_op.round!s:>2s} {child_op.tag}",
                flush=True,
            )
        return

    # Critical-block search over capacity-safe global resource orders.  For
    # an engine of capacity C, an order O is enforced by O[k] -> O[k+C].
    # The earliest schedule of the resulting DAG is automatically feasible.
    # This representation removes the millions of time-indexed booleans that
    # made broad CP neighborhoods stall, while retaining exactly the order
    # decisions that can shorten the critical path.
    global_order_trials = int(os.environ.get("SEARCH_GLOBAL_ORDERS", "0"))
    if global_order_trials:
        if set(hint_targets) != selected_set:
            raise ValueError(
                "global-order search requires a feasible joint hint"
            )
        for child in selected:
            for parent, lag in projected[child].items():
                if hint_targets[child] < hint_targets[parent] + lag:
                    raise ValueError("joint hint violates precedence")
        compact = {job: rank for rank, job in enumerate(selected)}
        base_children: list[list[tuple[int, int]]] = [
            [] for _ in selected
        ]
        base_indegree = [0] * len(selected)
        for child in selected:
            right = compact[child]
            for parent, lag in projected[child].items():
                left = compact[parent]
                base_children[left].append((right, lag))
                base_indegree[right] += 1

        rng = random.Random(int(os.environ.get("RANDOM_SEED", "1")))
        orders: dict[str, list[int]] = {}
        for engine in engines:
            expected = {i for i in selected if ops[i].engine == engine}
            order = hint_orders.get(engine)
            if order is None:
                by_cycle: dict[int, list[int]] = {}
                for i in selected:
                    if ops[i].engine == engine:
                        by_cycle.setdefault(hint_targets[i], []).append(i)
                order = []
                for cycle in sorted(by_cycle):
                    batch = by_cycle[cycle]
                    rng.shuffle(batch)
                    order.extend(batch)
            if len(order) != len(expected) or set(order) != expected:
                raise ValueError(f"invalid explicit order for {engine}")
            orders[engine] = order

        def evaluate(
            candidate_orders: dict[str, list[int]],
        ) -> tuple[int, dict[int, int] | None, list[tuple[str, int]]]:
            children = [list(row) for row in base_children]
            indegree = list(base_indegree)
            resource_parent: dict[tuple[int, int], tuple[str, int]] = {}
            for engine, order in candidate_orders.items():
                capacity = SLOT_LIMITS[engine]
                for rank in range(len(order) - capacity):
                    left = compact[order[rank]]
                    right = compact[order[rank + capacity]]
                    children[left].append((right, 1))
                    indegree[right] += 1
                    resource_parent[(left, right)] = (engine, rank)

            starts = [earliest[i] for i in selected]
            reasons: list[tuple[int, str, int] | None] = [None] * len(selected)
            ready = deque(i for i, degree in enumerate(indegree) if not degree)
            seen = 0
            while ready:
                parent = ready.popleft()
                seen += 1
                for child, lag in children[parent]:
                    candidate = starts[parent] + lag
                    if candidate > starts[child]:
                        starts[child] = candidate
                        resource = resource_parent.get((parent, child))
                        reasons[child] = (
                            parent,
                            resource[0] if resource else "dep",
                            resource[1] if resource else -1,
                        )
                    indegree[child] -= 1
                    if not indegree[child]:
                        ready.append(child)
            if seen != len(selected):
                return 1_000_000_000, None, []
            terminal = max(
                range(len(selected)),
                key=lambda rank: starts[rank] + tail[selected[rank]],
            )
            span = starts[terminal] + tail[selected[terminal]] + 1
            critical: list[tuple[str, int]] = []
            node = terminal
            while reasons[node] is not None:
                parent, kind, rank = reasons[node]
                if kind != "dep":
                    critical.append((kind, rank))
                node = parent
            return (
                span,
                {job: starts[compact[job]] for job in selected},
                critical,
            )

        best_span, best_cycles, critical = evaluate(orders)
        current_span = best_span
        print(
            f"global_order_start={best_span} critical_resource_edges="
            f"{len(critical)}",
            flush=True,
        )
        best_orders = {engine: list(order) for engine, order in orders.items()}
        stall = 0
        random_candidates = int(
            os.environ.get("GLOBAL_ORDER_RANDOM_CANDIDATES", "64")
        )
        critical_radius = int(
            os.environ.get("GLOBAL_ORDER_CRITICAL_RADIUS", "8")
        )
        max_uphill = int(os.environ.get("GLOBAL_ORDER_MAX_UPHILL", "2"))
        for trial in range(global_order_trials):
            swaps: set[tuple[str, int]] = set()
            for engine, rank in critical:
                capacity = SLOT_LIMITS[engine]
                for anchor in (rank, rank + capacity):
                    for delta in range(-critical_radius, critical_radius + 1):
                        index = anchor + delta
                        if 0 <= index < len(orders[engine]) - 1:
                            swaps.add((engine, index))
            for _ in range(random_candidates):
                engine = rng.choice(engines)
                swaps.add(
                    (engine, rng.randrange(len(orders[engine]) - 1))
                )
            candidates = list(swaps)
            rng.shuffle(candidates)
            chosen: tuple[int, str, int, dict[int, int], list[tuple[str, int]]] | None = None
            for engine, index in candidates:
                order = orders[engine]
                order[index], order[index + 1] = order[index + 1], order[index]
                span, cycles, candidate_critical = evaluate(orders)
                order[index], order[index + 1] = order[index + 1], order[index]
                if cycles is None:
                    continue
                item = (span, engine, index, cycles, candidate_critical)
                if chosen is None or item[:3] < chosen[:3]:
                    chosen = item
            if chosen is None:
                print(f"global_order_no_feasible trial={trial}", flush=True)
                break
            span, engine, index, cycles, candidate_critical = chosen
            accept = span < current_span
            if not accept and stall > 8:
                accept = span <= best_span + max_uphill
            if not accept:
                stall += 1
                continue
            order = orders[engine]
            order[index], order[index + 1] = order[index + 1], order[index]
            current_span = span
            critical = candidate_critical
            stall = 0 if span < best_span else stall + 1
            if span < best_span:
                best_span = span
                best_cycles = cycles
                best_orders = {
                    name: list(value) for name, value in orders.items()
                }
                print(
                    f"global_order_best={best_span} trial={trial} "
                    f"swap={engine}:{index} critical="
                    f"{len(candidate_critical)}",
                    flush=True,
                )
        assert best_cycles is not None
        output_prefix = Path(
            os.environ.get("OUT_PREFIX", "/tmp/aopt-global-order")
        )
        for engine in engines:
            Path(f"{output_prefix}-{engine}.json").write_text(
                json.dumps(
                    {
                        "engine": engine,
                        "horizon": best_span,
                        "cycles": {
                            str(i): best_cycles[i]
                            for i in selected
                            if ops[i].engine == engine
                        },
                        "order": best_orders[engine],
                    }
                )
            )
        print(
            f"global_order_final={best_span} trials={global_order_trials} "
            f"output_prefix={output_prefix}",
            flush=True,
        )
        return

    # Convert a feasible multi-engine schedule into persistent virtual lanes.
    # Each lane is a simple precedence chain, so the earliest schedule of the
    # combined DAG is capacity-feasible without Cumulative/AllDifferent.  The
    # assignment of same-cycle operations to lanes is semantically free but
    # can substantially change the length of those chains; random matching is
    # a cheap way to remove accidental ordering slack before exact LNS.
    lane_trials = int(os.environ.get("SEARCH_LANE_ORDERS", "0"))
    if lane_trials:
        if set(hint_targets) != selected_set:
            raise ValueError(
                "lane search requires one joint hint cycle per selected job"
            )
        source_horizons = {
            int(payload.get("horizon", 0)) for payload in hint_payloads
        }
        source_horizons.discard(0)
        if len(source_horizons) != 1:
            raise ValueError("lane search hints must share one horizon")
        source_span = source_horizons.pop()
        hint_order = sorted(
            selected, key=lambda i: (hint_targets[i], position[i])
        )
        for child in selected:
            if not earliest[child] <= hint_targets[child] < source_span - tail[child]:
                raise ValueError(f"invalid lane-search hint bound for {child}")
            for parent, lag in projected[child].items():
                if hint_targets[child] < hint_targets[parent] + lag:
                    raise ValueError("lane-search hint violates precedence")
        rng = random.Random(int(os.environ.get("RANDOM_SEED", "1")))
        best_span = source_span + 1
        best_cycles: dict[int, int] | None = None
        for trial in range(lane_trials):
            resource_parent: dict[int, int] = {}
            for engine in engines:
                capacity = SLOT_LIMITS[engine]
                lane_tail = [-1] * capacity
                by_cycle: dict[int, list[int]] = {}
                for i in selected:
                    if ops[i].engine == engine:
                        by_cycle.setdefault(hint_targets[i], []).append(i)
                for cycle in sorted(by_cycle):
                    current = by_cycle[cycle]
                    if trial:
                        rng.shuffle(current)
                        lanes = rng.sample(range(capacity), len(current))
                    else:
                        current.sort(key=position.__getitem__)
                        lanes = list(range(len(current)))
                    for lane, i in zip(lanes, current):
                        if lane_tail[lane] >= 0:
                            resource_parent[i] = lane_tail[lane]
                        lane_tail[lane] = i

            lane_cycles: dict[int, int] = {}
            for child in hint_order:
                ready = earliest[child]
                for parent, lag in projected[child].items():
                    ready = max(ready, lane_cycles[parent] + lag)
                if child in resource_parent:
                    ready = max(
                        ready, lane_cycles[resource_parent[child]] + 1
                    )
                lane_cycles[child] = ready
            span = max(
                (lane_cycles[i] + tail[i] for i in selected), default=-1
            ) + 1
            if span < best_span:
                best_span = span
                best_cycles = lane_cycles
                print(f"lane_trial={trial} span={span}", flush=True)
        assert best_cycles is not None
        output_prefix = Path(
            os.environ.get("OUT_PREFIX", "/tmp/aopt-joint-lanes")
        )
        for engine in engines:
            Path(f"{output_prefix}-{engine}.json").write_text(
                json.dumps(
                    {
                        "engine": engine,
                        "horizon": best_span,
                        "cycles": {
                            str(i): best_cycles[i]
                            for i in selected
                            if ops[i].engine == engine
                        },
                    }
                )
            )
        print(
            f"lane_best={best_span} trials={lane_trials} "
            f"output_prefix={output_prefix}",
            flush=True,
        )
        return

    # Local multi-engine cycle deletion.  A feasible joint schedule is much
    # more valuable than independent per-engine hints because its resource
    # orders cannot form a cross-engine cycle.  Compress that incumbent one
    # cycle at a time: delete a candidate cycle, shift the suffix left, and
    # let CP-SAT reorder only a bounded neighborhood around the deletion.
    # Successful steps are persisted immediately and become the next
    # incumbent, so long searches never lose progress.
    if bool(int(os.environ.get("MULTI_BUBBLE_DELETE_REPAIR", "0"))):
        if set(hint_targets) != selected_set:
            missing = selected_set - set(hint_targets)
            extra = set(hint_targets) - selected_set
            raise ValueError(
                "bubble repair requires one joint hint cycle per selected "
                f"operation: missing={len(missing)} extra={len(extra)}"
            )
        source_horizons = {
            int(payload.get("horizon", 0)) for payload in hint_payloads
        }
        source_horizons.discard(0)
        if len(source_horizons) != 1:
            raise ValueError(
                "bubble repair hints must share one explicit horizon"
            )
        current_span = source_horizons.pop()
        current = dict(hint_targets)
        target_span = horizon
        window_radius = int(os.environ.get("BUBBLE_WINDOW_RADIUS", "64"))
        movement = int(os.environ.get("BUBBLE_MOVEMENT", "24"))
        local_mode = os.environ.get("BUBBLE_LOCAL_MODE", "window")
        conflict_radius = int(
            os.environ.get("BUBBLE_CONFLICT_RADIUS", "32")
        )
        conflict_hops = int(os.environ.get("BUBBLE_CONFLICT_HOPS", "1"))
        candidate_limit = int(os.environ.get("BUBBLE_CANDIDATES", "96"))
        per_candidate_limit = float(
            os.environ.get("BUBBLE_TIME_LIMIT", "3")
        )
        time_indexed_repair = bool(
            int(os.environ.get("BUBBLE_TIME_INDEXED", "0"))
        )
        max_steps = int(os.environ.get("BUBBLE_MAX_STEPS", "1000000"))
        requested_cycles = tuple(
            int(value)
            for value in os.environ.get("BUBBLE_DELETE_CYCLES", "").split(",")
            if value
        )
        output_prefix = Path(
            os.environ.get("OUT_PREFIX", "/tmp/aopt-joint-bubble")
        )

        def check_joint(cycles: dict[int, int], span: int) -> None:
            for i in selected:
                cycle = cycles[i]
                if cycle < earliest[i] or cycle + tail[i] >= span:
                    raise AssertionError(
                        (i, cycle, earliest[i], tail[i], span)
                    )
                for parent, lag in projected[i].items():
                    if cycle < cycles[parent] + lag:
                        raise AssertionError(
                            (parent, i, lag, cycles[parent], cycle)
                        )
            for engine in engines:
                usage = Counter(
                    cycles[i]
                    for i in selected
                    if ops[i].engine == engine
                )
                if max(usage.values(), default=0) > SLOT_LIMITS[engine]:
                    raise AssertionError(f"{engine} capacity overflow")

        def write_joint(cycles: dict[int, int], span: int) -> None:
            for engine in engines:
                output = Path(f"{output_prefix}-{engine}.json")
                output.write_text(
                    json.dumps(
                        {
                            "engine": engine,
                            "horizon": span,
                            "cycles": {
                                str(i): cycles[i]
                                for i in selected
                                if ops[i].engine == engine
                            },
                        }
                    )
                )

        check_joint(current, current_span)
        write_joint(current, current_span)
        print(
            f"bubble_start={current_span} target={target_span} "
            f"window={window_radius} movement={movement}",
            flush=True,
        )
        completed_steps = 0
        while current_span > target_span and completed_steps < max_steps:
            usage_by_engine = {
                engine: Counter(
                    current[i]
                    for i in selected
                    if ops[i].engine == engine
                )
                for engine in engines
            }
            if requested_cycles:
                candidates = tuple(
                    cycle
                    for cycle in requested_cycles
                    if 0 < cycle < current_span
                )
            else:
                # Sparse cycles are the cheapest deletion points.  Prefer
                # central ties so both the prefix and suffix have room to
                # absorb a changed resource order.
                candidates = tuple(
                    cycle
                    for _, _, cycle in sorted(
                        (
                            sum(
                                usage_by_engine[engine][cycle]
                                for engine in engines
                            ),
                            abs(2 * cycle - current_span),
                            cycle,
                        )
                        for cycle in range(
                            1,
                            min(current_span, max(current.values()) + 1),
                        )
                    )
                )
            repaired: dict[int, int] | None = None
            for attempt, deleted in enumerate(
                candidates[:candidate_limit], start=1
            ):
                new_span = current_span - 1
                model = cp_model.CpModel()
                base_cycles = {
                    i: current[i] if current[i] < deleted else current[i] - 1
                    for i in selected
                }
                if local_mode == "window":
                    local = {
                        i
                        for i in selected
                        if abs(current[i] - deleted) <= window_radius
                    }
                elif local_mode == "conflict":
                    # Seed the neighborhood with exactly the constraints the
                    # one-cycle shift breaks.  A broad time window releases
                    # hundreds of unrelated jobs and makes finding the first
                    # feasible repair unnecessarily hard.
                    local = {
                        i
                        for i in selected
                        if base_cycles[i] < earliest[i]
                        or base_cycles[i] + tail[i] >= new_span
                    }
                    for child in selected:
                        for parent, lag in projected[child].items():
                            if base_cycles[child] < base_cycles[parent] + lag:
                                local.add(parent)
                                local.add(child)
                    for engine in engines:
                        by_cycle: dict[int, list[int]] = {}
                        for i in selected:
                            if ops[i].engine == engine:
                                by_cycle.setdefault(base_cycles[i], []).append(i)
                        for cycle_ops in by_cycle.values():
                            if len(cycle_ops) > SLOT_LIMITS[engine]:
                                local.update(cycle_ops)

                    # Dependency closure catches the producers/consumers
                    # that can exchange slack with a violated edge.
                    for _ in range(conflict_hops):
                        expanded = set(local)
                        for child in selected:
                            parents = projected[child]
                            if child in local:
                                expanded.update(parents)
                            elif any(parent in local for parent in parents):
                                expanded.add(child)
                        local = expanded

                    # A saturated engine moves a hole by shifting a corridor
                    # of jobs.  Release same-engine jobs near each conflict,
                    # without also freeing unrelated jobs on the other unit.
                    anchors = {
                        engine: [
                            base_cycles[i]
                            for i in local
                            if ops[i].engine == engine
                        ]
                        for engine in engines
                    }
                    local.update(
                        i
                        for i in selected
                        if any(
                            abs(base_cycles[i] - anchor) <= conflict_radius
                            for anchor in anchors[ops[i].engine]
                        )
                    )
                else:
                    raise ValueError(f"unknown BUBBLE_LOCAL_MODE={local_mode!r}")
                starts: dict[int, cp_model.IntVar] = {}
                repair_bounds: dict[int, tuple[int, int]] = {}
                impossible = False
                for i in selected:
                    old = current[i]
                    base = base_cycles[i]
                    natural_lower = earliest[i]
                    natural_upper = new_span - 1 - tail[i]
                    if i in local:
                        lower = max(natural_lower, base - movement)
                        upper = min(natural_upper, base + movement)
                    else:
                        lower = upper = base
                    if lower > upper:
                        impossible = True
                        break
                    starts[i] = model.new_int_var(lower, upper, f"s{i}")
                    repair_bounds[i] = (lower, upper)
                if impossible:
                    continue
                for child in selected:
                    for parent, lag in projected[child].items():
                        model.add(starts[child] >= starts[parent] + lag)
                for engine in engines:
                    engine_ops = [
                        i for i in selected if ops[i].engine == engine
                    ]
                    if not time_indexed_repair:
                        intervals = [
                            model.new_fixed_size_interval_var(
                                starts[i], 1, f"i{i}"
                            )
                            for i in engine_ops
                        ]
                        model.add_cumulative(
                            intervals,
                            [1] * len(intervals),
                            SLOT_LIMITS[engine],
                        )
                        continue

                    # On a nearly saturated engine, Cumulative can spend a
                    # short LNS solve rediscovering the few remaining holes.
                    # Exposing exact residual capacity per cycle is larger,
                    # but propagates the hard early Flow/Load bubbles much
                    # more strongly.
                    local_engine_ops = [i for i in engine_ops if i in local]
                    fixed_usage = Counter(
                        repair_bounds[i][0]
                        for i in engine_ops
                        if i not in local
                    )
                    choices_by_cycle: dict[int, list[cp_model.IntVar]] = {}
                    for i in local_engine_ops:
                        lower, upper = repair_bounds[i]
                        choices = []
                        for cycle in range(lower, upper + 1):
                            choice = model.new_bool_var(f"x{i}_{cycle}")
                            choices.append(choice)
                            choices_by_cycle.setdefault(cycle, []).append(
                                choice
                            )
                            model.add_hint(
                                choice, int(cycle == base_cycles[i])
                            )
                        model.add_exactly_one(choices)
                        model.add(
                            starts[i]
                            == sum(
                                cycle * choice
                                for cycle, choice in zip(
                                    range(lower, upper + 1), choices
                                )
                            )
                        )
                    capacity = SLOT_LIMITS[engine]
                    for cycle, choices in choices_by_cycle.items():
                        residual = capacity - fixed_usage[cycle]
                        if residual < 0:
                            impossible = True
                            break
                        model.add(sum(choices) <= residual)
                    if impossible:
                        break
                if impossible:
                    continue
                for i, start in starts.items():
                    model.add_hint(start, base_cycles[i])
                repair_solver = cp_model.CpSolver()
                repair_solver.parameters.max_time_in_seconds = (
                    per_candidate_limit
                )
                repair_solver.parameters.num_workers = int(
                    os.environ.get("WORKERS", "8")
                )
                repair_solver.parameters.random_seed = (
                    int(os.environ.get("RANDOM_SEED", "1"))
                    + current_span * 1009
                    + deleted
                    + attempt
                )
                repair_solver.parameters.randomize_search = True
                # Repeated multi-worker solves with repair_hint trigger an
                # OR-Tools fixed-search assertion on some builds.  Ordinary
                # hints still guide the search; keep the unstable repair mode
                # opt-in for single-shot diagnostics only.
                repair_solver.parameters.repair_hint = bool(
                    int(os.environ.get("BUBBLE_REPAIR_HINT", "0"))
                )
                repair_status = repair_solver.solve(model)
                print(
                    f"bubble_try span={current_span} delete={deleted} "
                    f"local={len(local)} attempt={attempt} "
                    f"status={repair_solver.status_name(repair_status)}",
                    flush=True,
                )
                if repair_status not in (
                    cp_model.FEASIBLE,
                    cp_model.OPTIMAL,
                ):
                    continue
                repaired = {
                    i: repair_solver.value(starts[i]) for i in selected
                }
                check_joint(repaired, new_span)
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
            completed_steps += 1
            write_joint(current, current_span)
        print(
            f"bubble_final={current_span} steps={completed_steps} "
            f"output_prefix={output_prefix}",
            flush=True,
        )
        return

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
        fixed_cycles = {
            int(index): int(cycle)
            for index, cycle in payload["cycles"].items()
        }
        expected = {index for index in fixed_cycles if index in starts}
        raw_order = payload.get("order")
        if raw_order is None:
            ordered = sorted(
                expected, key=lambda index: (fixed_cycles[index], index)
            )
        else:
            ordered = [
                int(index) for index in raw_order if int(index) in starts
            ]
            if len(ordered) != len(expected) or set(ordered) != expected:
                raise ValueError(
                    f"invalid explicit order for fixed {hint_engine} hint"
                )
        cycle_counts = Counter(fixed_cycles[index] for index in ordered)
        if max(cycle_counts.values(), default=0) > capacity:
            raise ValueError(f"hint for {hint_engine} exceeds capacity")
        for parent, child in zip(ordered, ordered[capacity:]):
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
