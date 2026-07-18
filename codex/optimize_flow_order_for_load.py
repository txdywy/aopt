"""Optimize a feasible flow order against a load Hall-window cut."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random

import numpy as np

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
    horizon = int(os.environ.get("TARGET", "959"))
    fixed_left, fixed_right = (
        int(value)
        for value in os.environ.get("LOAD_WINDOW", "73:839").split(":")
    )
    global_hall = bool(int(os.environ.get("GLOBAL_HALL", "0")))
    span_target = int(os.environ.get("SPAN_TARGET", "0"))
    hall_engines = tuple(
        engine
        for engine in os.environ.get("HALL_ENGINES", "load").split(",")
        if engine
    )
    order_engine = os.environ.get("ORDER_ENGINE", "flow")
    order_capacity = SLOT_LIMITS[order_engine]

    full_children: list[list[tuple[int, int]]] = [[] for _ in ops]
    full_earliest = [0] * len(ops)
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            full_children[parent].append((child, lag))
            full_earliest[child] = max(
                full_earliest[child], full_earliest[parent] + lag
            )
    full_tail = [0] * len(ops)
    for parent in reversed(range(len(ops))):
        full_tail[parent] = max(
            (lag + full_tail[child] for child, lag in full_children[parent]),
            default=0,
        )

    # Project away every engine that does not participate in this search.
    # A weighted edge between retained operations represents the longest path
    # through all omitted operations.  FLOW+LOAD has only ~2.8k vertices,
    # making each local-search evaluation about seven times cheaper than a
    # traversal of the complete ~20k-op DAG without changing any windows.
    selected_engines = frozenset((order_engine, *hall_engines))
    selected = [
        i for i, op in enumerate(ops) if op.engine in selected_engines
    ]
    selected_set = set(selected)
    local_of = {global_index: local for local, global_index in enumerate(selected)}
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

    count = len(selected)
    base_children: list[list[tuple[int, int]]] = [[] for _ in selected]
    base_parent_lag: list[dict[int, int]] = [dict() for _ in selected]
    base_indegree = [0] * count
    for child_global in selected:
        child = local_of[child_global]
        for parent_global, lag in projected[child_global].items():
            parent = local_of[parent_global]
            base_parent_lag[child][parent] = lag
            base_children[parent].append((child, lag))
            base_indegree[child] += 1
    base_earliest = [full_earliest[i] for i in selected]
    base_latest = [horizon - 1 - full_tail[i] for i in selected]
    engine_ops = {
        engine: [
            local_of[i]
            for i, op in enumerate(ops)
            if op.engine == engine
        ]
        for engine in hall_engines
    }
    selected_tail = [full_tail[i] for i in selected]

    def ordered_span(earliest: list[int]) -> int:
        return max(
            (
                earliest[local] + selected_tail[local]
                for local in range(count)
            ),
            default=0,
        ) + 1

    def worst_hall_cut(
        earliest: list[int], latest: list[int], engine: str
    ) -> tuple[int, int, int, int, int, int]:
        """Return overload, cut, trapped count, margin, and pressure."""
        indices = engine_ops[engine]
        matrix = np.zeros((horizon, horizon), dtype=np.int16)
        np.add.at(
            matrix,
            (
                np.fromiter((earliest[i] for i in indices), dtype=np.int32),
                np.fromiter((latest[i] for i in indices), dtype=np.int32),
            ),
            1,
        )
        contained = np.cumsum(
            np.cumsum(matrix[::-1], axis=0, dtype=np.int32)[::-1],
            axis=1,
            dtype=np.int32,
        )
        capacity = SLOT_LIMITS[engine]
        best_overload = -10**9
        best_left = best_right = 0
        for left in range(horizon):
            row = contained[left, left:]
            overloads = row - capacity * np.arange(
                1, horizon - left + 1, dtype=np.int32
            )
            offset = int(np.argmax(overloads))
            overload = int(overloads[offset])
            if overload > best_overload:
                best_overload = overload
                best_left = left
                best_right = left + offset
        trapped = [
            i
            for i in indices
            if earliest[i] >= best_left and latest[i] <= best_right
        ]
        margin = sum(
            min(
                earliest[i] - best_left + 1,
                best_right - latest[i] + 1,
            )
            for i in trapped
        )
        pressure = sum(
            max(0, earliest[i] - best_left + 1)
            * max(0, best_right - latest[i] + 1)
            for i in indices
        )
        return (
            best_overload,
            best_left,
            best_right,
            len(trapped),
            margin,
            pressure,
        )

    def score_hall_cut(
        earliest: list[int],
        latest: list[int],
        engine: str,
        left: int,
        right: int,
    ) -> tuple[int, int, int, int, int, int]:
        indices = engine_ops[engine]
        trapped = [
            i
            for i in indices
            if earliest[i] >= left and latest[i] <= right
        ]
        overload = len(trapped) - SLOT_LIMITS[engine] * (right - left + 1)
        margin = sum(
            min(earliest[i] - left + 1, right - latest[i] + 1)
            for i in trapped
        )
        pressure = sum(
            max(0, earliest[i] - left + 1)
            * max(0, right - latest[i] + 1)
            for i in indices
        )
        return overload, left, right, len(trapped), margin, pressure

    payload = json.loads(Path(os.environ["FLOW_HINT"]).read_text())
    flow_cycles = {int(i): int(cycle) for i, cycle in payload["cycles"].items()}
    expected = {
        i for i, op in enumerate(ops) if op.engine == order_engine
    }
    if set(flow_cycles) != expected:
        raise ValueError(
            f"FLOW_HINT does not match the target {order_engine} DAG"
        )
    order = sorted(expected, key=lambda i: (flow_cycles[i], i))
    for raw_move in os.environ.get("INITIAL_BLOCK_MOVES", "").split(","):
        if not raw_move:
            continue
        group, round_index, insertion = map(int, raw_move.split(":"))
        block = [
            node
            for node in order
            if (ops[node].group, ops[node].round) == (group, round_index)
        ]
        if not block:
            raise ValueError(f"empty initial FLOW block {group}:{round_index}")
        block_set = set(block)
        remaining = [node for node in order if node not in block_set]
        insertion = min(len(remaining), max(0, insertion))
        order = remaining[:insertion] + block + remaining[insertion:]
    for raw_move in os.environ.get("INITIAL_NODE_MOVES", "").split(","):
        if not raw_move:
            continue
        node, insertion = map(int, raw_move.split(":"))
        if node not in order:
            raise ValueError(f"missing initial FLOW node {node}")
        order.remove(node)
        insertion = min(len(order), max(0, insertion))
        order.insert(insertion, node)

    active_cut_history: list[tuple[str, int, int]] = []

    def evaluate(candidate_order: list[int], *, exact_hall: bool = True):
        extra_children: list[list[int]] = [[] for _ in selected]
        indegree = base_indegree.copy()
        for previous_global, current_global in zip(
            candidate_order, candidate_order[order_capacity:]
        ):
            previous = local_of[previous_global]
            current = local_of[current_global]
            if base_parent_lag[current].get(previous, 0) < 1:
                extra_children[previous].append(current)
                indegree[current] += 1

        ready = [i for i, degree in enumerate(indegree) if not degree]
        topological: list[int] = []
        earliest = base_earliest.copy()
        while ready:
            parent = ready.pop()
            topological.append(parent)
            start = earliest[parent]
            for child, lag in base_children[parent]:
                earliest[child] = max(earliest[child], start + lag)
                indegree[child] -= 1
                if not indegree[child]:
                    ready.append(child)
            for child in extra_children[parent]:
                earliest[child] = max(earliest[child], start + 1)
                indegree[child] -= 1
                if not indegree[child]:
                    ready.append(child)
        if len(topological) != count or max(earliest) >= horizon:
            return None

        latest = base_latest.copy()
        for parent in reversed(topological):
            upper = latest[parent]
            for child, lag in base_children[parent]:
                upper = min(upper, latest[child] - lag)
            for child in extra_children[parent]:
                upper = min(upper, latest[child] - 1)
            latest[parent] = upper
        if any(earliest[i] > latest[i] for i in range(count)):
            return None
        dag_span = ordered_span(earliest)

        if global_hall:
            cuts = (
                tuple(
                    worst_hall_cut(earliest, latest, engine)
                    for engine in hall_engines
                )
                if exact_hall
                else tuple(
                    score_hall_cut(earliest, latest, engine, left, right)
                    for engine, left, right in active_cut_history
                )
            )
            overloads = tuple(cut[0] for cut in cuts)
            hall_key = (
                max(overloads),
                sum(max(0, overload) for overload in overloads),
                sum(overloads),
                sum(cut[4] for cut in cuts),
                sum(cut[5] for cut in cuts),
            )
        else:
            cut = score_hall_cut(
                earliest,
                latest,
                hall_engines[0],
                fixed_left,
                fixed_right,
            )
            overload, _, _, _, margin, boundary_pressure = cut
            hall_key = (overload, margin, boundary_pressure)
            cuts = (cut,)
        key = (
            (
                max(0, dag_span - span_target),
                *hall_key,
                dag_span,
            )
            if span_target
            else hall_key
        )
        return key, earliest, latest, cuts

    rng = random.Random(int(os.environ.get("RANDOM_SEED", "1")))
    iterations = int(os.environ.get("ORDER_ITERATIONS", "300"))
    candidates = int(os.environ.get("ORDER_CANDIDATES", "100"))
    radius = int(os.environ.get("ORDER_RADIUS", "256"))
    compound_moves = int(os.environ.get("COMPOUND_MOVES", "1"))
    anneal_overload = int(os.environ.get("ANNEAL_OVERLOAD", "0"))
    output = Path(os.environ.get("OUT", "/tmp/aopt-flow-load-hall.json"))

    result = evaluate(order)
    if result is None:
        raise ValueError("initial flow order is invalid")
    current_key, current_earliest, current_latest, current_cuts = result
    if global_hall:
        active_cut_history.extend(
            (engine, cut[1], cut[2])
            for engine, cut in zip(hall_engines, current_cuts)
        )
    best_key = current_key
    best_order = order.copy()
    best_earliest = current_earliest
    print(f"hall_start={current_key}", flush=True)

    def save() -> None:
        output.write_text(
            json.dumps(
                {
                    "engine": order_engine,
                    "horizon": horizon,
                    "cycles": {
                        str(i): best_earliest[local_of[i]] for i in expected
                    },
                    "order": best_order,
                    "hall_engines": hall_engines,
                    "hall_cuts": best_cuts,
                    "hall_overload": best_key[0],
                }
            )
        )

    best_cuts = current_cuts
    save()

    def target_reached(key: tuple[int, ...]) -> bool:
        if span_target:
            # With a span target the leading component is schedule-span
            # excess, followed by the Hall overload.  Stopping on key[0]
            # alone incorrectly treats every already-short-enough schedule as
            # Hall-feasible after its first secondary-score improvement.
            return key[0] <= 0 and key[1] <= 0
        return key[0] <= 0

    block_scan_passes = int(os.environ.get("BLOCK_SCAN_PASSES", "0"))
    if block_scan_passes:
        block_rounds = frozenset(
            int(value)
            for value in os.environ.get(
                "BLOCK_SCAN_ROUNDS", "1,2,3,4,12,13,14,15"
            ).split(",")
            if value
        )
        block_stride = int(os.environ.get("BLOCK_SCAN_STRIDE", "4"))
        exact_limit = int(os.environ.get("BLOCK_SCAN_EXACT_LIMIT", "32"))
        for scan_pass in range(block_scan_passes):
            ranking_current = (
                evaluate(order, exact_hall=False)[0]
                if global_hall
                else current_key
            )
            blocks: dict[tuple[int, int], list[int]] = {}
            for node in order:
                op = ops[node]
                key = (op.group, op.round)
                if (
                    op.group is not None
                    and op.group >= 0
                    and op.round in block_rounds
                ):
                    blocks.setdefault(key, []).append(node)
            position = {node: p for p, node in enumerate(order)}
            ranked_blocks = []
            attempted = feasible = 0
            for (group, rnd), block in blocks.items():
                member_set = set(block)
                first = min(position[node] for node in block)
                last = max(position[node] for node in block)
                # Opening a load window means launching its early FLOW block
                # sooner or retiring its late FLOW block later.  Include both
                # directions for middle blocks because an active Hall cut can
                # be rooted on either side of them.
                if rnd <= 4:
                    insertions = range(0, first + 1, block_stride)
                elif rnd >= 12:
                    insertions = range(last + 1, len(order) + 1, block_stride)
                else:
                    insertions = range(0, len(order) + 1, block_stride)
                remaining = [
                    node for node in order if node not in member_set
                ]
                for raw_insertion in insertions:
                    attempted += 1
                    removed_before = sum(
                        position[node] < raw_insertion for node in block
                    )
                    insertion = min(
                        len(remaining),
                        max(0, raw_insertion - removed_before),
                    )
                    trial = (
                        remaining[:insertion]
                        + block
                        + remaining[insertion:]
                    )
                    if trial == order:
                        continue
                    candidate = evaluate(trial, exact_hall=not global_hall)
                    if candidate is None:
                        continue
                    feasible += 1
                    ranked_blocks.append(
                        (
                            candidate[0],
                            group,
                            rnd,
                            raw_insertion,
                            candidate,
                            trial,
                        )
                    )
            ranked_blocks.sort(key=lambda item: item[:4])
            if global_hall:
                exact_ranked = []
                for item in ranked_blocks[:exact_limit]:
                    exact_candidate = evaluate(item[5], exact_hall=True)
                    if exact_candidate is not None:
                        exact_ranked.append(
                            (
                                exact_candidate[0],
                                item[1],
                                item[2],
                                item[3],
                                exact_candidate,
                                item[5],
                            )
                        )
                exact_ranked.sort(key=lambda item: item[:4])
                ranked_blocks = exact_ranked
            if not ranked_blocks or ranked_blocks[0][0] >= current_key:
                approximate = (
                    ranked_blocks[0][0] if ranked_blocks else None
                )
                print(
                    f"block_scan_no_improvement pass={scan_pass} "
                    f"attempted={attempted} feasible={feasible} "
                    f"best={approximate} current={current_key} "
                    f"ranking_current={ranking_current}",
                    flush=True,
                )
                break
            (
                _,
                group,
                rnd,
                insertion,
                candidate,
                trial,
            ) = ranked_blocks[0]
            order = trial
            (
                current_key,
                current_earliest,
                current_latest,
                current_cuts,
            ) = candidate
            for engine, cut in zip(hall_engines, current_cuts):
                cut_spec = (engine, cut[1], cut[2])
                if cut_spec not in active_cut_history:
                    active_cut_history.append(cut_spec)
            if current_key < best_key:
                best_key = current_key
                best_order = order.copy()
                best_earliest = current_earliest
                best_cuts = current_cuts
                save()
            print(
                f"block_scan_best={current_key} pass={scan_pass} "
                f"block={group}:{rnd} insertion={insertion} "
                f"attempted={attempted} feasible={feasible}",
                flush=True,
            )
            if target_reached(best_key):
                break

    ejection_passes = int(os.environ.get("EJECTION_SCAN_PASSES", "0"))
    if ejection_passes:
        from_min = int(os.environ.get("EJECTION_FROM_MIN", "0"))
        from_max = int(
            os.environ.get("EJECTION_FROM_MAX", str(len(order) - 1))
        )
        to_min = int(os.environ.get("EJECTION_TO_MIN", str(len(order) - 1)))
        to_max = int(os.environ.get("EJECTION_TO_MAX", str(len(order))))
        stride = int(os.environ.get("EJECTION_STRIDE", "1"))
        allow_backward = bool(
            int(os.environ.get("EJECTION_ALLOW_BACKWARD", "0"))
        )
        print_feasible = bool(
            int(os.environ.get("EJECTION_PRINT_FEASIBLE", "0"))
        )
        for scan_pass in range(ejection_passes):
            scan_best = None
            attempted = feasible = 0
            upper_from = min(from_max, len(order) - 1)
            upper_to = min(to_max, len(order))
            for source_position in range(
                max(0, from_min), upper_from + 1, stride
            ):
                node = order[source_position]
                for insertion in range(
                    (
                        max(0, to_min)
                        if allow_backward
                        else max(to_min, source_position + 1)
                    ),
                    upper_to + 1,
                    stride,
                ):
                    if insertion in (
                        source_position,
                        source_position + 1,
                    ):
                        continue
                    attempted += 1
                    trial = order.copy()
                    trial.pop(source_position)
                    trial.insert(min(insertion, len(trial)), node)
                    candidate = evaluate(trial, exact_hall=False)
                    if candidate is None:
                        continue
                    feasible += 1
                    if print_feasible:
                        print(
                            f"ejection_feasible pass={scan_pass} "
                            f"move={source_position}:{insertion} "
                            f"key={candidate[0]}",
                            flush=True,
                        )
                    item = (
                        candidate[0],
                        source_position,
                        insertion,
                        candidate,
                        trial,
                    )
                    if scan_best is None or item[:3] < scan_best[:3]:
                        scan_best = item
            if scan_best is None:
                print(
                    f"ejection_stuck pass={scan_pass} "
                    f"attempted={attempted} feasible={feasible}",
                    flush=True,
                )
                break
            _, source_position, insertion, _, trial = scan_best
            exact_candidate = evaluate(trial, exact_hall=True)
            if exact_candidate is None:
                raise AssertionError("ejection candidate changed feasibility")
            if exact_candidate[0] >= current_key:
                print(
                    f"ejection_no_improvement pass={scan_pass} "
                    f"best={exact_candidate[0]} current={current_key} "
                    f"attempted={attempted} feasible={feasible}",
                    flush=True,
                )
                break
            order = trial
            (
                current_key,
                current_earliest,
                current_latest,
                current_cuts,
            ) = exact_candidate
            for engine, cut in zip(hall_engines, current_cuts):
                cut_spec = (engine, cut[1], cut[2])
                if cut_spec not in active_cut_history:
                    active_cut_history.append(cut_spec)
            if current_key < best_key:
                best_key = current_key
                best_order = order.copy()
                best_earliest = current_earliest
                best_cuts = current_cuts
                save()
            print(
                f"ejection_best={current_key} pass={scan_pass} "
                f"move={source_position}:{insertion} "
                f"attempted={attempted} feasible={feasible}",
                flush=True,
            )

    stagnant = 0
    for iteration in range(iterations):
        ranking_current = (
            evaluate(order, exact_hall=False)[0]
            if global_hall
            else current_key
        )
        # The cut boundaries correspond approximately to these flow-order
        # positions; sample them heavily while retaining global mutations.
        focus = [
            p for p, node in enumerate(order)
            if any(
                left - 96 <= current_earliest[local_of[node]] <= left + 160
                or right - 160 <= current_earliest[local_of[node]] <= right + 96
                for _, left, right, *_ in current_cuts
            )
        ]
        current_span = ordered_span(current_earliest)
        if span_target and current_span > span_target:
            focus.extend(
                p
                for p, node in enumerate(order)
                if current_earliest[local_of[node]] + full_tail[node]
                >= current_span - 4
            )
            focus = sorted(set(focus))
        if not focus:
            focus = list(range(len(order)))
        mutations: set[tuple[tuple[str, int, int], ...]] = set()
        strides = (1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256)
        for p in focus[::max(1, len(focus) // 40)]:
            for delta in strides:
                for q in (p - delta, p + delta):
                    if 0 <= q < len(order):
                        mutations.add((("insert", order[p], order[q]),))
                        mutations.add((("swap", order[p], order[q]),))
                        if ops[order[p]].group is not None and ops[order[p]].group >= 0:
                            mutations.add((("block", order[p], order[q]),))
        # Always add randomized compound moves.  The deterministic boundary
        # grid already contains thousands of singles, so a ``while len < N``
        # guard would otherwise prevent every compound mutation from ever
        # entering the candidate pool.
        for _ in range(2 * candidates):
            mutation = []
            for _ in range(rng.randint(1, max(1, compound_moves))):
                p = rng.choice(focus)
                q = max(
                    0,
                    min(len(order) - 1, p + rng.randint(-radius, radius)),
                )
                if p == q:
                    continue
                kind = rng.choice(("insert", "insert", "swap", "block", "block"))
                mutation.append((kind, order[p], order[q]))
            if mutation:
                mutations.add(tuple(mutation))

        ranked = []
        mutation_list = list(mutations)
        rng.shuffle(mutation_list)
        attempted = 0
        for mutation in mutation_list:
            if len(ranked) >= candidates or attempted >= 20 * candidates:
                break
            attempted += 1
            trial = order.copy()
            descriptions = []
            for kind, anchor, target in mutation:
                a = trial.index(anchor)
                b = trial.index(target)
                descriptions.append(f"{kind}={a}:{b}")
                if kind == "swap":
                    trial[a], trial[b] = trial[b], trial[a]
                elif kind == "insert":
                    node = trial.pop(a)
                    b = trial.index(target)
                    trial.insert(b, node)
                    continue
                else:
                    key = (ops[anchor].group, ops[anchor].round)
                    if key[0] is None or key[0] < 0:
                        continue
                    block = [
                        node
                        for node in trial
                        if (ops[node].group, ops[node].round) == key
                    ]
                    block_set = set(block)
                    remaining = [node for node in trial if node not in block_set]
                    if target in block_set:
                        continue
                    insertion = remaining.index(target)
                    trial = (
                        remaining[:insertion] + block + remaining[insertion:]
                    )
            if trial == order:
                continue
            candidate = evaluate(trial, exact_hall=not global_hall)
            if candidate is not None:
                ranked.append(
                    (candidate[0], ",".join(descriptions), candidate, trial)
                )
        if not ranked:
            print(f"hall_stuck iteration={iteration}", flush=True)
            break
        ranked.sort(key=lambda item: item[0])
        if ranked[0][0] < ranking_current:
            chosen = ranked[0]
            stagnant = 0
        else:
            stagnant += 1
            temperature = max(
                0,
                round(
                    anneal_overload
                    * (iterations - iteration)
                    / max(1, iterations)
                ),
            )
            pool = [
                item for item in ranked[:32]
                if item[0][0] <= best_key[0] + temperature
            ]
            chosen = rng.choice((pool or ranked)[:4])
        _, description, candidate, trial = chosen
        if global_hall:
            exact_candidate = evaluate(trial, exact_hall=True)
            if exact_candidate is None:
                raise AssertionError("approximate candidate changed feasibility")
            candidate = exact_candidate
            for engine, cut in zip(hall_engines, candidate[3]):
                cut_spec = (engine, cut[1], cut[2])
                if cut_spec not in active_cut_history:
                    active_cut_history.append(cut_spec)
        order = trial
        (
            current_key,
            current_earliest,
            current_latest,
            current_cuts,
        ) = candidate
        if current_key < best_key:
            best_key = current_key
            best_order = order.copy()
            best_earliest = current_earliest
            best_cuts = current_cuts
            save()
            print(
                f"hall_best={best_key} cuts={best_cuts} "
                f"iteration={iteration} {description}",
                flush=True,
            )
            if target_reached(best_key):
                break
        elif iteration % 10 == 0:
            print(
                f"hall_walk={current_key} best={best_key} iteration={iteration}",
                flush=True,
            )

    save()
    print(f"hall_final={best_key} output={output}", flush=True)


if __name__ == "__main__":
    main()
