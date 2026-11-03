"""Search a unary FLOW order that leaves LOAD Hall windows schedulable.

The independent FLOW problem has only fourteen holes at a 959 cycle
horizon.  Consequently, fixing any feasible FLOW order almost completely
fixes the times of all FLOW operations.  A perfectly good FLOW-only order can
then trap too many LOAD operations in a time interval, making the joint
problem impossible before the LOAD scheduler even considers an ordering.

This search keeps the compact unary resource-order representation, but
evaluates every accepted order against the complete kernel DAG.  Its primary
objective is the maximum LOAD Hall overload.  The sum of all positive Hall
overloads and total LOAD slack provide a smooth signal while the identity of
the worst interval changes.
"""

from __future__ import annotations

from collections import Counter
import heapq
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
    count = len(ops)
    horizon = int(os.environ.get("TARGET", "959"))
    load_capacity = SLOT_LIMITS["load"]

    base_parents = [dict(op.parents) for op in ops]
    base_children: list[list[tuple[int, int]]] = [[] for _ in ops]
    base_indegree = [len(item) for item in base_parents]
    for child, parents in enumerate(base_parents):
        for parent, lag in parents.items():
            base_children[parent].append((child, lag))

    flow_nodes = [i for i, op in enumerate(ops) if op.engine == "flow"]
    flow_set = set(flow_nodes)
    load_nodes = np.fromiter(
        (i for i, op in enumerate(ops) if op.engine == "load"),
        dtype=np.int32,
    )

    payload = json.loads(Path(os.environ["FLOW_HINT"]).read_text())
    raw_cycles = {int(i): int(cycle) for i, cycle in payload["cycles"].items()}
    if set(raw_cycles) != flow_set:
        raise ValueError("FLOW_HINT does not match the target DAG")
    if "order" in payload:
        order = [int(i) for i in payload["order"]]
        if set(order) != flow_set or len(order) != len(flow_nodes):
            raise ValueError("invalid FLOW resource order")
    else:
        order = sorted(flow_nodes, key=lambda i: (raw_cycles[i], i))

    # Capacity is constant and the horizon is tiny, so these arrays are
    # cheaper to reuse conceptually than a general interval-Hall algorithm.
    left_index = np.arange(horizon, dtype=np.int32)[:, None]
    right_index = np.arange(horizon, dtype=np.int32)[None, :]
    interval_capacity = load_capacity * (right_index - left_index + 1)
    valid_intervals = right_index >= left_index

    def evaluate(candidate_order: list[int]):
        resource_parent = [-1] * count
        resource_children: list[list[int]] = [[] for _ in ops]
        indegree = base_indegree.copy()
        for parent, child in zip(candidate_order, candidate_order[1:]):
            if base_parents[child].get(parent, 0) < 1:
                resource_parent[child] = parent
                resource_children[parent].append(child)
                indegree[child] += 1

        ready = [i for i, degree in enumerate(indegree) if not degree]
        heapq.heapify(ready)
        topological: list[int] = []
        earliest = [0] * count
        early_reason = [-1] * count
        while ready:
            parent = heapq.heappop(ready)
            topological.append(parent)
            parent_start = earliest[parent]
            for child, lag in base_children[parent]:
                value = parent_start + lag
                if value > earliest[child]:
                    earliest[child] = value
                    early_reason[child] = parent
                indegree[child] -= 1
                if not indegree[child]:
                    heapq.heappush(ready, child)
            for child in resource_children[parent]:
                value = parent_start + 1
                if value > earliest[child]:
                    earliest[child] = value
                    early_reason[child] = parent
                indegree[child] -= 1
                if not indegree[child]:
                    heapq.heappush(ready, child)
        if len(topological) != count:
            return None

        full_score = max(earliest) + 1
        if full_score > horizon:
            return None

        latest = [horizon - 1] * count
        late_reason = [-1] * count
        for parent in reversed(topological):
            for child, lag in base_children[parent]:
                value = latest[child] - lag
                if value < latest[parent]:
                    latest[parent] = value
                    late_reason[parent] = child
            for child in resource_children[parent]:
                value = latest[child] - 1
                if value < latest[parent]:
                    latest[parent] = value
                    late_reason[parent] = child
        if any(earliest[i] > latest[i] for i in range(count)):
            return None

        load_earliest = np.fromiter(
            (earliest[int(i)] for i in load_nodes), dtype=np.int32
        )
        load_latest = np.fromiter(
            (latest[int(i)] for i in load_nodes), dtype=np.int32
        )
        matrix = np.zeros((horizon, horizon), dtype=np.int16)
        np.add.at(matrix, (load_earliest, load_latest), 1)
        contained = np.cumsum(
            np.cumsum(matrix[::-1], axis=0, dtype=np.int32)[::-1],
            axis=1,
            dtype=np.int32,
        )
        hall = contained - interval_capacity
        hall = np.where(valid_intervals, hall, -1_000_000)
        flat_index = int(np.argmax(hall))
        left, right = np.unravel_index(flat_index, hall.shape)
        overload = int(hall[left, right])
        positive = np.maximum(hall, 0, dtype=np.int32)
        overload_area = int(np.sum(positive * positive, dtype=np.int64))
        slack = int(np.sum(load_latest - load_earliest, dtype=np.int64))
        # Minimize this tuple.  Slack and full-DAG score only break ties after
        # all Hall violations have been reduced.
        key = (overload, overload_area, -slack, full_score)

        trapped = [
            int(i)
            for i, early, late in zip(load_nodes, load_earliest, load_latest)
            if early >= left and late <= right
        ]

        def flow_root(start: int, reasons: list[int]) -> int:
            node = start
            seen: set[int] = set()
            while node >= 0 and node not in seen:
                seen.add(node)
                if node in flow_set:
                    return node
                node = reasons[node]
            return -1

        early_roots = Counter(
            flow_root(i, early_reason) for i in trapped
        )
        late_roots = Counter(
            flow_root(i, late_reason) for i in trapped
        )
        early_roots.pop(-1, None)
        late_roots.pop(-1, None)
        return {
            "key": key,
            "earliest": earliest,
            "latest": latest,
            "cut": (int(left), int(right)),
            "trapped": len(trapped),
            "early_roots": early_roots,
            "late_roots": late_roots,
        }

    rng = random.Random(int(os.environ.get("RANDOM_SEED", "1")))
    iterations = int(os.environ.get("HALL_ITERATIONS", "200"))
    candidate_limit = int(os.environ.get("HALL_CANDIDATES", "256"))
    radius = int(os.environ.get("HALL_RADIUS", "96"))
    plateau = int(os.environ.get("HALL_PLATEAU", "16"))
    output = Path(os.environ.get("OUT", "/tmp/aopt-flow-load-hall.json"))

    current = evaluate(order)
    if current is None:
        raise ValueError("initial FLOW order does not fit TARGET")
    best = current
    best_order = order.copy()

    def save(result, saved_order: list[int]) -> None:
        cycles = {str(i): result["earliest"][i] for i in flow_nodes}
        usage = Counter(cycles.values())
        if max(usage.values(), default=0) > 1:
            raise AssertionError("FLOW order schedule exceeds capacity")
        output.write_text(
            json.dumps(
                {
                    "engine": "flow",
                    "horizon": horizon,
                    "cycles": cycles,
                    "order": saved_order,
                    "load_hall_overload": result["key"][0],
                    "load_hall_cut": result["cut"],
                }
            )
        )

    save(best, best_order)
    print(
        f"hall_start key={current['key']} cut={current['cut']} "
        f"trapped={current['trapped']} flow_jobs={len(flow_nodes)}",
        flush=True,
    )

    stagnant = 0
    strides = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)

    def mutate(
        source: list[int], kind: str, a: int, b: int
    ) -> list[int]:
        candidate = source.copy()
        if kind == "swap":
            candidate[a], candidate[b] = candidate[b], candidate[a]
        elif kind == "insert":
            node = candidate.pop(a)
            candidate.insert(b, node)
        elif kind == "batch":
            root = a
            root_op = ops[root]
            members = [
                node
                for node in candidate
                if ops[node].group == root_op.group
                and ops[node].round == root_op.round
                and ops[node].tag == root_op.tag
            ]
            member_set = set(members)
            removed_before = sum(
                1
                for position, node in enumerate(candidate)
                if position < b and node in member_set
            )
            candidate = [node for node in candidate if node not in member_set]
            target = max(0, min(len(candidate), b - removed_before))
            candidate[target:target] = members
        else:
            raise ValueError(f"unknown mutation {kind!r}")
        return candidate

    for iteration in range(iterations):
        position = {node: p for p, node in enumerate(order)}
        mutations: set[tuple[str, int, int]] = set()

        early_focus = [
            node for node, _ in current["early_roots"].most_common(32)
        ]
        late_focus = [
            node for node, _ in current["late_roots"].most_common(32)
        ]
        for node in early_focus:
            p = position[node]
            for delta in strides:
                q = p - delta
                if q >= 0:
                    mutations.add(("insert", p, q))
                    mutations.add(("swap", min(p, q), max(p, q)))
                    mutations.add(("batch", node, q))
        for node in late_focus:
            p = position[node]
            for delta in strides:
                q = p + delta
                if q < len(order):
                    mutations.add(("insert", p, q))
                    mutations.add(("swap", min(p, q), max(p, q)))
                    mutations.add(("batch", node, q))

        # The worst cut can remain unchanged for several moves.  Random
        # root-directed moves diversify the exact deterministic candidates.
        focus = early_focus + late_focus
        while focus and len(mutations) < candidate_limit * 2:
            node = rng.choice(focus)
            p = position[node]
            direction = -1 if node in current["early_roots"] else 1
            q = max(
                0,
                min(len(order) - 1, p + direction * rng.randint(1, radius)),
            )
            if p != q:
                choice = rng.randrange(6)
                if choice == 0:
                    mutations.add(("swap", min(p, q), max(p, q)))
                elif choice == 1:
                    mutations.add(("batch", node, q))
                else:
                    mutations.add(("insert", p, q))

        ranked = []
        mutation_list = list(mutations)
        rng.shuffle(mutation_list)
        for kind, a, b in mutation_list[:candidate_limit]:
            candidate_order = mutate(order, kind, a, b)
            candidate = evaluate(candidate_order)
            if candidate is not None:
                ranked.append(
                    (candidate["key"], kind, a, b, candidate, candidate_order)
                )

        if not ranked:
            print(f"hall_stuck iteration={iteration} no_feasible_move", flush=True)
            break
        ranked.sort(key=lambda item: item[0])
        improving = [item for item in ranked if item[0] < current["key"]]
        if improving:
            chosen = improving[0]
            stagnant = 0
        else:
            stagnant += 1
            if stagnant > plateau:
                print(
                    f"hall_plateau iteration={iteration} key={current['key']}",
                    flush=True,
                )
                break
            # A neutral walk changes which moves are valid without accepting
            # a worse maximum Hall violation.
            neutral = [
                item
                for item in ranked[:64]
                if item[0][0] == current["key"][0]
                and item[0][1] <= current["key"][1]
            ]
            escape_overload = int(os.environ.get("HALL_ESCAPE_OVERLOAD", "2"))
            escape_area = float(os.environ.get("HALL_ESCAPE_AREA", "1.03"))
            escape = [
                item
                for item in ranked[:128]
                if item[0][0] <= best["key"][0] + escape_overload
                and item[0][1] <= int(current["key"][1] * escape_area)
            ]
            if not neutral and not escape:
                print(
                    f"hall_local_min iteration={iteration} key={current['key']}",
                    flush=True,
                )
                break
            chosen = rng.choice((neutral or escape)[:16])

        _, kind, a, b, candidate, candidate_order = chosen
        order = candidate_order
        current = candidate
        if current["key"] < best["key"]:
            best = current
            best_order = order.copy()
            save(best, best_order)
            print(
                f"hall_best key={best['key']} cut={best['cut']} "
                f"trapped={best['trapped']} iteration={iteration} "
                f"{kind}={a}:{b}",
                flush=True,
            )
            if best["key"][0] <= 0:
                break
        elif iteration % 10 == 0:
            print(
                f"hall_walk current={current['key']} best={best['key']} "
                f"iteration={iteration}",
                flush=True,
            )

    save(best, best_order)
    print(
        f"hall_final key={best['key']} cut={best['cut']} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
