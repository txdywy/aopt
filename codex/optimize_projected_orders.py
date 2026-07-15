"""Critical-path local search over several projected engine orders.

The starting hints must come from one compatible joint schedule.  Each
engine's flat order is converted into ``capacity`` resource lanes by adding
unit-lag edges from position p-capacity to p.  Local swaps and insertions are
accepted only when the union remains acyclic, and are ranked by the resulting
whole-projection longest path.
"""

from __future__ import annotations

from collections import Counter
import heapq
import json
import os
from pathlib import Path
import random

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

    hint_paths = [Path(value) for value in os.environ["HINTS"].split(",") if value]
    hint_payloads = [json.loads(path.read_text()) for path in hint_paths]
    engines = tuple(payload["engine"] for payload in hint_payloads)
    if len(set(engines)) != len(engines):
        raise ValueError("one compatible hint per engine is required")

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
    local_of = {global_index: local for local, global_index in enumerate(selected)}
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

    orders: dict[str, list[int]] = {}
    for payload in hint_payloads:
        engine = payload["engine"]
        raw_cycles = {int(i): int(cycle) for i, cycle in payload["cycles"].items()}
        expected = {i for i in selected_set if ops[i].engine == engine}
        if set(raw_cycles) != expected:
            raise ValueError(f"hint does not match {engine}")
        if "order" in payload:
            order = [local_of[int(i)] for i in payload["order"]]
        else:
            order = [
                local_of[i]
                for i in sorted(expected, key=lambda i: (raw_cycles[i], i))
            ]
        if set(order) != {local_of[i] for i in expected}:
            raise ValueError(f"invalid {engine} order")
        orders[engine] = order

    release = [earliest[i] for i in selected]
    local_tail = [tail[i] for i in selected]

    def evaluate(candidate_orders: dict[str, list[int]]):
        extra_children: list[list[tuple[int, str]]] = [[] for _ in selected]
        indegrees = base_indegree.copy()
        for engine, order in candidate_orders.items():
            capacity = SLOT_LIMITS[engine]
            for previous, current in zip(order, order[capacity:]):
                if base_parent_lag[current].get(previous, 0) < 1:
                    extra_children[previous].append((current, engine))
                    indegrees[current] += 1

        ready_nodes = [i for i, degree in enumerate(indegrees) if not degree]
        topo: list[int] = []
        starts = release.copy()
        reason = [-1] * count
        reason_engine: list[str | None] = [None] * count
        while ready_nodes:
            parent = ready_nodes.pop()
            topo.append(parent)
            start = starts[parent]
            for child, lag in base_children[parent]:
                candidate = start + lag
                if candidate > starts[child]:
                    starts[child] = candidate
                    reason[child] = parent
                    reason_engine[child] = None
                indegrees[child] -= 1
                if not indegrees[child]:
                    ready_nodes.append(child)
            for child, resource_engine in extra_children[parent]:
                candidate = start + 1
                if candidate > starts[child]:
                    starts[child] = candidate
                    reason[child] = parent
                    reason_engine[child] = resource_engine
                indegrees[child] -= 1
                if not indegrees[child]:
                    ready_nodes.append(child)
        if len(topo) != count:
            return None

        completion = [starts[i] + local_tail[i] for i in range(count)]
        endpoint = max(range(count), key=completion.__getitem__)
        score = completion[endpoint] + 1
        window = int(os.environ.get("ORDER_KEY_WINDOW", "96"))
        threshold = score - window
        pressure = sum(max(0, value - threshold) ** 2 for value in completion)
        completion_counts = Counter(completion)
        profile = tuple(
            completion_counts[cycle]
            for cycle in range(score - 1, max(-1, score - 33), -1)
        )
        chain: list[int] = []
        node = endpoint
        while node >= 0:
            chain.append(node)
            node = reason[node]
        chain.reverse()
        return (score, profile, pressure), starts, chain, reason_engine

    rng = random.Random(int(os.environ.get("RANDOM_SEED", "1")))
    iterations = int(os.environ.get("ORDER_ITERATIONS", "300"))
    candidates_per_iteration = int(os.environ.get("ORDER_CANDIDATES", "160"))
    radius = int(os.environ.get("ORDER_RADIUS", "64"))
    target = int(os.environ.get("TARGET", "959"))
    perturb_period = int(os.environ.get("ORDER_PERTURB_PERIOD", "0"))
    perturb_allowance = int(os.environ.get("ORDER_PERTURB_ALLOWANCE", "2"))
    output_prefix = Path(os.environ.get("OUT_PREFIX", "/tmp/aopt-joint-order"))

    result = evaluate(orders)
    if result is None:
        raise ValueError("initial joint resource orders are cyclic")
    current_key, current_starts, current_chain, current_reason_engine = result
    best_key = current_key
    best_orders = {engine: order.copy() for engine, order in orders.items()}
    best_starts = current_starts
    print(
        f"order_start={current_key[0]} pressure={current_key[-1]} jobs={count}",
        flush=True,
    )

    def save(score: int, starts: list[int], saved_orders: dict[str, list[int]]) -> None:
        for engine, order in saved_orders.items():
            cycles = {
                str(selected[i]): starts[i]
                for i in range(count)
                if ops[selected[i]].engine == engine
            }
            if max(Counter(cycles.values()).values(), default=0) > SLOT_LIMITS[engine]:
                raise AssertionError(f"{engine} capacity overflow")
            Path(f"{output_prefix}-{engine}.json").write_text(
                json.dumps(
                    {
                        "engine": engine,
                        "horizon": score,
                        "cycles": cycles,
                        "order": [selected[i] for i in order],
                    }
                )
            )

    save(best_key[0], best_starts, best_orders)
    stagnant = 0
    for iteration in range(iterations):
        positions = {
            engine: {node: p for p, node in enumerate(order)}
            for engine, order in orders.items()
        }
        focus: list[tuple[str, int]] = []
        for node in current_chain:
            resource_engine = current_reason_engine[node]
            engine = resource_engine or ops[selected[node]].engine
            if engine in orders:
                focus.append((engine, positions[engine][node]))
        if not focus:
            focus = [
                (engine, rng.randrange(len(order)))
                for engine, order in orders.items()
            ]
        mutations: set[tuple[str, str, int, int]] = set()
        strides = (
            1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64,
            96, 128, 192, 256, 384, 512, 768,
        )
        step = max(1, len(focus) // 80)
        for engine, p in focus[::step]:
            length = len(orders[engine])
            for delta in strides:
                for q in (p - delta, p + delta):
                    if 0 <= q < length:
                        mutations.add((engine, "swap", min(p, q), max(p, q)))
                        mutations.add((engine, "insert", p, q))
        while len(mutations) < candidates_per_iteration:
            engine, p = rng.choice(focus)
            length = len(orders[engine])
            q = max(0, min(length - 1, p + rng.randint(-radius, radius)))
            if p != q:
                kind = "insert" if rng.randrange(2) else "swap"
                if kind == "swap":
                    mutations.add((engine, kind, min(p, q), max(p, q)))
                else:
                    mutations.add((engine, kind, p, q))

        ranked = []
        for engine, kind, a, b in list(mutations)[:candidates_per_iteration]:
            order = orders[engine]
            if kind == "swap":
                order[a], order[b] = order[b], order[a]
            else:
                node = order.pop(a)
                order.insert(b, node)
            candidate = evaluate(orders)
            if kind == "swap":
                order[a], order[b] = order[b], order[a]
            else:
                node = order.pop(b)
                order.insert(a, node)
            if candidate is not None:
                ranked.append((candidate[0], engine, kind, a, b, candidate))
        if not ranked:
            print(f"order_stuck iteration={iteration}", flush=True)
            break
        ranked.sort(key=lambda item: item[0])
        improving = ranked[0][0] < current_key
        force_perturb = bool(perturb_period and (iteration + 1) % perturb_period == 0)
        if improving and not force_perturb:
            chosen = ranked[0]
            stagnant = 0
        else:
            stagnant += 1
            allowance = perturb_allowance if force_perturb else (1 if stagnant % 5 == 0 else 0)
            pool = [
                item for item in ranked[:128]
                if item[0][0] <= best_key[0] + allowance
            ]
            chosen = rng.choice(pool or ranked[:8])
        _, engine, kind, a, b, candidate = chosen
        order = orders[engine]
        if kind == "swap":
            order[a], order[b] = order[b], order[a]
        else:
            node = order.pop(a)
            order.insert(b, node)
        current_key, current_starts, current_chain, current_reason_engine = candidate
        if current_key < best_key:
            best_key = current_key
            best_orders = {name: value.copy() for name, value in orders.items()}
            best_starts = current_starts
            save(best_key[0], best_starts, best_orders)
            print(
                f"order_best={best_key[0]} pressure={best_key[-1]} "
                f"iteration={iteration} {engine}:{kind}={a}:{b}",
                flush=True,
            )
            if best_key[0] <= target:
                break
        elif iteration % 10 == 0:
            print(
                f"order_walk={current_key[0]} best={best_key[0]} "
                f"pressure={current_key[-1]} iteration={iteration}",
                flush=True,
            )

    save(best_key[0], best_starts, best_orders)
    print(
        f"order_final={best_key[0]} pressure={best_key[-1]} "
        f"output_prefix={output_prefix}",
        flush=True,
    )


if __name__ == "__main__":
    main()
