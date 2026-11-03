"""Critical-path local search for a two-lane projected engine schedule.

A flat order represents ``capacity`` processor lanes by assigning position
``p`` to lane ``p % capacity``.  Adding an edge from position ``p-capacity``
to ``p`` makes every longest-path solution a legal resource schedule.  This
representation lets a small order swap propagate a hole through hundreds of
otherwise full cycles, which contiguous time-window LNS cannot do.
"""

from __future__ import annotations

from collections import Counter
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
    engine = os.environ.get("ENGINE", "load")
    capacity = SLOT_LIMITS[engine]
    selected = [i for i, op in enumerate(ops) if op.engine == engine]
    selected_set = set(selected)
    local_of = {global_index: local for local, global_index in enumerate(selected)}
    count = len(selected)

    earliest = [0] * len(ops)
    all_children: list[list[tuple[int, int]]] = [[] for _ in ops]
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            earliest[child] = max(earliest[child], earliest[parent] + lag)
            all_children[parent].append((child, lag))
    tail = [0] * len(ops)
    for parent in reversed(range(len(ops))):
        tail[parent] = max(
            (lag + tail[child] for child, lag in all_children[parent]),
            default=0,
        )

    frontier: list[dict[int, int]] = [{} for _ in ops]
    projected: dict[int, dict[int, int]] = {}
    for child, op in enumerate(ops):
        incoming: dict[int, int] = {}
        for parent, lag in op.parents.items():
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
        for parent, lag in sorted(projected[child].items(), reverse=True):
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

    payload = json.loads(Path(os.environ["PROJECTED_HINT"]).read_text())
    raw_cycles = {int(i): int(cycle) for i, cycle in payload["cycles"].items()}
    if set(raw_cycles) != selected_set:
        raise ValueError("projected hint does not match selected engine")
    if "order" in payload:
        order = [local_of[int(i)] for i in payload["order"]]
        if sorted(order) != list(range(count)):
            raise ValueError("invalid projected resource order")
    else:
        order = [
            local_of[i]
            for i in sorted(selected, key=lambda i: (raw_cycles[i], i))
        ]

    release = [earliest[i] for i in selected]
    local_tail = [tail[i] for i in selected]

    def evaluate(candidate_order: list[int]):
        extra_children: list[list[int]] = [[] for _ in selected]
        indegree = base_indegree.copy()
        resource_edges: set[tuple[int, int]] = set()
        for position in range(capacity, count):
            parent = candidate_order[position - capacity]
            child = candidate_order[position]
            if base_parent_lag[child].get(parent, 0) < 1:
                extra_children[parent].append(child)
                indegree[child] += 1
                resource_edges.add((parent, child))

        ready = [i for i, degree in enumerate(indegree) if degree == 0]
        topo: list[int] = []
        starts = release.copy()
        reason = [-1] * count
        reason_resource = [False] * count
        while ready:
            parent = ready.pop()
            topo.append(parent)
            start = starts[parent]
            for child, lag in base_children[parent]:
                candidate = start + lag
                if candidate > starts[child]:
                    starts[child] = candidate
                    reason[child] = parent
                    reason_resource[child] = False
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
            for child in extra_children[parent]:
                candidate = start + 1
                if candidate > starts[child]:
                    starts[child] = candidate
                    reason[child] = parent
                    reason_resource[child] = True
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if len(topo) != count:
            return None

        completion = [starts[i] + local_tail[i] for i in range(count)]
        endpoint = max(range(count), key=completion.__getitem__)
        score = completion[endpoint] + 1
        threshold = score - int(os.environ.get("ORDER_KEY_WINDOW", "96"))
        pressure = sum(
            max(0, value - threshold) ** 2 for value in completion
        )
        completion_counts = Counter(completion)
        top_profile = tuple(
            completion_counts[cycle]
            for cycle in range(score - 1, max(-1, score - 33), -1)
        )
        chain: list[int] = []
        node = endpoint
        while node >= 0:
            chain.append(node)
            node = reason[node]
        chain.reverse()
        return (score, top_profile, pressure), starts, chain, reason_resource

    rng = random.Random(int(os.environ.get("RANDOM_SEED", "1")))
    iterations = int(os.environ.get("ORDER_ITERATIONS", "200"))
    candidates_per_iteration = int(os.environ.get("ORDER_CANDIDATES", "500"))
    radius = int(os.environ.get("ORDER_RADIUS", "64"))
    target = int(os.environ.get("TARGET", "959"))
    perturb_period = int(os.environ.get("ORDER_PERTURB_PERIOD", "0"))
    perturb_allowance = int(os.environ.get("ORDER_PERTURB_ALLOWANCE", "2"))
    output = Path(os.environ.get("OUT", "/tmp/aopt-load-order-search.json"))

    result = evaluate(order)
    if result is None:
        raise ValueError("initial resource order is cyclic")
    current_key, current_starts, current_chain, current_reason_resource = result
    best_key = current_key
    best_order = order.copy()
    best_starts = current_starts
    print(
        f"order_start={current_key[0]} pressure={current_key[-1]} jobs={count}",
        flush=True,
    )

    def save(score: int, starts: list[int], saved_order: list[int]) -> None:
        cycles = {str(selected[i]): starts[i] for i in range(count)}
        usage = Counter(cycles.values())
        if max(usage.values(), default=0) > capacity:
            raise AssertionError("order schedule exceeds engine capacity")
        output.write_text(
            json.dumps(
                {
                    "engine": engine,
                    "horizon": score,
                    "cycles": cycles,
                    "order": [selected[i] for i in saved_order],
                }
            )
        )

    save(best_key[0], best_starts, best_order)
    stagnant = 0
    for iteration in range(iterations):
        position = [0] * count
        for p, node in enumerate(order):
            position[node] = p
        critical_positions = [position[node] for node in current_chain]
        resource_positions = [
            position[node]
            for node in current_chain
            if current_reason_resource[node]
        ]
        focus_positions = resource_positions or critical_positions
        mutations: set[tuple[str, int, int]] = set()
        strides = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
        step = max(1, len(focus_positions) // 80)
        for p in focus_positions[::step]:
            for delta in strides:
                for q in (p - delta, p + delta):
                    if 0 <= q < count:
                        mutations.add(("swap", min(p, q), max(p, q)))
                        mutations.add(("insert", p, q))
        while len(mutations) < candidates_per_iteration:
            p = rng.choice(focus_positions)
            q = max(0, min(count - 1, p + rng.randint(-radius, radius)))
            if p != q:
                kind = "insert" if rng.randrange(2) else "swap"
                if kind == "swap":
                    mutations.add((kind, min(p, q), max(p, q)))
                else:
                    mutations.add((kind, p, q))

        ranked = []
        for kind, a, b in list(mutations)[:candidates_per_iteration]:
            if kind == "swap":
                order[a], order[b] = order[b], order[a]
            else:
                node = order.pop(a)
                order.insert(b, node)
            candidate = evaluate(order)
            if kind == "swap":
                order[a], order[b] = order[b], order[a]
            else:
                node = order.pop(b)
                order.insert(a, node)
            if candidate is not None:
                ranked.append((candidate[0], kind, a, b, candidate))
        if not ranked:
            print(f"order_stuck iteration={iteration} no_acyclic_swap", flush=True)
            break
        ranked.sort(key=lambda item: item[0])
        improving = ranked[0][0] < current_key
        force_perturb = bool(
            perturb_period and (iteration + 1) % perturb_period == 0
        )
        if improving and not force_perturb:
            chosen = ranked[0]
            stagnant = 0
        else:
            stagnant += 1
            # Plateau walks change the lane topology while bounding damage.
            allowance = (
                perturb_allowance
                if force_perturb
                else 1 if stagnant % 5 == 0 else 0
            )
            pool = [
                item
                for item in ranked[:128]
                if item[0][0] <= best_key[0] + allowance
            ]
            chosen = rng.choice(pool or ranked[:8])
        _, kind, a, b, candidate = chosen
        if kind == "swap":
            order[a], order[b] = order[b], order[a]
        else:
            node = order.pop(a)
            order.insert(b, node)
        current_key, current_starts, current_chain, current_reason_resource = candidate
        if current_key < best_key:
            best_key = current_key
            best_order = order.copy()
            best_starts = current_starts
            save(best_key[0], best_starts, best_order)
            print(
                f"order_best={best_key[0]} pressure={best_key[-1]} "
                f"iteration={iteration} {kind}={a}:{b}",
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

    final = evaluate(best_order)
    assert final is not None and final[0] == best_key
    save(best_key[0], best_starts, best_order)
    print(f"order_final={best_key[0]} pressure={best_key[-1]} output={output}")


if __name__ == "__main__":
    main()
