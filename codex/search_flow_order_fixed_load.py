"""Search a FLOW resource order compatible with a fixed saturated LOAD calendar.

The H959 gather38 graph has a useful special shape: every LOAD operation is
trapped in cycles 0..948 and the two LOAD lanes are exactly full.  This tool
keeps that calendar fixed and searches only the capacity-one FLOW permutation.
Unlike independent projected solves, every trial evaluates the original
weighted FLOW/LOAD precedence projection and reports the actual cross-engine
slack.  It is an offline order oracle; it never changes the production kernel.
"""

from __future__ import annotations

from collections import defaultdict
import heapq
import json
import os
import random
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops


def build_projection(ops: list[object]):
    children = [[] for _ in ops]
    indegree = [len(op.parents) for op in ops]
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            children[parent].append((child, lag))
    ready = [i for i, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    topo = []
    while ready:
        parent = heapq.heappop(ready)
        topo.append(parent)
        for child, _ in children[parent]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(topo) != len(ops):
        raise ValueError("kernel DAG is cyclic")
    selected = {
        i for i, op in enumerate(ops) if op.engine in {"flow", "load"}
    }
    frontier = [{} for _ in ops]
    projected = {}
    for child in topo:
        incoming = {}
        for parent, lag in ops[child].parents.items():
            sources = ({parent: 0} if parent in selected else frontier[parent])
            for source, distance in sources.items():
                incoming[source] = max(incoming.get(source, -1), distance + lag)
        if child in selected:
            projected[child] = incoming
            frontier[child] = {child: 0}
        else:
            frontier[child] = incoming
    # Exact weighted transitive reduction in topological order.
    ancestors = {}
    reduced = {}
    for child in topo:
        if child not in selected:
            continue
        implied = {}
        kept = {}
        for parent, lag in sorted(projected[child].items(), reverse=True):
            if implied.get(parent, -1) >= lag:
                continue
            kept[parent] = lag
            implied[parent] = max(implied.get(parent, -1), lag)
            for ancestor, distance in ancestors[parent].items():
                implied[ancestor] = max(implied.get(ancestor, -1), distance + lag)
        reduced[child] = kept
        ancestors[child] = implied
    return topo, [i for i in topo if i in selected], reduced


def load_cycles(path: str) -> dict[int, int]:
    payload = json.loads(Path(path).read_text())
    return {int(index): int(cycle) for index, cycle in payload["cycles"].items()}


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
    topo, selected, projected = build_projection(ops)
    flow = [i for i in selected if ops[i].engine == "flow"]
    load = [i for i in selected if ops[i].engine == "load"]
    horizon = int(os.environ.get("TARGET", "959"))
    fixed_load = load_cycles(os.environ["LOAD_HINT"])
    if set(fixed_load) != set(load):
        raise ValueError("LOAD_HINT does not cover exactly the LOAD projection")
    if max(fixed_load.values(), default=-1) >= horizon:
        raise ValueError("LOAD_HINT exceeds target horizon")

    # Build compact adjacency once.  The trial graph adds one +1 edge from
    # each FLOW operation to its successor in the candidate order.
    base_children = [[] for _ in ops]
    base_indegree = {i: 0 for i in selected}
    for child in selected:
        for parent, lag in projected[child].items():
            base_children[parent].append((child, lag))
            base_indegree[child] += 1

    def evaluate(order: list[int]):
        extra_children = defaultdict(list)
        extra_parents = defaultdict(list)
        indegree = dict(base_indegree)
        for parent, child in zip(order, order[1:]):
            # A direct projected edge already implies the resource edge.
            if projected[child].get(parent, 0) < 1:
                extra_children[parent].append((child, 1))
                extra_parents[child].append((parent, 1))
                indegree[child] += 1
        ready = [i for i in selected if indegree[i] == 0]
        heapq.heapify(ready)
        trial_topo = []
        while ready:
            parent = heapq.heappop(ready)
            trial_topo.append(parent)
            for child, _ in base_children[parent]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(ready, child)
            for child, _ in extra_children[parent]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(ready, child)
        if len(trial_topo) != len(selected):
            return (10**9, 10**9, 10**9, 10**9), None

        cycles = {}
        load_violation = 0
        for child in trial_topo:
            required = max(
                (cycles[parent] + lag for parent, lag in projected[child].items()),
                default=0,
            )
            for parent, lag in extra_parents.get(child, ()):
                required = max(required, cycles[parent] + lag)
            if ops[child].engine == "load":
                actual = fixed_load[child]
                load_violation += max(0, required - actual)
                cycles[child] = actual
            else:
                cycles[child] = required
        max_flow = max((cycles[i] for i in flow), default=0)
        flow_late = sum(max(0, cycles[i] - (horizon - 1)) for i in flow)
        # The first component strongly prefers a legal target calendar; the
        # remaining terms provide a smooth landscape while LOAD constraints
        # are still violated.
        key = (
            max(0, max_flow - (horizon - 1)) + load_violation * 1000,
            max_flow,
            load_violation,
            flow_late,
        )
        return key, cycles

    def initial_order(path: str) -> list[int]:
        payload = json.loads(Path(path).read_text())
        cycles = {int(i): int(c) for i, c in payload["cycles"].items()}
        if set(cycles) != set(flow):
            raise ValueError("FLOW_HINT does not cover exactly the FLOW projection")
        raw = payload.get("order")
        if raw is not None:
            order = [int(i) for i in raw]
            if set(order) == set(flow) and len(order) == len(flow):
                return order
        return sorted(flow, key=lambda i: (cycles[i], i))

    order = initial_order(os.environ["FLOW_HINT"])
    starts = os.environ.get("INITIAL_ORDER", "")
    if starts:
        order = [int(x) for x in starts.split(",") if x]
    result = evaluate(order)
    print(f"initial={result[0]}", flush=True)

    rng = random.Random(int(os.environ.get("SEED", "959")))
    iterations = int(os.environ.get("ITERATIONS", "20000"))
    candidates = int(os.environ.get("CANDIDATES", "128"))
    radius = int(os.environ.get("RADIUS", "256"))
    current_key, current_cycles = result
    best_key, best_order, best_cycles = current_key, order[:], current_cycles
    groups = {}
    for index in flow:
        groups.setdefault((ops[index].group, ops[index].round), []).append(index)

    def trial_move(base: list[int]):
        kind = rng.randrange(4)
        if kind == 0:
            left = rng.randrange(len(base) - 1)
            right = min(len(base) - 1, left + 1 + rng.randrange(max(1, radius)))
            trial = base[:]
            trial[left], trial[right] = trial[right], trial[left]
            return trial
        if kind == 1:
            left = rng.randrange(len(base))
            right = min(len(base), left + 1 + rng.randrange(max(1, radius)))
            node = base.pop(left)
            base.insert(right - 1 if right > left else right, node)
            return base
        keys = list(groups)
        key = rng.choice(keys)
        block_set = set(groups[key])
        block = [node for node in base if node in block_set]
        rest = [node for node in base if node not in block_set]
        insertion = rng.randrange(len(rest) + 1)
        return rest[:insertion] + block + rest[insertion:]

    for iteration in range(iterations):
        pool = []
        for _ in range(candidates):
            trial = trial_move(order[:])
            key, cycles = evaluate(trial)
            pool.append((key, trial, cycles))
        key, trial, cycles = min(pool, key=lambda item: item[0])
        # Accept an improving move, or an occasional equal move to escape a
        # local permutation basin.  Invalid cyclic unions are never accepted.
        if key <= current_key or (key[0] < 10**9 and rng.random() < 0.002):
            order, current_key, current_cycles = trial, key, cycles
        if current_key < best_key:
            best_key, best_order, best_cycles = current_key, order[:], current_cycles
            print(
                f"best={best_key} iteration={iteration} "
                f"flow_max={max(best_cycles[i] for i in flow)}",
                flush=True,
            )
            if best_key[0] == 0 and best_key[2] == 0:
                break

    print(f"final={best_key} flow_max={max(best_cycles[i] for i in flow)}", flush=True)
    if best_key[0] == 0 and best_key[2] == 0:
        out = Path(os.environ.get("OUT", ".search959/flow-fixed-load.json"))
        out.write_text(
            json.dumps(
                {
                    "engine": "flow",
                    "horizon": horizon,
                    "cycles": {str(i): best_cycles[i] for i in flow},
                    "order": best_order,
                    "fixed_load": os.environ["LOAD_HINT"],
                }
            )
        )
        print(f"output={out}", flush=True)


if __name__ == "__main__":
    main()
