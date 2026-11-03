"""Fast parameter sweep for a projected LOAD calendar.

The normal projected solver deliberately samples a small random priority
lattice.  LOAD is the fully saturated resource in the H959 candidates, so a
deterministic grid is more useful: it can search thousands of backward linear
extensions without rebuilding the 20k-operation DAG for every trial.
"""

from __future__ import annotations

from collections import Counter
import heapq
import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from codex.solve_full_dag import topological_data
from problem import SLOT_LIMITS


def build_projection(ops):
    children = [[] for _ in ops]
    indegree = [len(op.parents) for op in ops]
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            children[parent].append((child, lag))
    ready = [i for i, degree in enumerate(indegree) if not degree]
    heapq.heapify(ready)
    topo = []
    while ready:
        parent = heapq.heappop(ready)
        topo.append(parent)
        for child, _ in children[parent]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    position = {i: p for p, i in enumerate(topo)}
    selected = [i for i in topo if ops[i].engine == "load"]
    selected_set = set(selected)
    frontier = [{} for _ in ops]
    projected = {}
    for child in topo:
        incoming = {}
        for parent, lag in ops[child].parents.items():
            sources = {parent: 0} if parent in selected_set else frontier[parent]
            for source, distance in sources.items():
                incoming[source] = max(incoming.get(source, -1), distance + lag)
        if child in selected_set:
            projected[child] = incoming
            frontier[child] = {child: 0}
        else:
            frontier[child] = incoming
    reduced = {}
    ancestor_distance = {}
    for child in selected:
        kept = {}
        implied = {}
        for parent, lag in sorted(
            projected[child].items(), key=lambda item: position[item[0]], reverse=True
        ):
            if implied.get(parent, -1) >= lag:
                continue
            kept[parent] = lag
            implied[parent] = max(implied.get(parent, -1), lag)
            for ancestor, distance in ancestor_distance[parent].items():
                implied[ancestor] = max(implied.get(ancestor, -1), distance + lag)
        reduced[child] = kept
        ancestor_distance[child] = implied
    return selected, reduced, ancestor_distance


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
    _, _, full_earliest, full_tail = topological_data(ops)
    selected, projected, ancestor_distance = build_projection(ops)
    horizon = int(os.environ.get("TARGET", "959"))
    children = {i: [] for i in selected}
    for child, parents in projected.items():
        for parent, lag in parents.items():
            children[parent].append((child, lag))
    # Projected height/reach are stable across every priority candidate.
    height = {}
    reach = {}
    for parent in reversed(selected):
        height[parent] = max((lag + height[child] for child, lag in children[parent]), default=0)
        reach[parent] = min(
            1_000_000,
            sum(1 + reach[child] for child, _ in children[parent]),
        )
    # The projected resource graph determines the order, but release and
    # deadline bounds must come from the *full* DAG.  In particular a LOAD
    # near the tail may feed an omitted VALU/ALU consumer several cycles later;
    # using only the selected projection would incorrectly place it at H-1.
    earliest = {i: full_earliest[i] for i in selected}

    early_values = tuple(int(v) for v in os.environ.get("GRID_EARLY", "1,2,4,8,16,32,64").split(","))
    ancestor_values = tuple(int(v) for v in os.environ.get("GRID_ANCESTOR", "8,16,32,64,128,256,512").split(","))
    fanin_values = tuple(int(v) for v in os.environ.get("GRID_FANIN", "0,1,2,4,8,16").split(","))
    unlock_values = tuple(int(v) for v in os.environ.get("GRID_UNLOCK", "0,16,32,64,128,256,512,1024").split(","))
    group_values = tuple(int(v) for v in os.environ.get("GRID_GROUP", "-4,-1,0,1,4").split(","))
    best = None
    trials = 0
    # A no-noise sweep finds phase policies; optional deterministic noise
    # amplitudes then break ties without changing the graph construction.
    noise_amplitude = int(os.environ.get("GRID_NOISE", "0"))
    for early_weight in early_values:
        for ancestor_divisor in ancestor_values:
            for fanin_weight in fanin_values:
                for unlock_weight in unlock_values:
                    for group_bias in group_values:
                        trials += 1
                        successor_count = {i: len(children[i]) for i in selected}
                        latest_at = {
                            i: horizon - 1 - full_tail[i] for i in selected
                        }
                        future = {}
                        ready = []
                        for i in selected:
                            if not successor_count[i]:
                                future.setdefault(latest_at[i], []).append(i)

                        def priority(i):
                            op = ops[i]
                            unlock = sum(
                                1 + len(ancestor_distance[parent]) // ancestor_divisor
                                for parent in projected[i]
                                if successor_count[parent] == 1
                            )
                            return (
                                early_weight * earliest[i]
                                + len(ancestor_distance[i]) // ancestor_divisor
                                + fanin_weight * len(projected[i])
                                + unlock_weight * unlock
                                + group_bias * (op.group if op.group is not None else -1),
                                earliest[i],
                                len(ancestor_distance[i]),
                                len(projected[i]),
                                i,
                            )

                        def push(i):
                            heapq.heappush(ready, (tuple(-x for x in priority(i)), i))

                        cycles = {}
                        cycle = horizon - 1
                        while len(cycles) < len(selected):
                            for available in sorted((k for k in future if k >= cycle), reverse=True):
                                for i in future.pop(available):
                                    push(i)
                            used = 0
                            while ready and used < SLOT_LIMITS["load"]:
                                _, child = heapq.heappop(ready)
                                cycles[child] = cycle
                                used += 1
                                for parent, lag in projected[child].items():
                                    successor_count[parent] -= 1
                                    latest_at[parent] = min(latest_at[parent], cycle - lag)
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
                        violation = max(
                            (earliest[i] - cycles[i] for i in selected), default=0
                        )
                        shift = max(0, violation)
                        score = horizon + shift
                        if best is None or score < best[0]:
                            best = (score, early_weight, ancestor_divisor, fanin_weight, unlock_weight, group_bias, cycles)
                            print(
                                f"best={score} early={early_weight} ancestor={ancestor_divisor} "
                                f"fanin={fanin_weight} unlock={unlock_weight} group={group_bias}",
                                flush=True,
                            )
                            if score <= horizon:
                                out = Path(os.environ.get("OUT", ".search959/load-grid.json"))
                                out.write_text(json.dumps({
                                    "engine": "load",
                                    "horizon": score,
                                    "cycles": {str(i): c + shift for i, c in cycles.items()},
                                }))
                                print(f"found={score} output={out}", flush=True)
                                return
    assert best is not None
    print(f"trials={trials} best={best[:6]}", flush=True)
    out = Path(os.environ.get("OUT", ".search959/load-grid.json"))
    out.write_text(json.dumps({
        "engine": "load",
        "horizon": best[0],
        "cycles": {str(i): c + max(0, max(earliest[j] - best[-1][j] for j in selected)) for i, c in best[-1].items()},
    }))
    print(f"output={out}", flush=True)


if __name__ == "__main__":
    main()
