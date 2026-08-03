"""Map a feasible VALU/FLOW pair calendar onto a small target DAG rewrite.

The source calendar is a 959-cycle witness for the all-LOAD setup variant.
The target graph replaces twelve cached-node LOADs with eight ALU broadcasts
and four VALU broadcasts, restoring the LOAD Hall window.  This tool keeps
the common VALU/FLOW operations as hints and lets a compact CP-SAT model place
only the newly introduced pair operations (and any boundary operations whose
weighted projected edges changed).
"""

from __future__ import annotations

from collections import defaultdict, deque, Counter
import heapq
import json
import os
from pathlib import Path

from ortools.sat.python import cp_model

from problem import SLOT_LIMITS


def key(op):
    identity = op.get("identity")
    if identity:
        return (
            op["engine"],
            "identity",
            op["group"],
            op["round"],
            json.dumps(identity, sort_keys=True, separators=(",", ":")),
        )
    slot = op.get("slot") or []
    return (op["engine"], slot[0] if slot else None, op["tag"], op["group"], op["round"])


def projection(ops, selected_set):
    children = [[] for _ in ops]
    indegree = [0] * len(ops)
    for child, op in enumerate(ops):
        indegree[child] = len(op["parents"])
        for parent, lag in op["parents"].items():
            parent = int(parent)
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
        raise ValueError("target graph is cyclic")
    earliest = [0] * len(ops)
    for child in topo:
        for parent, lag in ops[child]["parents"].items():
            earliest[child] = max(earliest[child], earliest[int(parent)] + lag)
    tail = [0] * len(ops)
    for parent in reversed(topo):
        tail[parent] = max(
            (lag + tail[child] for child, lag in children[parent]),
            default=0,
        )
    position = {index: rank for rank, index in enumerate(topo)}
    frontier = [{} for _ in ops]
    raw = {}
    for child in topo:
        incoming = {}
        for parent_text, lag in ops[child]["parents"].items():
            parent = int(parent_text)
            sources = {parent: 0} if parent in selected_set else frontier[parent]
            for source, distance in sources.items():
                incoming[source] = max(incoming.get(source, -1), distance + lag)
        if child in selected_set:
            raw[child] = incoming
            frontier[child] = {child: 0}
        else:
            frontier[child] = incoming
    reduced = {}
    ancestors = {}
    for child in topo:
        if child not in selected_set:
            continue
        kept = {}
        implied = {}
        for parent, lag in sorted(
            raw[child].items(), key=lambda item: position[item[0]], reverse=True
        ):
            if implied.get(parent, -1) >= lag:
                continue
            kept[parent] = lag
            implied[parent] = max(implied.get(parent, -1), lag)
            for ancestor, distance in ancestors[parent].items():
                implied[ancestor] = max(implied.get(ancestor, -1), distance + lag)
        reduced[child] = kept
        ancestors[child] = implied
    return topo, reduced, earliest, tail


def main():
    source = json.loads(Path(os.environ["SOURCE_META"]).read_text())["ops"]
    target = json.loads(Path(os.environ["TARGET_META"]).read_text())["ops"]
    source_valu = json.loads(Path(os.environ["SOURCE_VALU"]).read_text())["cycles"]
    source_flow = json.loads(Path(os.environ["SOURCE_FLOW"]).read_text())["cycles"]
    source_cycles = {
        int(index): int(cycle)
        for payload in (source_valu, source_flow)
        for index, cycle in payload.items()
    }
    selected_set = {
        i for i, op in enumerate(target) if op["engine"] in ("valu", "flow")
    }
    source_by_key = defaultdict(deque)
    for index, op in enumerate(source):
        if op["engine"] in ("valu", "flow"):
            source_by_key[key(op)].append(index)
    mapped = {}
    for index, op in enumerate(target):
        if op["engine"] not in ("valu", "flow"):
            continue
        queue = source_by_key[key(op)]
        while queue and queue[0] not in source_cycles:
            queue.popleft()
        if queue:
            mapped[index] = source_cycles[queue.popleft()]
    new = sorted(selected_set - set(mapped))
    print(
        f"target_pair={len(selected_set)} mapped={len(mapped)} new={len(new)}",
        flush=True,
    )
    for index in new:
        op = target[index]
        print(
            f"new id={index} engine={op['engine']} g={op['group']} "
            f"r={op['round']} tag={op['tag']} identity={op['identity']}",
            flush=True,
        )
    topo, projected, earliest, tail = projection(target, selected_set)
    horizon = int(os.environ.get("TARGET", "959"))
    common_bad = []
    for child in selected_set:
        for parent, lag in projected[child].items():
            if parent in mapped and child in mapped:
                amount = mapped[parent] + lag - mapped[child]
                if amount > 0:
                    common_bad.append((amount, parent, child, lag))
    print(f"common_projected_violations={len(common_bad)}", flush=True)
    if common_bad:
        print("worst_common=" + repr(sorted(common_bad, reverse=True)[:20]), flush=True)
    for index in new:
        print(
            f"new_bounds id={index} earliest={earliest[index]} "
            f"latest={horizon - 1 - tail[index]} tail={tail[index]}",
            flush=True,
        )
    fixed_usage = Counter(
        (cycle, target[index]["engine"])
        for index, cycle in mapped.items()
    )
    print(
        "fixed_usage_peak="
        + repr(
            sorted(
                ((cycle, engine, demand) for (cycle, engine), demand in fixed_usage.items()),
                key=lambda item: item[2],
                reverse=True,
            )[:16]
        ),
        flush=True,
    )

    model = cp_model.CpModel()
    starts = {}
    movable = set(new)
    # If the source/target rewrite changes a projected edge between two common
    # operations, release both endpoints in a small radius around their old
    # cycles.  The default keeps all otherwise common operations pinned.
    radius = int(os.environ.get("COMMON_RADIUS", "0"))
    for index in sorted(selected_set):
        if index in mapped and radius == 0:
            starts[index] = model.new_int_var(mapped[index], mapped[index], f"s{index}")
        elif index in mapped:
            starts[index] = model.new_int_var(
                max(0, mapped[index] - radius),
                min(horizon - 1, mapped[index] + radius),
                f"s{index}",
            )
            movable.add(index)
        else:
            starts[index] = model.new_int_var(
                earliest[index], horizon - 1 - tail[index], f"s{index}"
            )
    for child in selected_set:
        for parent, lag in projected[child].items():
            model.add(starts[child] >= starts[parent] + lag)
    for engine, capacity in (("valu", SLOT_LIMITS["valu"]), ("flow", SLOT_LIMITS["flow"])):
        intervals = [
            model.new_fixed_size_interval_var(starts[index], 1, f"i{index}")
            for index in selected_set
            if target[index]["engine"] == engine
        ]
        model.add_cumulative(intervals, [1] * len(intervals), capacity)
    for index in selected_set:
        if index in mapped:
            model.add_hint(starts[index], mapped[index])
        else:
            model.add_hint(starts[index], min(horizon - 1 - tail[index], max(earliest[index], 1)))
    makespan = model.new_int_var(0, horizon - 1, "makespan")
    for index in selected_set:
        model.add(makespan >= starts[index] + tail[index])
    model.minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(os.environ.get("TIME_LIMIT", "60"))
    solver.parameters.num_workers = int(os.environ.get("WORKERS", "4"))
    solver.parameters.random_seed = int(os.environ.get("RANDOM_SEED", "1"))
    status = solver.solve(model)
    print(
        f"status={solver.status_name(status)} objective={solver.objective_value + 1} "
        f"bound={solver.best_objective_bound + 1} wall={solver.wall_time:.3f}",
        flush=True,
    )
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return
    cycles = {index: solver.value(starts[index]) for index in selected_set}
    usage = Counter((cycles[index], target[index]["engine"]) for index in selected_set)
    if any(demand > SLOT_LIMITS[engine] for (_, engine), demand in usage.items()):
        raise AssertionError("capacity overflow")
    score = max(cycles[index] + tail[index] for index in selected_set) + 1
    prefix = Path(os.environ.get("OUT_PREFIX", "/tmp/aopt-pair-incremental"))
    for engine in ("valu", "flow"):
        Path(f"{prefix}-{engine}.json").write_text(
            json.dumps(
                {
                    "engine": engine,
                    "horizon": score,
                    "cycles": {
                        str(index): cycles[index]
                        for index in selected_set
                        if target[index]["engine"] == engine
                    },
                }
            )
        )
    print(f"score={score} output_prefix={prefix}", flush=True)


if __name__ == "__main__":
    main()
