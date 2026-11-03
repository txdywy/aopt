"""Enumerate final-select engine placements with an exact Flow feasibility test.

At the sub-960 frontier, aggregate engine counts are not enough: moving one
tree select between Flow and VALU changes which operations are trapped in the
long saturated Flow window.  This tool enumerates small group subsets and
solves the projected single-machine Flow problem for each candidate.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
import heapq
import json
import os
from pathlib import Path

from ortools.sat.python import cp_model

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops


def solve_combo(combo: tuple[int, ...]) -> dict[str, object]:
    configure_target()
    kernel.VALU_FINAL_CACHE_COUNTS = {group: 1 for group in combo}
    kernel.SCHEDULE_EXACT_CYCLES = None
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        if not hasattr(builder, "dag_ops"):
            raise
    ops = real_tail_ops(builder.dag_ops)
    horizon = int(os.environ.get("TARGET", "959"))

    children: list[list[tuple[int, int]]] = [[] for _ in ops]
    indegree = [len(op.parents) for op in ops]
    for child, op in enumerate(ops):
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

    selected = [i for i in topological if ops[i].engine == "flow"]
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

    latest = [horizon - 1 - value for value in tail]
    best_overload = -10**9
    best_window = (0, horizon - 1)
    for left in range(horizon):
        histogram = [0] * horizon
        for i in selected:
            if earliest[i] >= left and 0 <= latest[i] < horizon:
                histogram[latest[i]] += 1
        contained = 0
        for right in range(left, horizon):
            contained += histogram[right]
            overload = contained - (right - left + 1)
            if overload > best_overload:
                best_overload = overload
                best_window = (left, right)
    record: dict[str, object] = {
        "combo": combo,
        "ops": len(ops),
        "engine_counts": dict(Counter(op.engine for op in ops)),
        "jobs": len(selected),
        "projected_edges": sum(map(len, projected.values())),
        "hall_overload": best_overload,
        "hall_window": best_window,
    }
    if best_overload > 0:
        record["status"] = "HALL_INFEASIBLE"
        return record

    model = cp_model.CpModel()
    starts = {
        i: model.new_int_var(earliest[i], latest[i], f"s{i}")
        for i in selected
    }
    for child in selected:
        for parent, lag in projected[child].items():
            model.add(starts[child] >= starts[parent] + lag)
    model.add_all_different(list(starts.values()))

    left, right = best_window
    trapped = {
        i
        for i in selected
        if earliest[i] >= left and latest[i] <= right
    }
    if len(trapped) == right - left + 1:
        for i in selected:
            if i in trapped or latest[i] < left or earliest[i] > right:
                continue
            can_be_early = earliest[i] <= left - 1
            can_be_late = latest[i] >= right + 1
            if can_be_early and can_be_late:
                early_side = model.new_bool_var(f"early_{i}")
                model.add(starts[i] <= left - 1).only_enforce_if(early_side)
                model.add(starts[i] >= right + 1).only_enforce_if(
                    early_side.negated()
                )
            elif can_be_early:
                model.add(starts[i] <= left - 1)
            elif can_be_late:
                model.add(starts[i] >= right + 1)
            else:
                model.add_bool_or([])
        record["saturated_window"] = best_window

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(
        os.environ.get("COMBO_TIME_LIMIT", "20")
    )
    solver.parameters.num_workers = int(os.environ.get("COMBO_WORKERS", "1"))
    solver.parameters.random_seed = (
        int(os.environ.get("RANDOM_SEED", "1"))
        + sum((position + 1) * group for position, group in enumerate(combo))
    )
    status = solver.solve(model)
    record["status"] = solver.status_name(status)
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        record["cycles"] = {
            str(i): solver.value(starts[i]) for i in selected
        }
    return record


def main() -> None:
    groups = tuple(
        int(value)
        for value in os.environ.get(
            "COUNT_GROUPS", "0,3,4,7,9,11,17,23,25,28"
        ).split(",")
        if value
    )
    count = int(os.environ.get("COUNT_TOTAL", "4"))
    candidates = list(combinations(groups, count))
    workers = int(os.environ.get("SEARCH_WORKERS", "8"))
    print(
        f"groups={groups} count={count} candidates={len(candidates)} "
        f"workers={workers}",
        flush=True,
    )
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(solve_combo, combo): combo for combo in candidates
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result["status"] in {"FEASIBLE", "OPTIMAL"}:
                print(
                    f"feasible combo={result['combo']} "
                    f"hall={result['hall_overload']}@"
                    f"{result['hall_window']}",
                    flush=True,
                )

    status_rank = {
        "OPTIMAL": 0,
        "FEASIBLE": 1,
        "UNKNOWN": 2,
        "INFEASIBLE": 3,
        "HALL_INFEASIBLE": 4,
    }
    results.sort(
        key=lambda result: (
            status_rank.get(str(result["status"]), 9),
            int(result["hall_overload"]),
            tuple(result["combo"]),
        )
    )
    summary = Counter(str(result["status"]) for result in results)
    print(f"status_counts={dict(summary)}", flush=True)
    output_prefix = os.environ.get(
        "OUT_PREFIX", "/tmp/aopt-valu-final-count-search"
    )
    saved = 0
    for rank, result in enumerate(results):
        print(
            f"rank={rank:3d} status={result['status']:15s} "
            f"hall={int(result['hall_overload']):2d}@"
            f"{result['hall_window']} combo={result['combo']}",
            flush=True,
        )
        if "cycles" not in result:
            continue
        combo = tuple(result["combo"])
        suffix = "-".join(map(str, combo))
        Path(f"{output_prefix}-{suffix}.json").write_text(
            json.dumps(
                {
                    "engine": "flow",
                    "horizon": int(os.environ.get("TARGET", "959")),
                    "combo": combo,
                    "cycles": result["cycles"],
                }
            )
        )
        saved += 1
    print(f"saved={saved} output_prefix={output_prefix}", flush=True)


if __name__ == "__main__":
    main()
