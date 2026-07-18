"""Search two-buffer cached-node placements by exact resource Hall pressure.

The memory-broadcast rewrite serializes materializations that reuse the same
eight-word staging buffer.  Moving one node from the long first chain to the
end of the second chain can shorten a critical gather without delaying the
early nodes at the head of the second chain.  This tool evaluates precisely
that local transformation without invoking CP-SAT.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
import json
import os

import numpy as np

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from problem import SLOT_LIMITS


def build_ops() -> list[kernel._Op]:
    kernel.SCHEDULE_EXACT_CYCLES = []
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration, ValueError):
        if not hasattr(builder, "dag_ops"):
            raise
    return real_tail_ops(builder.dag_ops)


def timing(
    ops: list[kernel._Op],
    horizon: int,
) -> tuple[list[int], list[int], list[list[tuple[int, int]]]]:
    earliest = [0] * len(ops)
    children: list[list[tuple[int, int]]] = [[] for _ in ops]
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            earliest[child] = max(earliest[child], earliest[parent] + lag)
            children[parent].append((child, lag))

    tails = [0] * len(ops)
    for parent in reversed(range(len(ops))):
        tails[parent] = max(
            (lag + tails[child] for child, lag in children[parent]),
            default=0,
        )
    latest = [horizon - 1 - tail for tail in tails]
    return earliest, latest, children


def evaluate(ops: list[kernel._Op], horizon: int) -> dict[str, object]:
    earliest, latest, _ = timing(ops, horizon)
    counts = Counter(op.engine for op in ops)

    result: dict[str, object] = {
        "dag": max(earliest, default=-1) + 1,
        "ops": len(ops),
    }
    for engine, capacity in SLOT_LIMITS.items():
        if engine == "debug":
            continue
        matrix = np.zeros((horizon, horizon), dtype=np.int32)
        for index, op in enumerate(ops):
            if op.engine != engine:
                continue
            lower, upper = earliest[index], latest[index]
            if 0 <= lower <= upper < horizon:
                matrix[lower, upper] += 1
        contained = matrix[::-1].cumsum(axis=0)[::-1].cumsum(axis=1)
        best = (-10**9, 0, 0)
        for left in range(horizon):
            lengths = np.arange(1, horizon - left + 1, dtype=np.int32)
            overloads = contained[left, left:] - capacity * lengths
            offset = int(np.argmax(overloads))
            candidate = (int(overloads[offset]), left, left + offset)
            if candidate > best:
                best = candidate
        result[engine] = {
            "count": counts[engine],
            "hall": best,
        }
    return result


def print_detail(ops: list[kernel._Op], horizon: int) -> None:
    earliest, latest, children = timing(ops, horizon)
    for engine, capacity in SLOT_LIMITS.items():
        if engine == "debug":
            continue
        result = evaluate(ops, horizon)[engine]
        overload, left, right = result["hall"]  # type: ignore[index]
        if overload <= 0:
            continue
        contained = [
            index
            for index, op in enumerate(ops)
            if op.engine == engine
            and earliest[index] >= left
            and latest[index] <= right
        ]
        by_tag = Counter(ops[index].tag for index in contained)
        print(
            "detail="
            + json.dumps(
                {
                    "engine": engine,
                    "window": (left, right),
                    "overload": overload,
                    "jobs": len(contained),
                    "capacity": capacity * (right - left + 1),
                    "tags": by_tag.most_common(),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        boundary = [
            index
            for index in contained
            if earliest[index] == left or latest[index] == right
        ]
        boundary.sort(
            key=lambda index: (
                earliest[index] != left,
                latest[index] != right,
                ops[index].group,
                ops[index].round,
                index,
            )
        )
        for index in boundary[:80]:
            op = ops[index]
            print(
                f"  i={index:5d} natural=[{earliest[index]:3d},"
                f"{latest[index]:3d}] {op.engine:5s} g={op.group:2d} "
                f"r={op.round:2d} {op.tag} "
                f"{op.slot[0] if op.slot else None} "
                f"children={len(children[index])}",
                flush=True,
            )
        if bool(int(os.environ.get("PRINT_EARLIEST_TRACES", "0"))):
            for index in [
                value for value in boundary if earliest[value] == left
            ][:16]:
                path = [index]
                current = index
                while earliest[current] > 0:
                    parents = [
                        (parent, lag)
                        for parent, lag in ops[current].parents.items()
                        if earliest[parent] + lag == earliest[current]
                    ]
                    if not parents:
                        break
                    current = max(
                        parents,
                        key=lambda item: (item[1], earliest[item[0]]),
                    )[0]
                    path.append(current)
                path.reverse()
                print(
                    "  trace "
                    + " -> ".join(
                        f"{value}:{earliest[value]}:{ops[value].engine}:"
                        f"{ops[value].tag}:g{ops[value].group}:r{ops[value].round}"
                        for value in path
                    ),
                    flush=True,
                )


def score(result: dict[str, object]) -> tuple[int, ...]:
    halls = [
        int(result[engine]["hall"][0])  # type: ignore[index]
        for engine in ("alu", "valu", "load", "store", "flow")
    ]
    positive = [max(0, value) for value in halls]
    return (
        max(positive),
        sum(positive),
        *positive,
        int(result["dag"]),
    )


def main() -> None:
    configure_target()
    horizon = int(os.environ.get("TARGET", "959"))
    split = int(os.environ.get(
        "SEARCH_BUFFER_SPLIT",
        str(kernel.MEMORY_VECTOR_CONSTANT_SWITCH_AFTER),
    ))
    order = tuple(kernel.MEMORY_CACHED_NODE_ORDER)
    if not 0 < split < len(order):
        raise ValueError("SEARCH_BUFFER_SPLIT must divide the configured order")

    first, second = order[:split], order[split:]
    candidates: list[tuple[str, tuple[int, ...], int]] = [
        ("baseline", order, split),
    ]
    if bool(int(os.environ.get("SEARCH_SINGLE_MOVES", "1"))):
        for node in first:
            candidates.append(
                (
                    f"move-{node}",
                    tuple(value for value in first if value != node)
                    + second
                    + (node,),
                    split - 1,
                )
            )

    max_moves = min(
        int(os.environ.get("SEARCH_MAX_MOVES", "8")),
        len(first) - 1,
    )
    for count in range(2, max_moves + 1):
        for label, moved in (
            ("head", first[:count]),
            ("tail", first[-count:]),
        ):
            moved_set = frozenset(moved)
            candidates.append(
                (
                    f"move-{label}-{count}",
                    tuple(value for value in first if value not in moved_set)
                    + second
                    + moved,
                    split - count,
                )
            )
    exhaustive_move_count = int(os.environ.get("SEARCH_MOVE_COUNT", "0"))
    if exhaustive_move_count:
        for moved in combinations(first, exhaustive_move_count):
            moved_set = frozenset(moved)
            candidates.append(
                (
                    "move-" + "-".join(map(str, moved)),
                    tuple(value for value in first if value not in moved_set)
                    + second
                    + moved,
                    split - exhaustive_move_count,
                )
            )

    results = []
    for name, candidate_order, candidate_split in candidates:
        kernel.MEMORY_CACHED_NODE_ORDER = candidate_order
        kernel.MEMORY_VECTOR_CONSTANT_SWITCH_AFTER = candidate_split
        try:
            result = evaluate(build_ops(), horizon)
        except (AssertionError, StopIteration, ValueError) as error:
            result = {"error": type(error).__name__ + ": " + str(error)}
        result.update(
            {
                "candidate": name,
                "split": candidate_split,
                "order": candidate_order,
            }
        )
        results.append(result)

    results.sort(
        key=lambda result: score(result)
        if "error" not in result
        else (10**9,),
    )
    for result in results:
        print(json.dumps(result, separators=(",", ":")), flush=True)
    if bool(int(os.environ.get("PRINT_BEST_DETAIL", "0"))):
        best = results[0]
        if "error" not in best:
            kernel.MEMORY_CACHED_NODE_ORDER = tuple(best["order"])
            kernel.MEMORY_VECTOR_CONSTANT_SWITCH_AFTER = int(best["split"])
            print_detail(build_ops(), horizon)


if __name__ == "__main__":
    main()
