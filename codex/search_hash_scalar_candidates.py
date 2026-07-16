"""Rank selective hash scalarizations under a fixed FLOW order.

Changing one hash stage from VALU to eight scalar ALU operations preserves the
logical FLOW graph but shifts operation indices and resource pressure.  This
tool rebuilds each candidate variant, transfers the semantic FLOW order from a
known hint, and evaluates exact Hall-window overloads at the target horizon.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import heapq
import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_engine_order_from_source import semantic_key
from codex.map_variant_schedule import configure_target, real_tail_ops
from codex.rank_fixed_engine_hints import hall_overload
from problem import SLOT_LIMITS


def build_ops() -> list[kernel._Op]:
    kernel.SCHEDULE_EXACT_CYCLES = None
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        if not hasattr(builder, "dag_ops"):
            raise
    if bool(int(os.environ.get("PRINT_SCRATCH", "0"))):
        print(f"scratch={builder.scratch_ptr}", flush=True)
    return real_tail_ops(builder.dag_ops)


def evaluate(
    ops: list[kernel._Op],
    source_positions: dict[tuple[object, ...], deque[int]],
    horizon: int,
) -> tuple[int, tuple[int, ...]] | None:
    positions = {key: deque(values) for key, values in source_positions.items()}
    flow_scores: dict[int, int] = {}
    for index, op in enumerate(ops):
        if op.engine != "flow":
            continue
        candidates = positions.get(semantic_key(op))
        if not candidates:
            return None
        flow_scores[index] = candidates.popleft()
    if any(candidates for candidates in positions.values()):
        return None

    flow_order = sorted(flow_scores, key=lambda index: flow_scores[index])
    parents = [dict(op.parents) for op in ops]
    for previous, current in zip(flow_order, flow_order[1:]):
        parents[current][previous] = max(parents[current].get(previous, 0), 1)

    children: list[list[tuple[int, int]]] = [[] for _ in ops]
    indegree = [len(item) for item in parents]
    for child, child_parents in enumerate(parents):
        for parent, lag in child_parents.items():
            children[parent].append((child, lag))
    ready = [index for index, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    topological: list[int] = []
    while ready:
        parent = heapq.heappop(ready)
        topological.append(parent)
        for child, _ in children[parent]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(topological) != len(ops):
        return None

    earliest = [0] * len(ops)
    for child in topological:
        earliest[child] = max(
            (
                earliest[parent] + lag
                for parent, lag in parents[child].items()
            ),
            default=0,
        )
    latest = [horizon - 1] * len(ops)
    for parent in reversed(topological):
        latest[parent] = min(
            (
                latest[child] - lag
                for child, lag in children[parent]
            ),
            default=horizon - 1,
        )
    if any(left > right for left, right in zip(earliest, latest)):
        return None

    overloads = tuple(
        hall_overload(ops, earliest, latest, engine, horizon)
        for engine in ("alu", "valu", "load", "store", "flow")
    )
    return max(earliest) + 1, overloads


def main() -> None:
    configure_target()
    horizon = int(os.environ.get("TARGET", "959"))
    baseline_extra = kernel.HASH_SCALAR_EXTRA
    source_ops = build_ops()
    counts = Counter(op.engine for op in source_ops)
    print(
        "counts="
        + repr(dict(counts))
        + " floors="
        + repr(
            {
                engine: (count + SLOT_LIMITS[engine] - 1)
                // SLOT_LIMITS[engine]
                for engine, count in counts.items()
                if engine != "debug"
            }
        ),
        flush=True,
    )
    breakdown_engine = os.environ.get("PRINT_ENGINE_BREAKDOWN", "")
    if breakdown_engine:
        print(
            f"{breakdown_engine}_breakdown="
            + repr(
                Counter(
                    (op.slot[0] if op.slot else None, op.tag)
                    for op in source_ops
                    if op.engine == breakdown_engine
                ).most_common()
            ),
            flush=True,
        )
    payload = json.loads(Path(os.environ["FLOW_HINT"]).read_text())
    source_cycles = {
        int(index): int(cycle)
        for index, cycle in payload["cycles"].items()
    }
    source_flow = {i for i, op in enumerate(source_ops) if op.engine == "flow"}
    if set(source_cycles) != source_flow:
        raise ValueError("FLOW_HINT does not match the baseline graph")

    source_positions: dict[tuple[object, ...], deque[int]] = defaultdict(deque)
    for index, op in enumerate(source_ops):
        if op.engine == "flow":
            # Match repeated semantic keys in construction order, exactly as
            # map_engine_order_from_source does.  The payload cycle is only
            # the major ordering key; using FLOW-order insertion here can
            # pair duplicate operations with the wrong target instance.
            source_positions[semantic_key(op)].append(source_cycles[index])

    start = int(os.environ.get("CANDIDATE_START", "0"))
    end = int(
        os.environ.get("CANDIDATE_END", str(len(kernel._SCALAR_CANDIDATES)))
    )
    results = []
    if bool(int(os.environ.get("INCLUDE_BASE", "1"))):
        base = evaluate(source_ops, source_positions, horizon)
        if base is not None:
            dag, overloads = base
            results.append(
                (
                    max(overloads),
                    sum(max(0, value) for value in overloads),
                    dag,
                    -1,
                    None,
                    overloads,
                )
            )

    for candidate_index in range(start, min(end, len(kernel._SCALAR_CANDIDATES))):
        pair = kernel._SCALAR_CANDIDATES[candidate_index]
        if pair in baseline_extra or pair in kernel.HASH_VECTOR_FORCE_SET:
            continue
        kernel.HASH_SCALAR_EXTRA = frozenset(set(baseline_extra) | {pair})
        ops = build_ops()
        result = evaluate(ops, source_positions, horizon)
        if result is None:
            print(f"skip index={candidate_index} pair={pair}", flush=True)
            continue
        dag, overloads = result
        results.append(
            (
                max(overloads),
                sum(max(0, value) for value in overloads),
                dag,
                candidate_index,
                pair,
                overloads,
            )
        )
        print(
            f"scan index={candidate_index:3d} pair={pair!s:9s} "
            f"dag={dag:3d} overloads={overloads}",
            flush=True,
        )

    print("ranked")
    for maximum, total, dag, candidate_index, pair, overloads in sorted(results):
        print(
            f"max={maximum:3d} sum={total:3d} dag={dag:3d} "
            f"index={candidate_index:3d} pair={pair!s:9s} "
            f"overloads={overloads}"
        )


if __name__ == "__main__":
    main()
