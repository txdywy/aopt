"""Search Load-vs-Flow placement for one-shot scalar immediates.

The kernel can materialize selected pointer/constants either with a Load
``const`` instruction or with a Flow ``add_imm`` instruction.  This tool
enumerates tag subsets, transfers a known semantic Flow order, and evaluates
the exact full-DAG span and Load Hall overload at the requested horizon.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from itertools import combinations
import heapq
import json
import os
from pathlib import Path

import numpy as np

import codex.perf_takehome_under1000 as kernel
from codex.map_engine_order_from_source import (
    ordered_earliest,
    semantic_key,
    semantic_key_without_engine,
)
from codex.map_variant_schedule import configure_target, real_tail_ops
from problem import SLOT_LIMITS


def build_ops() -> list[kernel._Op]:
    kernel.SCHEDULE_EXACT_CYCLES = None
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        if not hasattr(builder, "dag_ops"):
            raise
    return real_tail_ops(builder.dag_ops)


def legalize_order(
    ops: list[kernel._Op],
    scores: dict[int, int],
) -> list[int]:
    children: list[list[int]] = [[] for _ in ops]
    indegree = [len(op.parents) for op in ops]
    for child, op in enumerate(ops):
        for parent in op.parents:
            children[parent].append(child)
    ready: list[tuple[int, int]] = []
    scale = len(ops) + 1
    for index, degree in enumerate(indegree):
        if not degree:
            heapq.heappush(
                ready,
                (scores.get(index, index * scale), index),
            )
    topological: list[int] = []
    while ready:
        _, parent = heapq.heappop(ready)
        topological.append(parent)
        for child in children[parent]:
            indegree[child] -= 1
            if not indegree[child]:
                heapq.heappush(
                    ready,
                    (scores.get(child, child * scale), child),
                )
    if len(topological) != len(ops):
        raise ValueError("kernel graph is cyclic")
    return [index for index in topological if ops[index].engine == "flow"]


def evaluate(
    ops: list[kernel._Op],
    flow_order: list[int],
    horizon: int,
) -> tuple[int, int, int, int] | None:
    parents = [dict(op.parents) for op in ops]
    for previous, current in zip(flow_order, flow_order[1:]):
        parents[current][previous] = max(
            parents[current].get(previous, 0), 1
        )
    children: list[list[tuple[int, int]]] = [[] for _ in ops]
    indegree = [len(item) for item in parents]
    for child, child_parents in enumerate(parents):
        for parent, lag in child_parents.items():
            children[parent].append((child, lag))
    ready = [index for index, degree in enumerate(indegree) if not degree]
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
    span = max(earliest) + 1
    if any(left > right for left, right in zip(earliest, latest)):
        return span, 10**6, 0, 0

    loads = [i for i, op in enumerate(ops) if op.engine == "load"]
    matrix = np.zeros((horizon, horizon), dtype=np.int16)
    np.add.at(
        matrix,
        (
            np.fromiter((earliest[i] for i in loads), dtype=np.int32),
            np.fromiter((latest[i] for i in loads), dtype=np.int32),
        ),
        1,
    )
    contained = np.cumsum(
        np.cumsum(matrix[::-1], axis=0, dtype=np.int32)[::-1],
        axis=1,
        dtype=np.int32,
    )
    best = (-10**9, 0, 0)
    for left in range(horizon):
        overloads = contained[left, left:] - SLOT_LIMITS["load"] * np.arange(
            1, horizon - left + 1, dtype=np.int32
        )
        offset = int(np.argmax(overloads))
        candidate = (int(overloads[offset]), left, left + offset)
        if candidate > best:
            best = candidate
    return span, *best


def main() -> None:
    configure_target()
    horizon = int(os.environ.get("TARGET", "959"))
    source_tags = kernel.LOAD_IMMEDIATE_TAGS
    source_ops = build_ops()
    payload = json.loads(Path(os.environ["FLOW_HINT"]).read_text())
    source_cycles = {
        int(index): int(cycle)
        for index, cycle in payload["cycles"].items()
    }
    source_flow = {
        i for i, op in enumerate(source_ops) if op.engine == "flow"
    }
    if set(source_cycles) != source_flow:
        raise ValueError("FLOW_HINT does not match the source graph")
    source_earliest = ordered_earliest(
        source_ops, "flow", source_cycles
    )

    tag_counts = Counter(
        op.tag
        for op in source_ops
        if op.engine == "load" and op.tag in source_tags
    )
    tags = tuple(sorted(tag_counts))
    min_shift = int(os.environ.get("MIN_SHIFT", "0"))
    max_shift = int(
        os.environ.get("MAX_SHIFT", str(sum(tag_counts.values())))
    )
    print(f"source_tag_counts={dict(tag_counts)}", flush=True)

    source_flow_by_key: dict[tuple[object, ...], deque[int]] = defaultdict(deque)
    source_cross_by_key: dict[
        tuple[object, ...], deque[tuple[int, int]]
    ] = defaultdict(deque)
    source_cross_by_tag: dict[
        tuple[object, ...], deque[tuple[int, int]]
    ] = defaultdict(deque)
    for index, op in enumerate(source_ops):
        if op.engine == "flow":
            source_flow_by_key[semantic_key(op)].append(source_cycles[index])
        else:
            source_cross_by_key[semantic_key_without_engine(op)].append(
                (source_earliest[index], index)
            )
            source_cross_by_tag[(op.tag, op.group, op.round)].append(
                (source_earliest[index], index)
            )

    results = []
    for retained_count in range(len(tags) + 1):
        for retained_tuple in combinations(tags, retained_count):
            retained = frozenset(retained_tuple)
            shifted = sum(
                count for tag, count in tag_counts.items()
                if tag not in retained
            )
            if not min_shift <= shifted <= max_shift:
                continue
            kernel.LOAD_IMMEDIATE_TAGS = retained
            ops = build_ops()

            flow_queues = {
                key: deque(values)
                for key, values in source_flow_by_key.items()
            }
            cross_queues = {
                key: deque(values)
                for key, values in source_cross_by_key.items()
            }
            cross_tag_queues = {
                key: deque(values)
                for key, values in source_cross_by_tag.items()
            }
            scores: dict[int, int] = {}
            scale = len(ops) + 1
            for index, op in enumerate(ops):
                if op.engine != "flow":
                    continue
                candidates = flow_queues.get(semantic_key(op))
                if candidates:
                    major = candidates.popleft()
                else:
                    cross = cross_queues.get(semantic_key_without_engine(op))
                    if not cross:
                        cross = cross_tag_queues.get(
                            (op.tag, op.group, op.round)
                        )
                    if not cross:
                        raise ValueError(
                            f"cannot map new Flow op {semantic_key(op)!r}"
                        )
                    major, _ = cross.popleft()
                scores[index] = major * scale + index

            raw_order = sorted(scores, key=lambda index: (scores[index], index))
            legalized_order = legalize_order(ops, scores)
            candidates = []
            for name, order in (
                ("raw", raw_order),
                ("legalized", legalized_order),
            ):
                result = evaluate(ops, order, horizon)
                if result is not None:
                    candidates.append((result, name, order))
            if not candidates:
                continue
            result, order_kind, order = min(
                candidates,
                key=lambda item: (
                    max(0, item[0][0] - horizon),
                    item[0][1],
                    item[0][0],
                ),
            )
            span, overload, left, right = result
            counts = Counter(op.engine for op in ops)
            record = (
                max(0, span - horizon),
                overload,
                span,
                counts["load"],
                counts["flow"],
                retained_tuple,
                order_kind,
                left,
                right,
                order,
            )
            results.append(record)
            print(
                f"shift={shifted:2d} load={counts['load']:4d} "
                f"flow={counts['flow']:3d} span={span:3d} "
                f"hall={overload:3d}@{left}:{right} "
                f"order={order_kind:9s} retained={retained_tuple}",
                flush=True,
            )

    kernel.LOAD_IMMEDIATE_TAGS = source_tags
    print("ranked", flush=True)
    output_prefix = os.environ.get("OUT_PREFIX", "")
    for rank, record in enumerate(sorted(results)[:32]):
        (
            span_excess,
            overload,
            span,
            load_count,
            flow_count,
            retained,
            order_kind,
            left,
            right,
            order,
        ) = record
        print(
            f"rank={rank:2d} excess={span_excess:2d} hall={overload:3d} "
            f"span={span:3d} load={load_count:4d} flow={flow_count:3d} "
            f"cut={left}:{right} order={order_kind} retained={retained}",
            flush=True,
        )
        if output_prefix and rank < int(os.environ.get("SAVE_TOP", "8")):
            kernel.LOAD_IMMEDIATE_TAGS = frozenset(retained)
            ops = build_ops()
            parents = [dict(op.parents) for op in ops]
            for previous, current in zip(order, order[1:]):
                parents[current][previous] = max(
                    parents[current].get(previous, 0), 1
                )
            children: list[list[tuple[int, int]]] = [[] for _ in ops]
            indegree = [len(item) for item in parents]
            for child, child_parents in enumerate(parents):
                for parent, lag in child_parents.items():
                    children[parent].append((child, lag))
            ready = [i for i, degree in enumerate(indegree) if not degree]
            heapq.heapify(ready)
            topological = []
            earliest = [0] * len(ops)
            while ready:
                parent = heapq.heappop(ready)
                topological.append(parent)
                for child, lag in children[parent]:
                    earliest[child] = max(
                        earliest[child], earliest[parent] + lag
                    )
                    indegree[child] -= 1
                    if not indegree[child]:
                        heapq.heappush(ready, child)
            output = Path(f"{output_prefix}-{rank}.json")
            output.write_text(
                json.dumps(
                    {
                        "engine": "flow",
                        "horizon": horizon,
                        "retained_load_immediate_tags": retained,
                        "cycles": {
                            str(i): earliest[i] for i in order
                        },
                    }
                )
            )
            print(f"output={output}", flush=True)
    kernel.LOAD_IMMEDIATE_TAGS = source_tags


if __name__ == "__main__":
    main()
