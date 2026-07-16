"""Explain the Hall cut induced by a fixed unary FLOW order."""

from __future__ import annotations

from collections import Counter
import heapq
import json
import os
from pathlib import Path

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
    horizon = int(os.environ.get("TARGET", "959"))
    engine = os.environ.get("ENGINE", "load")

    payload = json.loads(Path(os.environ["FLOW_HINT"]).read_text())
    raw_cycles = {int(i): int(c) for i, c in payload["cycles"].items()}
    flow_set = {i for i, op in enumerate(ops) if op.engine == "flow"}
    if set(raw_cycles) != flow_set:
        raise ValueError("FLOW_HINT does not match the target DAG")
    flow_order = sorted(flow_set, key=lambda i: (raw_cycles[i], i))
    flow_position = {node: position for position, node in enumerate(flow_order)}

    parents = [dict(op.parents) for op in ops]
    for previous, current in zip(flow_order, flow_order[1:]):
        parents[current][previous] = max(parents[current].get(previous, 0), 1)
    children: list[list[tuple[int, int]]] = [[] for _ in ops]
    indegree = [len(item) for item in parents]
    for child, child_parents in enumerate(parents):
        for parent, lag in child_parents.items():
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
        raise ValueError("fixed FLOW order is cyclic")

    earliest = [0] * len(ops)
    early_reason = [-1] * len(ops)
    for child in topological:
        for parent, lag in parents[child].items():
            value = earliest[parent] + lag
            if value > earliest[child]:
                earliest[child] = value
                early_reason[child] = parent
    latest = [horizon - 1] * len(ops)
    late_reason = [-1] * len(ops)
    for parent in reversed(topological):
        for child, lag in children[parent]:
            value = latest[child] - lag
            if value < latest[parent]:
                latest[parent] = value
                late_reason[parent] = child
    if any(left > right for left, right in zip(earliest, latest)):
        raise ValueError("fixed FLOW order does not fit the horizon")

    indices = [i for i, op in enumerate(ops) if op.engine == engine]
    matrix = np.zeros((horizon, horizon), dtype=np.int16)
    np.add.at(
        matrix,
        (
            np.fromiter((earliest[i] for i in indices), dtype=np.int32),
            np.fromiter((latest[i] for i in indices), dtype=np.int32),
        ),
        1,
    )
    contained = np.cumsum(
        np.cumsum(matrix[::-1], axis=0, dtype=np.int32)[::-1],
        axis=1,
        dtype=np.int32,
    )
    best = (-10**9, 0, 0)
    capacity = SLOT_LIMITS[engine]
    for left in range(horizon):
        overloads = contained[left, left:] - capacity * np.arange(
            1, horizon - left + 1, dtype=np.int32
        )
        offset = int(np.argmax(overloads))
        best = max(best, (int(overloads[offset]), left, left + offset))
    overload, left, right = best
    trapped = [
        i for i in indices if earliest[i] >= left and latest[i] <= right
    ]
    early_eligible = [i for i in indices if earliest[i] < left]
    late_eligible = [i for i in indices if latest[i] > right]
    summary_only = bool(int(os.environ.get("SUMMARY_ONLY", "0")))
    print(
        f"engine={engine} overload={overload} cut={left}:{right} "
        f"trapped={len(trapped)} capacity={capacity * (right - left + 1)} "
        f"total={len(indices)} early_eligible={len(early_eligible)} "
        f"late_eligible={len(late_eligible)}"
    )
    if not summary_only:
        print("trapped_group_round=" + repr(Counter(
            (ops[i].group, ops[i].round) for i in trapped
        ).most_common()))

    def flow_root(start: int, reasons: list[int]) -> int:
        node = start
        seen = set()
        while node >= 0 and node not in seen:
            seen.add(node)
            if ops[node].engine == "flow":
                return node
            node = reasons[node]
        return -1

    early_roots = Counter(flow_root(i, early_reason) for i in trapped)
    late_roots = Counter(flow_root(i, late_reason) for i in trapped)

    def describe(counter: Counter[int]) -> None:
        for node, count in counter.most_common(50):
            if node < 0:
                print(f"{count:4d} no_flow_root")
                continue
            op = ops[node]
            print(
                f"{count:4d} pos={flow_position[node]:3d} "
                f"e={earliest[node]:3d} l={latest[node]:3d} "
                f"g={op.group:2d} r={op.round:2d} {op.tag} i={node}"
            )

    if not summary_only:
        print("early_flow_roots")
        describe(early_roots)
        print("late_flow_roots")
        describe(late_roots)

    if bool(int(os.environ.get("DETAIL_BOUNDARY", "0"))):
        detail_limit = int(os.environ.get("BOUNDARY_DETAIL_LIMIT", "64"))

        def describe_boundary(
            title: str,
            candidates: list[int],
            reasons: list[int],
        ) -> None:
            print(title)
            print(
                "tag_counts="
                + repr(Counter(ops[i].tag for i in candidates).most_common())
            )
            for index in candidates[:detail_limit]:
                op = ops[index]
                chain = []
                node = index
                seen = set()
                while node >= 0 and node not in seen and len(chain) < 10:
                    seen.add(node)
                    parent = reasons[node]
                    if parent < 0:
                        break
                    parent_op = ops[parent]
                    chain.append(
                        (
                            parent,
                            parent_op.engine,
                            parent_op.tag,
                            parent_op.group,
                            parent_op.round,
                            earliest[parent],
                            latest[parent],
                        )
                    )
                    node = parent
                print(
                    f"i={index} e={earliest[index]} l={latest[index]} "
                    f"g={op.group} r={op.round} tag={op.tag} "
                    f"slot={op.slot!r} chain={chain!r}"
                )

        describe_boundary(
            "early_boundary_trapped",
            [
                i
                for i in trapped
                if earliest[i] == left
            ],
            early_reason,
        )
        describe_boundary(
            "late_boundary_trapped",
            [
                i
                for i in trapped
                if latest[i] == right
            ],
            late_reason,
        )

    if not summary_only:
        radius = int(os.environ.get("BOUNDARY_RADIUS", "32"))
        for boundary in (left, right):
            print(f"flow_boundary={boundary}")
            for node in flow_order:
                if abs(earliest[node] - boundary) > radius:
                    continue
                op = ops[node]
                print(
                    f"pos={flow_position[node]:3d} e={earliest[node]:3d} "
                    f"l={latest[node]:3d} g={op.group:2d} r={op.round:2d} "
                    f"{op.tag} i={node}"
                )


if __name__ == "__main__":
    main()
