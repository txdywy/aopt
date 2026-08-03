"""Serialize the complete VLIW resource-order search problem.

``search_resource_orders.cpp`` originally consumed a projected two-engine
problem.  For a candidate close to the 959 resource floors, however, the
important failures are *between* all engines: independently optimal ALU,
VALU, Flow and Load calendars can form a global cycle.  This serializer keeps
every effective-DAG edge and emits one capacity order for every engine, so the
C++ local search evaluates a real, globally legal schedule whenever its
resource orders are acyclic.

The supplied per-engine JSON schedules are teachers only: their cycle values
define the initial permutations.  The output does not claim that their union
is legal.  ``search_resource_orders`` is responsible for repairing the
resource-order feedback cycles and minimizing the resulting longest path.

Example::

    source codex/sub959_candidate_balanced_hash193.env
    HINTS=/private/tmp/hash193-alu959.json,... \\
      OUT=/private/tmp/hash193-full.order \\
      python3 -m codex.dump_full_order_problem
"""

from __future__ import annotations

import heapq
import json
import os
from pathlib import Path

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

    children: list[list[tuple[int, int]]] = [[] for _ in ops]
    indegree = [len(op.parents) for op in ops]
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
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
        raise ValueError("kernel DAG is cyclic")
    position = {index: rank for rank, index in enumerate(topological)}

    earliest = [0] * len(ops)
    for child in topological:
        for parent, lag in ops[child].parents.items():
            earliest[child] = max(earliest[child], earliest[parent] + lag)
    tail = [0] * len(ops)
    for parent in reversed(topological):
        tail[parent] = max(
            (lag + tail[child] for child, lag in children[parent]), default=0
        )

    raw_paths = [value for value in os.environ.get("HINTS", "").split(",") if value]
    payloads = [json.loads(Path(path).read_text()) for path in raw_paths]
    by_engine = {str(payload.get("engine")): payload for payload in payloads}
    if len(by_engine) != len(payloads):
        raise ValueError("HINTS has duplicate or missing engine fields")

    engine_names = tuple(
        engine for engine in SLOT_LIMITS if any(op.engine == engine for op in ops)
    )
    cycle_hint: dict[int, int] = {}
    orders: dict[str, list[int]] = {}
    for engine in engine_names:
        expected = {index for index, op in enumerate(ops) if op.engine == engine}
        payload = by_engine.get(engine)
        if payload is None:
            # Debug carries no meaningful resource contention.  A stable
            # topological fallback keeps the input format total without
            # inventing an execution restriction.
            if engine == "debug":
                order = sorted(expected, key=position.__getitem__)
                for index in order:
                    cycle_hint[index] = earliest[index]
                orders[engine] = order
                continue
            raise ValueError(f"HINTS has no complete {engine} schedule")
        cycles = {int(index): int(cycle) for index, cycle in payload["cycles"].items()}
        if set(cycles) != expected:
            raise ValueError(
                f"{engine} hint mismatch missing={len(expected - set(cycles))} "
                f"extra={len(set(cycles) - expected)}"
            )
        raw_order = payload.get("order")
        if raw_order is None:
            order = sorted(expected, key=lambda index: (cycles[index], position[index]))
        else:
            order = [int(index) for index in raw_order]
            if set(order) != expected or len(order) != len(expected):
                raise ValueError(f"{engine} order does not cover its jobs")
        cycle_hint.update(cycles)
        orders[engine] = order

    partial_paths = [
        value
        for value in os.environ.get("PARTIAL_ORDER_HINTS", "").split(",")
        if value
    ]
    partial_edge_count = 0
    globalize_hints = bool(int(os.environ.get("GLOBALIZE_HINT_ORDERS", "0")))
    if partial_paths or globalize_hints:
        # A cycle-broken partial teacher is a much better starting state than
        # five complete, mutually cyclic calendars.  Extend its DAG once,
        # then derive every full engine permutation from that one global
        # extension.  Every capacity edge will consequently point forward and
        # the C++ search starts from a valid (though not necessarily compact)
        # resource order.
        partial_payloads = [json.loads(Path(path).read_text()) for path in partial_paths]
        partial_by_engine = {
            str(payload.get("engine")): payload for payload in partial_payloads
        }
        if len(partial_by_engine) != len(partial_payloads):
            raise ValueError("PARTIAL_ORDER_HINTS has duplicate engine payloads")
        partial_children: list[list[int]] = [
            [child for child, _ in row] for row in children
        ]
        partial_indegree = [len(op.parents) for op in ops]
        partial_stride = max(1, int(os.environ.get("PARTIAL_EDGE_STRIDE", "1")))
        for engine, payload in partial_by_engine.items():
            if engine not in SLOT_LIMITS:
                raise ValueError(f"unknown partial-order engine {engine}")
            for edge_rank, (raw_parent, raw_child) in enumerate(
                payload.get("resource_edges", [])
            ):
                if edge_rank % partial_stride:
                    continue
                parent, child = int(raw_parent), int(raw_child)
                if not (0 <= parent < len(ops) and 0 <= child < len(ops)):
                    raise ValueError(f"partial {engine} edge out of range")
                if ops[parent].engine != engine or ops[child].engine != engine:
                    raise ValueError(f"partial {engine} edge has wrong endpoint")
                partial_children[parent].append(child)
                partial_indegree[child] += 1
                partial_edge_count += 1
        partial_ready = [
            (cycle_hint[index], position[index], index)
            for index, degree in enumerate(partial_indegree)
            if not degree
        ]
        heapq.heapify(partial_ready)
        global_order: list[int] = []
        while partial_ready:
            _, _, parent = heapq.heappop(partial_ready)
            global_order.append(parent)
            for child in partial_children[parent]:
                partial_indegree[child] -= 1
                if not partial_indegree[child]:
                    heapq.heappush(
                        partial_ready,
                        (cycle_hint[child], position[child], child),
                    )
        if len(global_order) != len(ops):
            residual = sum(degree > 0 for degree in partial_indegree)
            raise ValueError(
                f"PARTIAL_ORDER_HINTS is cyclic with effective DAG residual={residual}"
            )
        orders = {
            engine: [
                index for index in global_order if ops[index].engine == engine
            ]
            for engine in engine_names
        }

    engine_number = {engine: rank for rank, engine in enumerate(engine_names)}
    lines = [
        f"{len(ops)} {sum(len(op.parents) for op in ops)} {len(engine_names)}"
    ]
    for index, op in enumerate(ops):
        lines.append(
            f"{index} {engine_number[op.engine]} {earliest[index]} "
            f"{tail[index]} {cycle_hint[index]}"
        )
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            lines.append(f"{parent} {child} {lag}")
    for engine in engine_names:
        order = orders[engine]
        lines.append(
            f"{engine} {SLOT_LIMITS[engine]} {len(order)} "
            + " ".join(map(str, order))
        )
    output = Path(os.environ.get("OUT", "/tmp/aopt-full.order"))
    output.write_text("\n".join(lines) + "\n")
    print(
        f"output={output} ops={len(ops)} edges={sum(len(op.parents) for op in ops)} "
        f"engines={','.join(engine_names)} partial_edges={partial_edge_count} "
        f"dag_lb={max(earliest, default=-1) + 1}",
        flush=True,
    )


if __name__ == "__main__":
    main()
