"""Build a dependency-aware phase teacher from the 987 schedule.

Per-engine projected calendars are intentionally incomplete teachers: a new
operation can have a parent/consumer on an engine that was not projected.
This utility maps the unchanged operations from the embedded schedule and
interpolates rewritten operations between their nearest mapped DAG boundaries.
The result is a phase hint only; ``schedule_hint_list`` still validates the
complete DAG and all capacities.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import heapq
import json
import os
from pathlib import Path
import statistics

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, op_key, real_tail_ops


def build_default() -> list[kernel._Op]:
    # At import time the kernel carries the embedded default schedule.  Build
    # this graph before applying the target environment.
    builder = kernel.KernelBuilder()
    builder.build_kernel(10, 2047, 256, 16)
    return real_tail_ops(builder.dag_ops)


def operation_ids(ops: list[kernel._Op]) -> list[tuple[tuple[object, ...], int]]:
    occurrences: Counter[tuple[object, ...]] = Counter()
    result = []
    for op in ops:
        key = op_key(op)
        result.append((key, occurrences[key]))
        occurrences[key] += 1
    return result


def main() -> None:
    source_ops = build_default()
    source_cycles = json.loads(
        Path(os.environ.get("SOURCE", "/tmp/default-full.json")).read_text()
    )["cycles"]
    if isinstance(source_cycles, dict):
        source_cycles = [int(source_cycles[str(i)]) for i in range(len(source_ops))]
    source_cycles = [int(x) for x in source_cycles]
    if len(source_cycles) != len(source_ops):
        raise ValueError("source schedule length does not match default graph")

    configure_target()
    kernel.SCHEDULE_EXACT_CYCLES = None
    target_builder = kernel.KernelBuilder()
    try:
        target_builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        if not hasattr(target_builder, "dag_ops"):
            raise
    target_ops = real_tail_ops(target_builder.dag_ops)

    source_ids = operation_ids(source_ops)
    source_by_id = {identity: index for index, identity in enumerate(source_ids)}
    target_ids = operation_ids(target_ops)
    mapped: dict[int, int] = {
        index: source_cycles[source_by_id[identity]]
        for index, identity in enumerate(target_ids)
        if identity in source_by_id
    }

    children: list[list[tuple[int, int]]] = [[] for _ in target_ops]
    indegree = [len(op.parents) for op in target_ops]
    for child, op in enumerate(target_ops):
        for parent, lag in op.parents.items():
            children[parent].append((child, lag))
    ready = [i for i, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    topo: list[int] = []
    while ready:
        parent = heapq.heappop(ready)
        topo.append(parent)
        for child, _ in children[parent]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(topo) != len(target_ops):
        raise ValueError("target graph is cyclic")

    # Tag/round medians are only used as a soft center.  Explicit mapped DAG
    # boundaries always dominate them.
    tag_values: dict[tuple[object, ...], list[int]] = defaultdict(list)
    round_values: dict[tuple[object, ...], list[int]] = defaultdict(list)
    engine_values: dict[str, list[int]] = defaultdict(list)
    for index, op in enumerate(source_ops):
        cycle = source_cycles[index]
        tag_values[(op.engine, op.tag, op.group, op.round)].append(cycle)
        round_values[(op.engine, op.group, op.round)].append(cycle)
        engine_values[op.engine].append(cycle)

    phase = [0] * len(target_ops)
    for index in topo:
        if index in mapped:
            phase[index] = mapped[index]
            continue
        op = target_ops[index]
        parents = [phase[parent] + lag for parent, lag in op.parents.items()]
        lower = max(parents, default=0)
        child_bounds = [
            phase[child] - lag
            for child, lag in children[index]
            if child in mapped
        ]
        tag = tag_values.get((op.engine, op.tag, op.group, op.round))
        rnd = round_values.get((op.engine, op.group, op.round))
        center = statistics.median(tag or rnd or engine_values.get(op.engine, [0]))
        # A mapped consumer is a hard upper preference; if the rewrite made
        # the source phase interval inconsistent, keep the dependency lower
        # bound and let the real list scheduler move the consumer forward.
        phase[index] = max(lower, min(int(center), min(child_bounds, default=int(center))))

    # One backward pass supplies a useful upper-bound hint to predecessors
    # whose children are all rewritten and therefore were not in the source.
    for index in reversed(topo):
        if index in mapped:
            continue
        child_upper = [phase[child] - lag for child, lag in children[index]]
        if child_upper:
            phase[index] = min(phase[index], min(child_upper))
        if index not in mapped:
            phase[index] = max(0, phase[index])

    out = Path(os.environ.get("OUT", "/tmp/aopt-phase-teacher.json"))
    out.write_text(json.dumps({"cycles": phase, "mapped": len(mapped)}))
    print(f"target_ops={len(target_ops)} mapped={len(mapped)} output={out}")


if __name__ == "__main__":
    main()
