"""Transfer one engine's semantic order from a known global schedule."""

from __future__ import annotations

from collections import defaultdict, deque
import heapq
import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from problem import SLOT_LIMITS


def semantic_key(op: kernel._Op) -> tuple[object, ...]:
    opcode = op.slot[0] if op.slot else None
    return (op.engine, opcode, op.tag, op.group, op.round)


def main() -> None:
    source_payload = json.loads(Path(os.environ["SOURCE"]).read_text())
    source_cycles = source_payload["cycles"]
    projected_source = isinstance(source_cycles, dict)

    # A graph rewrite can change operation indices even when the semantic
    # operations shared by both variants are identical.  Allow the source
    # graph to use selected configuration values that differ from the target,
    # then restore the target environment before constructing its DAG.  This
    # is intentionally opt-in and generic: ``SOURCE_CONFIG_FOO=x`` means the
    # source build sees ``FOO=x`` while the target keeps its normal ``FOO``.
    source_config = {
        key.removeprefix("SOURCE_CONFIG_"): value
        for key, value in os.environ.items()
        if key.startswith("SOURCE_CONFIG_")
    }
    target_config = {
        key: os.environ.get(key)
        for key in source_config
    }
    for key, value in source_config.items():
        os.environ[key] = value
    configure_target()
    if "SOURCE_HASH_SCALAR_EXTRA_COUNT" in os.environ:
        count = int(os.environ["SOURCE_HASH_SCALAR_EXTRA_COUNT"])
        kernel.HASH_SCALAR_EXTRA = frozenset(
            kernel._BASE_SCALAR | set(kernel._SCALAR_CANDIDATES[:count])
        )
    kernel.SCHEDULE_EXACT_CYCLES = None if projected_source else source_cycles
    source_builder = kernel.KernelBuilder()
    try:
        source_builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        if not hasattr(source_builder, "dag_ops"):
            raise
    source_ops = real_tail_ops(source_builder.dag_ops)
    if not projected_source and len(source_ops) != len(source_cycles):
        raise ValueError("source schedule does not match source graph")

    engine = os.environ.get("ENGINE", "flow")
    if projected_source:
        source_cycles = {
            int(index): int(cycle)
            for index, cycle in source_cycles.items()
        }
        source_expected = {
            i for i, op in enumerate(source_ops) if op.engine == engine
        }
        if set(source_cycles) != source_expected:
            raise ValueError("projected source does not match source graph")
    source_by_key: dict[tuple[object, ...], deque[int]] = defaultdict(deque)
    for i, op in enumerate(source_ops):
        if op.engine == engine:
            source_by_key[semantic_key(op)].append(source_cycles[i])

    for key, value in target_config.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    kernel.SCHEDULE_EXACT_CYCLES = None
    configure_target()
    target_builder = kernel.KernelBuilder()
    try:
        target_builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        if not hasattr(target_builder, "dag_ops"):
            raise
    target_ops = real_tail_ops(target_builder.dag_ops)
    _, fallback = target_builder._schedule(
        target_builder.dag_ops, int(os.environ.get("POLICY", "4")),
        return_cycles=True,
    )

    # Preserve source cycles as the major key.  New operations use the target
    # list schedule projected into the source span, and a tiny target-order
    # fraction gives deterministic placement among same-cycle source ops.
    source_span = max(source_cycles.values() if projected_source else source_cycles) + 1
    fallback_span = max(fallback) + 1
    scores: dict[int, int] = {}
    matched_indices: set[int] = set()
    matched = 0
    for i, op in enumerate(target_ops):
        if op.engine != engine:
            continue
        candidates = source_by_key[semantic_key(op)]
        if candidates:
            major = candidates.popleft()
            matched += 1
            matched_indices.add(i)
        else:
            major = round(fallback[i] * (source_span - 1) / max(1, fallback_span - 1))
        scores[i] = major * (len(target_ops) + 1) + i

    # Project the preferred scores onto a legal topological extension of the
    # target DAG.  Directly copying an old total order can contradict a new
    # dependency introduced by a rewrite; Kahn priority preserves as much of
    # the old order as possible without ever creating a cycle.
    children: list[list[int]] = [[] for _ in target_ops]
    indegree = [len(op.parents) for op in target_ops]
    for child, op in enumerate(target_ops):
        for parent in op.parents:
            children[parent].append(child)
    ready: list[tuple[int, int]] = []
    for i, degree in enumerate(indegree):
        if not degree:
            fallback_score = fallback[i] * (len(target_ops) + 1) + i
            heapq.heappush(ready, (scores.get(i, fallback_score), i))
    topological = []
    while ready:
        _, parent = heapq.heappop(ready)
        topological.append(parent)
        for child in children[parent]:
            indegree[child] -= 1
            if not indegree[child]:
                fallback_score = fallback[child] * (len(target_ops) + 1) + child
                heapq.heappush(
                    ready, (scores.get(child, fallback_score), child)
                )
    if len(topological) != len(target_ops):
        raise ValueError("target graph is cyclic")
    if bool(int(os.environ.get("PRESERVE_ENGINE_ORDER", "0"))):
        order_scores = scores
    else:
        order_scores = {
            i: position
            for position, i in enumerate(topological)
            if target_ops[i].engine == engine
        }
    if bool(int(os.environ.get("NORMALIZE_ENGINE_CYCLES", "0"))):
        sequence = sorted(order_scores, key=lambda i: (order_scores[i], i))
        capacity = SLOT_LIMITS[engine]
        order_scores = {
            i: position // capacity
            for position, i in enumerate(sequence)
        }

    output = Path(os.environ.get("OUT", f"/tmp/aopt-{engine}-mapped-order.json"))
    output.write_text(
        json.dumps(
            {
                "engine": engine,
                "source": os.environ["SOURCE"],
                "matched": matched,
                "matched_indices": sorted(matched_indices),
                "cycles": {str(i): score for i, score in order_scores.items()},
            }
        )
    )
    print(
        f"engine={engine} matched={matched}/{len(scores)} output={output}"
    )


if __name__ == "__main__":
    main()
