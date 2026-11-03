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
    if op.identity:
        return (op.engine, "identity", op.group, op.round, op.identity)
    opcode = op.slot[0] if op.slot else None
    return (op.engine, opcode, op.tag, op.group, op.round)


def semantic_key_without_engine(op: kernel._Op) -> tuple[object, ...]:
    if op.identity:
        return ("identity", op.group, op.round, op.identity)
    opcode = op.slot[0] if op.slot else None
    return (opcode, op.tag, op.group, op.round)


def ordered_earliest(
    ops: list[kernel._Op],
    engine: str,
    cycles: dict[int, int],
) -> list[int]:
    """Materialize all-node earliest times under one fixed engine order."""
    parents = [dict(op.parents) for op in ops]
    selected = sorted(
        (i for i, op in enumerate(ops) if op.engine == engine),
        key=lambda i: (cycles[i], i),
    )
    capacity = SLOT_LIMITS[engine]
    for previous, current in zip(selected, selected[capacity:]):
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
    topological: list[int] = []
    while ready:
        parent = heapq.heappop(ready)
        topological.append(parent)
        for child, _ in children[parent]:
            indegree[child] -= 1
            if not indegree[child]:
                heapq.heappush(ready, child)
    if len(topological) != len(ops):
        raise ValueError("source engine order is cyclic")
    earliest = [0] * len(ops)
    for child in topological:
        earliest[child] = max(
            (
                earliest[parent] + lag
                for parent, lag in parents[child].items()
            ),
            default=0,
        )
    return earliest


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
    # The production module's embedded 987-cycle schedule belongs to the
    # module-default DAG.  Target variants are normally selected through
    # environment variables, so allow the source build to happen before
    # those variables mutate the imported module globals.
    if not bool(int(os.environ.get("SOURCE_USE_MODULE_DEFAULTS", "0"))):
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
    cross_engine = bool(int(os.environ.get("MATCH_CROSS_ENGINE", "0")))
    source_ordered_earliest = (
        ordered_earliest(source_ops, engine, source_cycles)
        if cross_engine and projected_source
        else None
    )
    cross_candidates: dict[
        tuple[object, ...], list[int]
    ] = defaultdict(list)
    if source_ordered_earliest is not None:
        for i, op in enumerate(source_ops):
            if op.engine != engine:
                cross_candidates[semantic_key_without_engine(op)].append(i)
    cross_candidate_queues = {
        key: deque(indices)
        for key, indices in cross_candidates.items()
    }

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
    cross_matched_indices: list[int] = []
    cross_match_sources: dict[int, int] = {}
    unmatched_indices: list[int] = []
    unmatched_front_tags = frozenset(
        value
        for value in os.environ.get(
            "UNMATCHED_FRONT_TAGS", ""
        ).split(",")
        if value
    )
    unmatched_last = bool(int(os.environ.get("UNMATCHED_LAST", "0")))
    matched = 0
    for i, op in enumerate(target_ops):
        if op.engine != engine:
            continue
        candidates = source_by_key[semantic_key(op)]
        if candidates:
            major = candidates.popleft()
            matched += 1
            matched_indices.add(i)
        elif (
            source_ordered_earliest is not None
            and (
                cross_queue := cross_candidate_queues.get(
                    semantic_key_without_engine(op)
                )
            )
        ):
            source_index = cross_queue.popleft()
            major = source_ordered_earliest[source_index]
            cross_matched_indices.append(i)
            cross_match_sources[i] = source_index
        else:
            unmatched_indices.append(i)
            major = (
                source_span + 1
                if unmatched_last
                else -1
                if op.tag in unmatched_front_tags
                else round(
                    fallback[i]
                    * (source_span - 1)
                    / max(1, fallback_span - 1)
                )
            )
        scores[i] = major * (len(target_ops) + 1) + i

    unmatched_after_tag = os.environ.get("UNMATCHED_AFTER_TAG", "")
    unmatched_after_group_round = os.environ.get(
        "UNMATCHED_AFTER_GROUP_ROUND", ""
    )
    if (unmatched_after_tag or unmatched_after_group_round) and unmatched_indices:
        anchor_group_round = (
            tuple(
                int(component)
                for component in unmatched_after_group_round.split(":")
            )
            if unmatched_after_group_round
            else None
        )
        if (
            anchor_group_round is not None
            and len(anchor_group_round) != 2
        ):
            raise ValueError(
                "UNMATCHED_AFTER_GROUP_ROUND must be group:round"
            )
        anchor_scores = [
            scores[i]
            for i, op in enumerate(target_ops)
            if op.engine == engine
            and (
                op.tag == unmatched_after_tag
                if unmatched_after_tag
                else (op.group, op.round) == anchor_group_round
            )
            and i in scores
            and i not in unmatched_indices
        ]
        if not anchor_scores:
            anchor_description = (
                f"tag {unmatched_after_tag}"
                if unmatched_after_tag
                else f"group/round {anchor_group_round}"
            )
            raise ValueError(
                f"missing matched insertion anchor {anchor_description}"
            )
        anchor_score = max(anchor_scores)
        for offset, i in enumerate(unmatched_indices, start=1):
            scores[i] = anchor_score + offset

    # Project the preferred scores onto a legal topological extension of the
    # target DAG.  Directly copying an old total order can contradict a new
    # dependency introduced by a rewrite; Kahn priority preserves as much of
    # the old order as possible without ever creating a cycle.
    target_parents = [dict(op.parents) for op in target_ops]
    if bool(int(os.environ.get("PRESERVE_MATCHED_ENGINE_ORDER", "0"))):
        matched_sequence = sorted(
            matched_indices,
            key=lambda i: (scores[i], i),
        )
        capacity = SLOT_LIMITS[engine]
        for previous, current in zip(
            matched_sequence,
            matched_sequence[capacity:],
        ):
            target_parents[current][previous] = max(
                target_parents[current].get(previous, 0),
                1,
            )

    children: list[list[int]] = [[] for _ in target_ops]
    indegree = [len(item) for item in target_parents]
    for child, child_parents in enumerate(target_parents):
        for parent in child_parents:
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
    if bool(int(os.environ.get("OUTPUT_SOURCE_PHASES", "0"))):
        # Soft CP-SAT hints need the inherited source cycle, not the large
        # lexicographic score used to rank operations.  These values are not
        # required to form a legal target schedule: graph rewrites can create
        # collisions, and the joint solver repairs them while retaining the
        # incumbent's global phase structure.
        score_scale = len(target_ops) + 1
        order_scores = {
            i: score // score_scale for i, score in scores.items()
        }
    elif bool(int(os.environ.get("PRESERVE_ENGINE_ORDER", "0"))):
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
                "horizon": source_span,
                "matched": matched,
                "matched_indices": sorted(matched_indices),
                "cross_matched_indices": cross_matched_indices,
                "unmatched_indices": unmatched_indices,
                "cross_matched": [
                    {
                        "index": i,
                        "source_index": cross_match_sources[i],
                        "source_engine": source_ops[
                            cross_match_sources[i]
                        ].engine,
                        "source_cycle": source_ordered_earliest[
                            cross_match_sources[i]
                        ],
                        "opcode": target_ops[i].slot[0]
                        if target_ops[i].slot
                        else None,
                        "tag": target_ops[i].tag,
                        "group": target_ops[i].group,
                        "round": target_ops[i].round,
                    }
                    for i in cross_matched_indices
                ],
                "unmatched": [
                    {
                        "index": i,
                        "opcode": target_ops[i].slot[0] if target_ops[i].slot else None,
                        "tag": target_ops[i].tag,
                        "group": target_ops[i].group,
                        "round": target_ops[i].round,
                        "fallback_cycle": fallback[i],
                        "source_same_index": (
                            {
                                "engine": source_ops[i].engine,
                                "opcode": source_ops[i].slot[0]
                                if source_ops[i].slot
                                else None,
                                "tag": source_ops[i].tag,
                                "group": source_ops[i].group,
                                "round": source_ops[i].round,
                            }
                            if i < len(source_ops)
                            else None
                        ),
                        "cross_candidates": [
                            {
                                "index": source_index,
                                "engine": source_ops[source_index].engine,
                                "cycle": source_ordered_earliest[source_index],
                            }
                            for source_index in cross_candidates.get(
                                semantic_key_without_engine(target_ops[i]), ()
                            )
                        ],
                    }
                    for i in unmatched_indices
                ],
                "cycles": {str(i): score for i, score in order_scores.items()},
            }
        )
    )
    print(
        f"engine={engine} matched={matched}/{len(scores)} output={output}"
    )


if __name__ == "__main__":
    main()
