"""Rank nodes moved between STORE+LOAD staging and VALU broadcast.

The original search only removed staged nodes.  ``ADD_COUNT`` and
``ADD_POOL`` search the inverse transformation, which is useful after a tree
cache rewrite frees load slots and needs to recover VALU capacity.
"""

from __future__ import annotations

from itertools import combinations
import json
import os

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target
from codex.search_memory_buffer_hall import build_ops, evaluate, score


def main() -> None:
    configure_target()
    horizon = int(os.environ.get("TARGET", "959"))
    base = frozenset(kernel.MEMORY_CACHED_NODE_SET)
    base_alu = frozenset(kernel.ALU_CACHED_NODE_SET)
    candidates = tuple(sorted(base))
    add_count = int(os.environ.get("ADD_COUNT", "0"))
    alu_count = int(os.environ.get("ALU_COUNT", "0"))
    remove_count = int(os.environ.get("REMOVE_COUNT", "1"))
    if sum(bool(value) for value in (add_count, alu_count)) > 1:
        raise ValueError("choose ADD_COUNT, ALU_COUNT, or REMOVE_COUNT")
    if (add_count or alu_count) and "REMOVE_COUNT" in os.environ:
        raise ValueError("choose ADD_COUNT/ALU_COUNT or REMOVE_COUNT")
    if alu_count:
        pool = tuple(
            int(value)
            for value in os.environ.get(
                "ALU_POOL", ",".join(map(str, candidates))
            ).split(",")
            if value
        )
        move_count = alu_count
    elif add_count:
        pool = tuple(
            int(value)
            for value in os.environ.get(
                "ADD_POOL",
                ",".join(map(str, sorted(set(range(31)) - base))),
            ).split(",")
            if value
        )
        move_count = add_count
    else:
        pool = tuple(
            int(value)
            for value in os.environ.get(
                "REMOVE_POOL",
                ",".join(map(str, candidates)),
            ).split(",")
            if value
        )
        move_count = remove_count
    limit = int(os.environ.get("COMBINATION_LIMIT", "0"))
    results = []
    for attempt, moved in enumerate(combinations(pool, move_count), 1):
        if limit and attempt > limit:
            break
        if alu_count:
            kernel.MEMORY_CACHED_NODE_SET = base - frozenset(moved)
            kernel.ALU_CACHED_NODE_SET = base_alu | frozenset(moved)
        elif add_count:
            kernel.MEMORY_CACHED_NODE_SET = base | frozenset(moved)
        else:
            kernel.MEMORY_CACHED_NODE_SET = base - frozenset(moved)
        try:
            result = evaluate(build_ops(), horizon)
        except (AssertionError, StopIteration, ValueError) as error:
            result = {"error": type(error).__name__ + ": " + str(error)}
        result[
            "alu_nodes" if alu_count else "added" if add_count else "removed"
        ] = moved
        results.append(result)
    results.sort(
        key=lambda result: score(result)
        if "error" not in result
        else (10**9,),
    )
    result_limit = int(os.environ.get("RESULT_LIMIT", "0"))
    for result in results[:result_limit or None]:
        print(json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
