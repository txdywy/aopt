"""Explain pure-DAG conflicts caused by independently generated engine hints.

Each per-engine 959 schedule can be perfectly valid in isolation while being
globally impossible: an ALU operation can be constrained late by its own hint
and a dependent Flow operation constrained early by another hint.  Before
blaming a resource capacity constraint, this tool propagates the exact lower
and upper time windows through *every original DAG edge*.  A non-empty
contradiction set is therefore a mathematical certificate that the selected
hint-radius neighborhood cannot contain a legal global schedule, even on an
infinite-width machine.

The tool is deliberately read-only.  It shares graph construction and hint
validation with ``solve_full_dag`` but creates no CP-SAT model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from codex.solve_full_dag import load_hints, topological_data


@dataclass(frozen=True)
class Anchor:
    """An initial hint bound that reaches a propagated bound."""

    index: int
    side: str
    hinted_cycle: int


def ancestors(
    index: int,
    reason: list[int | None],
    anchors: list[Anchor | None],
) -> tuple[Anchor | None, list[int]]:
    """Trace a propagated bound back to its one initial hint bound."""
    path = [index]
    seen = {index}
    while reason[index] is not None:
        index = reason[index]
        if index in seen:
            raise AssertionError("DAG bound reason cycle")
        seen.add(index)
        path.append(index)
    return anchors[index], path


def describe(index: int, ops: list[kernel._Op]) -> str:
    op = ops[index]
    return (
        f"i={index} {op.engine} g={op.group} r={op.round} {op.tag}"
    )


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
    topological, children, earliest, tail = topological_data(ops)
    horizon = int(os.environ.get("TARGET", "959"))
    radius = int(os.environ.get("HINT_RADIUS", "64"))
    if radius < 0:
        raise ValueError("HINT_RADIUS must be non-negative for conflict analysis")
    hints, _ = load_hints(os.environ.get("HINTS", "").split(","), ops)
    if not hints:
        raise ValueError("HINTS is required")

    lower = earliest.copy()
    upper = [horizon - 1 - value for value in tail]
    lower_reason: list[int | None] = [None] * len(ops)
    upper_reason: list[int | None] = [None] * len(ops)
    lower_anchor: list[Anchor | None] = [None] * len(ops)
    upper_anchor: list[Anchor | None] = [None] * len(ops)
    tightened = Counter()
    for index, cycle in hints.items():
        lower_bound = max(lower[index], cycle - radius)
        upper_bound = min(upper[index], cycle + radius)
        if lower_bound > lower[index]:
            lower[index] = lower_bound
            lower_anchor[index] = Anchor(index, "lower", cycle)
            tightened["lower"] += 1
        if upper_bound < upper[index]:
            upper[index] = upper_bound
            upper_anchor[index] = Anchor(index, "upper", cycle)
            tightened["upper"] += 1

    # The graph is acyclic.  One forward and one backward topological sweep
    # compute the exact longest lower-bound and shortest upper-bound paths.
    for child in topological:
        for parent, lag in ops[child].parents.items():
            candidate = lower[parent] + lag
            if candidate > lower[child]:
                lower[child] = candidate
                lower_reason[child] = parent
                lower_anchor[child] = lower_anchor[parent]
    for parent in reversed(topological):
        for child, lag in children[parent]:
            candidate = upper[child] - lag
            if candidate < upper[parent]:
                upper[parent] = candidate
                upper_reason[parent] = child
                upper_anchor[parent] = upper_anchor[child]

    conflicts = [index for index in range(len(ops)) if lower[index] > upper[index]]
    print(
        f"full_hint_bounds ops={len(ops)} edges="
        f"{sum(len(op.parents) for op in ops)} horizon={horizon} radius={radius} "
        f"hints={len(hints)} tightened={dict(tightened)} "
        f"dag_conflicts={len(conflicts)}",
        flush=True,
    )
    if not conflicts:
        print(
            "result=NO_PURE_DAG_CONFLICT "
            "(any CP-SAT infeasibility is resource/combined-window related)",
            flush=True,
        )
        return

    # Group certificates by the two concrete engine anchors.  This makes it
    # clear whether the incompatibility is an internal order mismatch or a
    # cross-engine phase mismatch.
    pairs: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], int] = {}
    for index in conflicts:
        low_anchor, _ = ancestors(index, lower_reason, lower_anchor)
        high_anchor, _ = ancestors(index, upper_reason, upper_anchor)
        left = ops[low_anchor.index].engine if low_anchor is not None else "base"
        right = ops[high_anchor.index].engine if high_anchor is not None else "base"
        key = (left, right)
        pairs[key] += 1
        previous = examples.get(key)
        if previous is None or lower[index] - upper[index] < lower[previous] - upper[previous]:
            examples[key] = index
    print(
        "anchor_engine_pairs="
        + ", ".join(
            f"{left}->{right}:{count}@i={examples[left, right]}"
            for (left, right), count in pairs.most_common(16)
        ),
        flush=True,
    )

    limit = int(os.environ.get("LIMIT", "16"))
    # Prefer the smallest contradiction first: it is the minimum local phase
    # displacement that must be released before this particular certificate
    # can disappear.  Then prefer a short causal chain.
    ordered: list[tuple[int, int, int, int, list[int], list[int], Anchor | None, Anchor | None]] = []
    for index in conflicts:
        low_anchor, low_path = ancestors(index, lower_reason, lower_anchor)
        high_anchor, high_path = ancestors(index, upper_reason, upper_anchor)
        ordered.append(
            (
                lower[index] - upper[index],
                len(low_path) + len(high_path),
                index,
                len(low_path),
                low_path,
                high_path,
                low_anchor,
                high_anchor,
            )
        )
    ordered.sort()
    for gap, _, index, low_len, low_path, high_path, low_anchor, high_anchor in ordered[:limit]:
        low_text = (
            describe(low_anchor.index, ops)
            + f" hint={low_anchor.hinted_cycle} lower={low_anchor.hinted_cycle - radius}"
            if low_anchor is not None
            else "unhinted exact-release bound"
        )
        high_text = (
            describe(high_anchor.index, ops)
            + f" hint={high_anchor.hinted_cycle} upper={high_anchor.hinted_cycle + radius}"
            if high_anchor is not None
            else "unhinted exact-deadline bound"
        )
        print(
            f"certificate gap={gap} pivot=[{describe(index, ops)}] "
            f"window={lower[index]}:{upper[index]} "
            f"lower_anchor=[{low_text}] path_len={low_len - 1} "
            f"upper_anchor=[{high_text}] path_len={len(high_path) - 1}",
            flush=True,
        )
        if bool(int(os.environ.get("PRINT_PATHS", "0"))):
            print(
                "  lower_path=" + " -> ".join(describe(item, ops) for item in reversed(low_path)),
                flush=True,
            )
            print(
                "  upper_path=" + " -> ".join(describe(item, ops) for item in high_path),
                flush=True,
            )


if __name__ == "__main__":
    main()
