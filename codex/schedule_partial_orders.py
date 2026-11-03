"""Schedule a strict acyclic partial resource order with the production heuristic.

The five exact projected calendars are useful teachers, but hardening all of
their capacity-lane arcs creates cross-engine cycles.  ``partial_order_hints``
contains only a globally acyclic subset.  This tool turns that subset into
ordinary +1 precedence constraints and then runs the proven production list
scheduler on the *augmented* DAG.  It is therefore a fast way to construct a
fully legal incumbent for strict LNS, without ever mistaking independent
engine calendars for a complete schedule.

All emitted schedules are checked against both the original DAG/capacities and
the retained partial arcs.
"""

from __future__ import annotations

from dataclasses import replace
from collections import Counter
import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from codex.partial_order_hints import audit_augmented_dag, load_partial_order_hints
from codex.solve_full_dag import load_hints, validate_solution


def csv_ints(name: str, default: str) -> tuple[int, ...]:
    return tuple(int(value) for value in os.environ.get(name, default).split(",") if value)


def noise_value(index: int, seed: int, amplitude: int) -> int:
    """Stable small perturbation, so a trial is reproducible across processes."""
    word = (seed + 0x9E3779B9 * (index + 1)) & 0xFFFFFFFF
    word ^= word >> 16
    word = (word * 0x7FEB352D) & 0xFFFFFFFF
    word ^= word >> 15
    word = (word * 0x846CA68B) & 0xFFFFFFFF
    word ^= word >> 16
    return word % (2 * amplitude + 1) - amplitude


def build_ops() -> tuple[kernel.KernelBuilder, list[kernel._Op]]:
    configure_target()
    kernel.SCHEDULE_EXACT_CYCLES = None
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        if not hasattr(builder, "dag_ops"):
            raise
    return builder, real_tail_ops(builder.dag_ops)


def selected_edges(
    ops: list[kernel._Op],
) -> tuple[tuple[tuple[int, int], ...], int]:
    paths = tuple(value for value in os.environ.get("PARTIAL_ORDER_HINTS", "").split(",") if value)
    if not paths:
        return (), 0
    partial = load_partial_order_hints(paths, ops)
    audit = audit_augmented_dag(ops, partial)
    stride = int(os.environ.get("PARTIAL_ARC_STRIDE", "1"))
    maximum = int(os.environ.get("PARTIAL_ARC_MAX", "0"))
    if stride <= 0 or maximum < 0:
        raise ValueError("PARTIAL_ARC_STRIDE must be positive and PARTIAL_ARC_MAX non-negative")
    edges = tuple(edge for position, edge in enumerate(partial.effective_edges) if position % stride == 0)
    if maximum:
        edges = edges[:maximum]
    selected = replace(partial, effective_edges=edges)
    selected_audit = audit_augmented_dag(ops, selected)
    print(
        f"partial raw={len(partial.resource_edges)} effective={len(partial.effective_edges)} "
        f"selected={len(edges)} stride={stride} cap={maximum} "
        f"full_span={audit.longest_path_span} selected_span={selected_audit.longest_path_span}",
        flush=True,
    )
    return edges, selected_audit.longest_path_span


def augment(ops: list[kernel._Op], edges: tuple[tuple[int, int], ...]) -> list[kernel._Op]:
    parents = [dict(op.parents) for op in ops]
    for parent, child in edges:
        parents[child][parent] = max(parents[child].get(parent, 0), 1)
    return [replace(op, parents=parents[index]) for index, op in enumerate(ops)]


def validate_augmented(
    ops: list[kernel._Op],
    cycles: list[int],
    partial_edges: tuple[tuple[int, int], ...],
) -> None:
    validate_solution(ops, cycles, max(cycles) + 1)
    for parent, child in partial_edges:
        if cycles[child] < cycles[parent] + 1:
            raise AssertionError(
                f"partial edge {parent}->{child} violated: {cycles[parent]}->{cycles[child]}"
            )


def main() -> None:
    builder, ops = build_ops()
    partial_edges, partial_span = selected_edges(ops)
    augmented = augment(ops, partial_edges)
    # ``real_tail_ops`` changes a handful of post-schedule pointer advances
    # into zero-cost debug placeholders.  The production scheduler is defined
    # only for physical engines, so temporarily place those detached leaves
    # on Store (the intentionally very slack engine).  Their returned cycle
    # is still validated as ``debug`` below and never consumes a real slot.
    scheduler_ops = [
        replace(op, engine="store") if op.engine == "debug" else op
        for op in augmented
    ]
    # A second audit catches an implementation mistake in ``augment`` rather
    # than leaving the list scheduler to fail cryptically on a cyclic graph.
    raw_paths = tuple(value for value in os.environ.get("PARTIAL_ORDER_HINTS", "").split(",") if value)
    if raw_paths:
        partial = load_partial_order_hints(raw_paths, ops)
        audit_augmented_dag(ops, replace(partial, effective_edges=partial_edges))

    hints, hint_counts = load_hints(os.environ.get("HINTS", "").split(","), ops)
    expected_hints = {index for index, op in enumerate(ops) if op.engine != "debug"}
    if hints and set(hints) != expected_hints:
        raise ValueError(
            f"HINTS must cover every non-debug operation, got {len(hints)}/{len(expected_hints)}"
        )
    targets = [hints.get(index, 0) for index in range(len(ops))]
    policies = csv_ints("POLICIES", "0,1,2,3")
    cycle_weights = csv_ints("CYCLE_WEIGHTS", "0,1,2,4,8")
    height_weights = csv_ints("HEIGHT_WEIGHTS", "1,2,4,8")
    amplitudes = csv_ints("AMPLITUDES", "0,1,2,4")
    seeds = int(os.environ.get("SEEDS", "4"))
    start = int(os.environ.get("START", "0"))
    limit = int(os.environ.get("LIMIT", "0"))
    output = Path(os.environ.get("OUT", "/tmp/aopt-partial-order-schedule.json"))

    best: tuple[int, tuple[int, int, int, int, int], list[int]] | None = None
    evaluated = 0
    for cycle_weight in cycle_weights:
        for height_weight in height_weights:
            for policy in policies:
                for amplitude in amplitudes:
                    for seed in range(start, start + seeds):
                        if limit and evaluated >= limit:
                            break
                        scores = (
                            [
                                -cycle_weight * targets[index]
                                + noise_value(index, seed, amplitude)
                                for index in range(len(ops))
                            ]
                            if cycle_weight
                            else None
                        )
                        _, cycles = builder._schedule(
                            scheduler_ops,
                            policy,
                            return_cycles=True,
                            external_scores=scores,
                            height_weight=height_weight,
                        )
                        validate_augmented(ops, cycles, partial_edges)
                        evaluated += 1
                        makespan = max(cycles) + 1
                        params = (cycle_weight, height_weight, policy, amplitude, seed)
                        if best is None or (makespan, params) < (best[0], best[1]):
                            best = (makespan, params, cycles)
                            print(
                                f"best={makespan} cycle_weight={cycle_weight} "
                                f"height_weight={height_weight} policy={policy} "
                                f"amplitude={amplitude} seed={seed}",
                                flush=True,
                            )
                    if limit and evaluated >= limit:
                        break
                if limit and evaluated >= limit:
                    break
            if limit and evaluated >= limit:
                break
        if limit and evaluated >= limit:
            break
    if best is None:
        raise ValueError("no scheduling trials requested")

    makespan, params, cycles = best
    usage = Counter((cycle, op.engine) for cycle, op in zip(cycles, ops))
    holes = {
        engine: sum(
            kernel.SLOT_LIMITS[engine] - usage[cycle, engine]
            for cycle in range(makespan)
        )
        for engine in kernel.SLOT_LIMITS
        if engine != "debug"
    }
    output.write_text(
        json.dumps(
            {
                "makespan": makespan,
                "cycles": cycles,
                "partial_edges": len(partial_edges),
                "partial_span": partial_span,
                "hint_counts": hint_counts,
                "params": {
                    "cycle_weight": params[0],
                    "height_weight": params[1],
                    "policy": params[2],
                    "amplitude": params[3],
                    "seed": params[4],
                },
            }
        )
    )
    print(
        f"evaluated={evaluated} best={makespan} holes={holes} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
