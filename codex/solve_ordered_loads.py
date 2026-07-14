"""Reorder the saturated load port while preserving proven other-engine orders.

The incumbent schedule's critical path is dominated by its load resource
chain.  Fixing ALU/VALU/flow/store relative order turns the full RCPSP into a
much smaller cumulative scheduling problem over the two load slots.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import replace
import json
import os
from pathlib import Path

from ortools.sat.python import cp_model

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import real_tail_ops, validate
from problem import SLOT_LIMITS


def group_set(name: str) -> frozenset[int]:
    return frozenset(
        int(value)
        for value in os.environ.get(name, "").split(",")
        if value
    )


def pair_set(name: str) -> frozenset[tuple[int, int]]:
    return frozenset(
        tuple(int(component) for component in item.split(":"))
        for item in os.environ.get(name, "").split(",")
        if item
    )


def direct_lookup_map(name: str) -> dict[int, tuple[int, ...]]:
    return {
        int(group): tuple(range(int(count)))
        for group, count in (
            item.split(":")
            for item in os.environ.get(name, "").split(",")
            if item
        )
    }


def hybrid_override_map(name: str) -> dict[tuple[int, int], int]:
    return {
        (int(group), int(rnd)): int(count)
        for group, rnd, count in (
            item.split(":")
            for item in os.environ.get(name, "").split(",")
            if item
        )
    }


def configure() -> None:
    kernel.SCHEDULE_EXACT_CYCLES = None
    for env_name, attribute in (
        ("SCALAR_FINAL_C5", "SCALAR_FINAL_C5_SET"),
        ("SCALAR_FINAL_JOIN", "SCALAR_FINAL_JOIN_SET"),
        ("SCALAR_FINAL_SHIFT", "SCALAR_FINAL_SHIFT_SET"),
        ("SCALAR_FINAL_HASH23_JOIN", "SCALAR_FINAL_HASH23_JOIN_SET"),
        ("SCALAR_FINAL_HASH4", "SCALAR_FINAL_HASH4_SET"),
    ):
        if env_name in os.environ:
            setattr(kernel, attribute, group_set(env_name))
    for env_name, attribute in (
        ("SCALAR_HASH1_JOIN", "SCALAR_HASH1_JOIN_SET"),
        ("SCALAR_HASH23_JOIN", "SCALAR_HASH23_JOIN_SET"),
        ("SCALAR_HASH5_JOIN", "SCALAR_HASH5_JOIN_SET"),
    ):
        if env_name in os.environ:
            setattr(kernel, attribute, pair_set(env_name))
    if "SAVED_SECOND_PATH_EXTRA_GROUPS" in os.environ:
        kernel.SAVED_SECOND_PATH_EXTRA_GROUPS = group_set(
            "SAVED_SECOND_PATH_EXTRA_GROUPS"
        )
    if "HASH_SCALAR_EXTRA_REMOVE" in os.environ:
        kernel.HASH_SCALAR_EXTRA = frozenset(
            set(kernel.HASH_SCALAR_EXTRA)
            - set(pair_set("HASH_SCALAR_EXTRA_REMOVE"))
        )
    if "CHAINED_DIRECT_BRANCH_BASE" in os.environ:
        kernel.CHAINED_DIRECT_BRANCH_BASE = bool(
            int(os.environ["CHAINED_DIRECT_BRANCH_BASE"])
        )
    if "DIRECT_BRANCH_LOOKUPS" in os.environ:
        kernel.DIRECT_BRANCH_LOOKUPS = direct_lookup_map("DIRECT_BRANCH_LOOKUPS")
    if "FLOW_OUTPUT_ADVANCE_POSITIONS" in os.environ:
        kernel.FLOW_OUTPUT_ADVANCE_POSITIONS = group_set(
            "FLOW_OUTPUT_ADVANCE_POSITIONS"
        )
    if "FINAL_CACHE_SET" in os.environ:
        kernel.FINAL_CACHE_SET = group_set("FINAL_CACHE_SET")
    if "EARLY_FINAL_CACHE_SET" in os.environ:
        kernel.EARLY_FINAL_CACHE_SET = group_set("EARLY_FINAL_CACHE_SET")
    if "VECTOR_PARITY_SET" in os.environ:
        kernel.VECTOR_PARITY_SET = pair_set("VECTOR_PARITY_SET")
    if "VECTOR_NODE_XOR_SET" in os.environ:
        kernel.VECTOR_NODE_XOR_SET = pair_set("VECTOR_NODE_XOR_SET")
    if "HYBRID_MADD_OVERRIDES" in os.environ:
        kernel.HYBRID_MADD_OVERRIDES = hybrid_override_map(
            "HYBRID_MADD_OVERRIDES"
        )


def main() -> None:
    hint = json.loads(Path(os.environ["HINT"]).read_text())["cycles"]
    horizon = int(os.environ.get("TARGET", "970"))
    configure()
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        if not hasattr(builder, "dag_ops"):
            raise
    ops = real_tail_ops(builder.dag_ops)
    if len(hint) != len(ops):
        raise ValueError(f"hint length {len(hint)} != DAG {len(ops)}")
    validate(ops, hint)

    ordered_engines = frozenset(
        value
        for value in os.environ.get(
            "ORDERED_ENGINES", "alu,valu,flow,store"
        ).split(",")
        if value
    )
    unlock_groups = group_set("UNLOCK_GROUPS")
    unlock_engines = frozenset(
        value
        for value in os.environ.get(
            "UNLOCK_ENGINES", "alu,valu,load,flow,store"
        ).split(",")
        if value
    )
    unlock_cycle_start = int(os.environ.get("UNLOCK_CYCLE_START", "-1"))
    unlock_cycle_stop = int(os.environ.get("UNLOCK_CYCLE_STOP", str(max(hint) + 1)))
    unlock_group_round_start = int(
        os.environ.get("UNLOCK_GROUP_ROUND_START", "-1")
    )
    unlock_group_round_stop = int(
        os.environ.get("UNLOCK_GROUP_ROUND_STOP", "1000000")
    )
    unlocked = {
        i
        for i, op in enumerate(ops)
        if op.engine in unlock_engines
        and (
            (
                op.group in unlock_groups
                and (
                    unlock_group_round_start < 0
                    or unlock_group_round_start
                    <= op.round
                    < unlock_group_round_stop
                )
            )
            or (
                unlock_cycle_start >= 0
                and unlock_cycle_start <= hint[i] < unlock_cycle_stop
            )
        )
    }
    if "UNLOCK_INDICES_FILE" in os.environ:
        unlocked.update(
            json.loads(Path(os.environ["UNLOCK_INDICES_FILE"]).read_text())[
                "indices"
            ]
        )
    unlocked_engines = frozenset(ops[i].engine for i in unlocked)
    print(
        f"ops={len(ops)} unlocked={len(unlocked)} "
        f"unlocked_engines={sorted(unlocked_engines)}",
        flush=True,
    )
    parents = [dict(op.parents) for op in ops]
    for engine in ordered_engines:
        capacity = SLOT_LIMITS[engine]
        sequence = sorted(
            (i for i, op in enumerate(ops) if op.engine == engine),
            key=lambda i: (hint[i], i),
        )
        for position in range(capacity, len(sequence)):
            parent = sequence[position - capacity]
            child = sequence[position]
            if parent in unlocked or child in unlocked:
                continue
            parents[child][parent] = max(parents[child].get(parent, 0), 1)

    old_span = max(hint) or 1
    scaled_hints = [
        min(horizon - 1, round(cycle * (horizon - 1) / old_span))
        for cycle in hint
    ]
    domain_radius = int(os.environ.get("DOMAIN_RADIUS", "-1"))
    model = cp_model.CpModel()
    starts = []
    for i, scaled in enumerate(scaled_hints):
        if domain_radius >= 0:
            lower = max(0, scaled - domain_radius)
            upper = min(horizon - 1, scaled + domain_radius)
        else:
            lower, upper = 0, horizon - 1
        starts.append(model.new_int_var(lower, upper, f"s{i}"))
    for child, child_parents in enumerate(parents):
        for parent, lag in child_parents.items():
            model.add(starts[child] >= starts[parent] + lag)

    for engine, capacity in SLOT_LIMITS.items():
        if engine == "debug" or (
            engine in ordered_engines and engine not in unlocked_engines
        ):
            continue
        intervals = [
            model.new_fixed_size_interval_var(starts[i], 1, f"i{i}")
            for i, op in enumerate(ops)
            if op.engine == engine
        ]
        model.add_cumulative(intervals, [1] * len(intervals), capacity)

    copy_tags = {
        "direct_branch_copy",
        "paired_direct_branch_copy",
        "paired_branch_copy",
        "paired_branch_delayed_copy",
    }
    target_tags = {
        "direct_branch_target",
        "paired_direct_branch_target",
        "paired_branch_target",
        "paired_branch_delayed_target",
    }
    for child, op in enumerate(ops):
        if op.tag in copy_tags:
            jump_parent = next(
                (
                    parent
                    for parent in op.parents
                    if ops[parent].tag.endswith("branch_jump")
                ),
                None,
            )
            if jump_parent is not None:
                model.add(starts[child] == starts[jump_parent] + 1)
        elif op.tag in target_tags:
            for parent in op.parents:
                if ops[parent].tag in copy_tags:
                    model.add(starts[child] == starts[parent])

    makespan_var = None
    if bool(int(os.environ.get("OPTIMIZE", "0"))):
        makespan_var = model.new_int_var(0, horizon - 1, "last_cycle")
        model.add_max_equality(makespan_var, starts)
        model.minimize(makespan_var)

    for i, scaled in enumerate(scaled_hints):
        model.add_hint(starts[i], scaled)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(
        os.environ.get("TIME_LIMIT", "300")
    )
    solver.parameters.num_workers = int(os.environ.get("WORKERS", "8"))
    solver.parameters.repair_hint = True
    solver.parameters.hint_conflict_limit = 500_000
    solver.parameters.random_seed = int(os.environ.get("RANDOM_SEED", "1"))
    solver.parameters.log_search_progress = bool(int(os.environ.get("LOG", "0")))
    status = solver.solve(model)
    bound_text = (
        f" objective={solver.objective_value:g} bound={solver.best_objective_bound:g}"
        if makespan_var is not None
        and status in (cp_model.FEASIBLE, cp_model.OPTIMAL)
        else ""
    )
    print("status", solver.status_name(status) + bound_text, flush=True)
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return
    result = [solver.value(start) for start in starts]
    validate(ops, result)
    output = Path(os.environ.get("OUT", "/tmp/aopt-ordered-loads.json"))
    output.write_text(json.dumps({"makespan": max(result) + 1, "cycles": result}))
    print(f"makespan={max(result) + 1} output={output}", flush=True)


if __name__ == "__main__":
    main()
