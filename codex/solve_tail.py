"""Exact epilogue rescheduler with the steady-state prefix held fixed."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from dataclasses import replace

from ortools.sat.python import cp_model

import codex.perf_takehome_under1000 as kernel
from problem import SLOT_LIMITS


def main() -> None:
    cutoff = int(os.environ.get("CUTOFF", "850"))
    horizon = int(os.environ.get("TARGET", "999"))
    limit = float(os.environ.get("TIME_LIMIT", "180"))
    early_slack = int(os.environ.get("EARLY_SLACK", "0"))
    ancestor_steps = int(os.environ.get("ANCESTOR_STEPS", "0"))
    if "ALU_DERIVED_MISC_SET" in os.environ:
        kernel.ALU_DERIVED_MISC_SET = frozenset(
            value
            for value in os.environ["ALU_DERIVED_MISC_SET"].split(",")
            if value
        )
    if "PREPROCESS_DEPTH" in os.environ:
        kernel.PREPROCESS_MAX_DEPTH = int(os.environ["PREPROCESS_DEPTH"])
    if "FLOW_SCALAR_CONSTANT_COUNT" in os.environ:
        kernel.FLOW_SCALAR_CONSTANT_COUNT = int(
            os.environ["FLOW_SCALAR_CONSTANT_COUNT"]
        )
    if "FLOW_ONE_CONSTANT" in os.environ:
        kernel.FLOW_ONE_CONSTANT = bool(int(os.environ["FLOW_ONE_CONSTANT"]))
    if "REUSE_TOP_RELOCATION_LEVEL4" in os.environ:
        kernel.REUSE_TOP_RELOCATION_LEVEL4 = bool(
            int(os.environ["REUSE_TOP_RELOCATION_LEVEL4"])
        )
    if "INDEPENDENT_TOP_P0" in os.environ:
        kernel.INDEPENDENT_TOP_P0 = bool(int(os.environ["INDEPENDENT_TOP_P0"]))
    if "INDEPENDENT_TOP_P1" in os.environ:
        kernel.INDEPENDENT_TOP_P1 = bool(int(os.environ["INDEPENDENT_TOP_P1"]))
    if "INDEPENDENT_RELOCATION_LOAD_POINTERS" in os.environ:
        kernel.INDEPENDENT_RELOCATION_LOAD_POINTERS = bool(
            int(os.environ["INDEPENDENT_RELOCATION_LOAD_POINTERS"])
        )
    if "INDEPENDENT_INPUT_POINTERS" in os.environ:
        kernel.INDEPENDENT_INPUT_POINTERS = bool(
            int(os.environ["INDEPENDENT_INPUT_POINTERS"])
        )
    if "DERIVE_TOP_P1_FROM_P0" in os.environ:
        kernel.DERIVE_TOP_P1_FROM_P0 = bool(
            int(os.environ["DERIVE_TOP_P1_FROM_P0"])
        )
    if "DERIVE_SETUP_SECOND_POINTERS" in os.environ:
        kernel.DERIVE_SETUP_SECOND_POINTERS = bool(
            int(os.environ["DERIVE_SETUP_SECOND_POINTERS"])
        )
    if "REVERSED_RELOCATED_TREE" in os.environ:
        kernel.REVERSED_RELOCATED_TREE = bool(
            int(os.environ["REVERSED_RELOCATED_TREE"])
        )
    if "SSA_WORKSPACES" in os.environ:
        kernel.SSA_WORKSPACES = bool(int(os.environ["SSA_WORKSPACES"]))
    if "SSA_LEVEL4_WORKSPACES" in os.environ:
        kernel.SSA_LEVEL4_WORKSPACES = bool(
            int(os.environ["SSA_LEVEL4_WORKSPACES"])
        )
    if "SSA_ALL_WORKSPACES" in os.environ:
        kernel.SSA_ALL_WORKSPACES = bool(
            int(os.environ["SSA_ALL_WORKSPACES"])
        )
    if "SSA_FIRST_WORKSPACE_GROUPS" in os.environ:
        kernel.SSA_FIRST_WORKSPACE_GROUPS = frozenset(
            int(value)
            for value in os.environ["SSA_FIRST_WORKSPACE_GROUPS"].split(",")
            if value
        )
    if "TAIL_GROUP_COUNT" in os.environ:
        kernel.INDEPENDENT_TAIL_GROUP_COUNT = int(
            os.environ["TAIL_GROUP_COUNT"]
        )
    if "BRANCH_FINAL_GROUP" in os.environ:
        kernel.BRANCH_FINAL_GROUP = int(os.environ["BRANCH_FINAL_GROUP"])
    if "PAIRED_BRANCH_FINAL" in os.environ:
        kernel.PAIRED_BRANCH_FINAL = bool(int(os.environ["PAIRED_BRANCH_FINAL"]))
    if "BRANCH_FINAL_LANES" in os.environ:
        kernel.BRANCH_FINAL_LANES = tuple(
            int(value)
            for value in os.environ["BRANCH_FINAL_LANES"].split(",")
            if value
        )
    if "DELAYED_PAIR_BRANCH_GROUPS" in os.environ:
        kernel.DELAYED_PAIR_BRANCH_GROUPS = frozenset(
            int(value)
            for value in os.environ["DELAYED_PAIR_BRANCH_GROUPS"].split(",")
            if value
        )
    if "PAIRED_EARLY_XOR" in os.environ:
        kernel.PAIRED_EARLY_XOR = bool(int(os.environ["PAIRED_EARLY_XOR"]))
    if "PAIRED_FLOW_SELECT" in os.environ:
        kernel.PAIRED_FLOW_SELECT = bool(int(os.environ["PAIRED_FLOW_SELECT"]))
    if "HYBRID_MADD_PAIRS" in os.environ:
        kernel.HYBRID_MADD_PAIRS = int(os.environ["HYBRID_MADD_PAIRS"])
    if "HYBRID_MADD_OVERRIDES" in os.environ:
        overrides: dict[tuple[int, int], int] = {}
        for item in os.environ["HYBRID_MADD_OVERRIDES"].split(","):
            group, round_index, pair_limit = (int(value) for value in item.split(":"))
            overrides[(group, round_index)] = pair_limit
        kernel.HYBRID_MADD_OVERRIDES = overrides
    if "DIRECT_BRANCH_LOOKUPS" in os.environ:
        kernel.DIRECT_BRANCH_LOOKUPS = {
            int(group): tuple(range(int(count)))
            for group, count in (
                item.split(":")
                for item in os.environ["DIRECT_BRANCH_LOOKUPS"].split(",")
                if item
            )
        }
    if "PAIRED_DIRECT_BRANCH_LOOKUPS" in os.environ:
        kernel.PAIRED_DIRECT_BRANCH_LOOKUPS = {
            int(group): tuple((2 * lane, 2 * lane + 1) for lane in range(int(count)))
            for group, count in (
                item.split(":")
                for item in os.environ["PAIRED_DIRECT_BRANCH_LOOKUPS"].split(",")
                if item
            )
        }
    if "SCALAR_COUNT" in os.environ:
        scalar_count = int(os.environ["SCALAR_COUNT"])
        kernel.HASH_SCALAR_EXTRA = frozenset(
            kernel._BASE_SCALAR | set(kernel._SCALAR_CANDIDATES[:scalar_count])
        )
    if "HASH_SCALAR_EXTRA_COUNT" in os.environ:
        scalar_count = int(os.environ["HASH_SCALAR_EXTRA_COUNT"])
        kernel.HASH_SCALAR_EXTRA = frozenset(
            kernel._BASE_SCALAR | set(kernel._SCALAR_CANDIDATES[:scalar_count])
        )
    def group_set(name: str) -> frozenset[int]:
        value = os.environ.get(name, "").strip()
        return frozenset(int(x) for x in value.split(",") if x)

    if "SAVED_SECOND_PATH_EXTRA_GROUPS" in os.environ:
        kernel.SAVED_SECOND_PATH_EXTRA_GROUPS = group_set(
            "SAVED_SECOND_PATH_EXTRA_GROUPS"
        )
    if "FINAL_CACHE_SET" in os.environ:
        kernel.FINAL_CACHE_SET = group_set("FINAL_CACHE_SET")
    if "VALU_FINAL_CACHE_SET" in os.environ:
        kernel.VALU_FINAL_CACHE_SET = group_set("VALU_FINAL_CACHE_SET")
    if "VALU_FINAL_CACHE_COUNTS" in os.environ:
        kernel.VALU_FINAL_CACHE_COUNTS = {
            int(group): int(count)
            for group, count in (
                item.split(":")
                for item in os.environ["VALU_FINAL_CACHE_COUNTS"].split(",")
                if item
            )
        }
    if "SCALAR_VALU_FINAL_DIFF_SET" in os.environ:
        kernel.SCALAR_VALU_FINAL_DIFF_SET = frozenset(
            tuple(int(component) for component in item.split(":"))
            for item in os.environ["SCALAR_VALU_FINAL_DIFF_SET"].split(",")
            if item
        )
    if "EARLY_FINAL_CACHE_SET" in os.environ:
        kernel.EARLY_FINAL_CACHE_SET = group_set("EARLY_FINAL_CACHE_SET")
    if "RELOCATION_STAGE_ORDER" in os.environ:
        kernel.RELOCATION_STAGE_ORDER = os.environ["RELOCATION_STAGE_ORDER"]
    if "RELOCATION_STORE_STREAMS" in os.environ:
        kernel.RELOCATION_STORE_STREAMS = int(
            os.environ["RELOCATION_STORE_STREAMS"]
        )
    if "VECTOR_TOP_C5_BLOCKS" in os.environ:
        kernel.VECTOR_TOP_C5_BLOCKS = int(os.environ["VECTOR_TOP_C5_BLOCKS"])
    if "FLOW_OUTPUT_ADVANCE_POSITIONS" in os.environ:
        kernel.FLOW_OUTPUT_ADVANCE_POSITIONS = group_set(
            "FLOW_OUTPUT_ADVANCE_POSITIONS"
        )
    if "SCALAR_FINAL_C5" in os.environ:
        kernel.SCALAR_FINAL_C5_SET = group_set("SCALAR_FINAL_C5")
    if "SCALAR_FINAL_JOIN" in os.environ:
        kernel.SCALAR_FINAL_JOIN_SET = group_set("SCALAR_FINAL_JOIN")
    if "SCALAR_FINAL_SHIFT" in os.environ:
        kernel.SCALAR_FINAL_SHIFT_SET = group_set("SCALAR_FINAL_SHIFT")
    if "SCALAR_FINAL_HASH23_JOIN" in os.environ:
        kernel.SCALAR_FINAL_HASH23_JOIN_SET = group_set(
            "SCALAR_FINAL_HASH23_JOIN"
        )
    if "SCALAR_FINAL_HASH4" in os.environ:
        kernel.SCALAR_FINAL_HASH4_SET = group_set("SCALAR_FINAL_HASH4")
    def pair_set(name: str) -> frozenset[tuple[int, int]]:
        value = os.environ.get(name, "").strip()
        return frozenset(
            tuple(int(component) for component in item.split(":"))
            for item in value.split(",")
            if item
        )

    if "HASH_SCALAR_EXTRA_ADD" in os.environ:
        kernel.HASH_SCALAR_EXTRA = frozenset(
            set(kernel.HASH_SCALAR_EXTRA) | set(pair_set("HASH_SCALAR_EXTRA_ADD"))
        )
    if "HASH_SCALAR_EXTRA_REMOVE" in os.environ:
        kernel.HASH_SCALAR_EXTRA = frozenset(
            set(kernel.HASH_SCALAR_EXTRA)
            - set(pair_set("HASH_SCALAR_EXTRA_REMOVE"))
        )
    if "SCALAR_DEPTH3_GROUPS" in os.environ:
        kernel.SCALAR_SECOND_PATH_DEPTH3_GROUPS = group_set(
            "SCALAR_DEPTH3_GROUPS"
        )
    if "SCALAR_LEVEL4_GROUPS" in os.environ:
        kernel.SCALAR_LEVEL4_CONDITION_GROUPS = group_set(
            "SCALAR_LEVEL4_GROUPS"
        )
    if "OFFSET14" in os.environ:
        offsets = list(kernel.FULL_ROUND_OFFSETS)
        offsets[14] = int(os.environ["OFFSET14"])
        kernel.FULL_ROUND_OFFSETS = tuple(offsets)
    policy = int(os.environ.get("POLICY", str(kernel.SCHEDULE_POLICIES[0])))
    # The embedded production schedule belongs to the default graph only.
    kernel.SCHEDULE_EXACT_CYCLES = None

    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        # Some experimental resource-balancing configurations leave no hole
        # for a postpass or fail its heuristic trace packing.  The DAG and
        # list schedule already exist at that point, which is all the exact
        # scheduler needs; CP-SAT adds the required packet constraints below.
        if not hasattr(builder, "dag_ops"):
            raise
    ops = builder.dag_ops
    _, old = builder._schedule(ops, policy, return_cycles=True)
    if bool(int(os.environ.get("REAL_TAIL_STORES", "1"))):
        # The production postpass removes the rolling pointer chain after
        # group 23 and fills groups 24..31 into free store slots using eight
        # independent scalar addresses.  Model that real epilogue rather than
        # the nominal rolling-address DAG, which otherwise introduces false
        # dependencies and ALU pressure precisely where the target is tight.
        first_tail_group = kernel.N_GROUPS - kernel.INDEPENDENT_TAIL_GROUP_COUNT
        last_prefix_store = max(
            i
            for i, op in enumerate(ops)
            if op.tag == "output_store"
            and op.group is not None
            and op.group < first_tail_group
        )
        adjusted = []
        for i, op in enumerate(ops):
            if op.tag == "pointer_advance" and i > last_prefix_store:
                adjusted.append(
                    replace(op, engine="debug", parents={}, reads=(), writes=())
                )
            elif (
                op.tag == "output_store"
                and op.group is not None
                and op.group >= first_tail_group
            ):
                adjusted.append(
                    replace(
                        op,
                        parents={
                            parent: lag
                            for parent, lag in op.parents.items()
                            if ops[parent].tag not in {"output_pointer", "pointer_advance"}
                        },
                    )
                )
            else:
                adjusted.append(op)
        ops = adjusted
    if "HINT" in os.environ:
        old = json.loads(Path(os.environ["HINT"]).read_text())["cycles"]
        if len(old) != len(ops):
            raise ValueError("hint schedule does not match DAG")
    projected_fixed: set[int] = set()
    if "FIX_PROJECTED" in os.environ:
        payload = json.loads(Path(os.environ["FIX_PROJECTED"]).read_text())
        fixed_engine = payload["engine"]
        for raw_index, cycle in payload["cycles"].items():
            index = int(raw_index)
            if ops[index].engine != fixed_engine:
                raise ValueError("projected fixed cycle has wrong engine")
            old[index] = int(cycle)
            projected_fixed.add(index)
    if bool(int(os.environ.get("DROP_ZERO", "0"))):
        ops = [
            replace(op, parents={p: lag for p, lag in op.parents.items() if lag})
            for op in ops
        ]
    selected_groups = group_set("LOCAL_GROUPS")
    if selected_groups:
        local_set = {
            i
            for i, op in enumerate(ops)
            if op.group in selected_groups or old[i] >= horizon
        }
        # Zero-lag WAR chains should not be cut at the LNS boundary: doing so
        # pins one endpoint to the incumbent cycle and hides legal swaps.
        changed = True
        while changed:
            changed = False
            for child, op in enumerate(ops):
                for parent, lag in op.parents.items():
                    if not lag and ((child in local_set) != (parent in local_set)):
                        candidate = parent if child in local_set else child
                        if candidate not in local_set:
                            local_set.add(candidate)
                            changed = True
    else:
        local_set = {i for i, cycle in enumerate(old) if cycle >= cutoff}
    local_set.difference_update(projected_fixed)
    frontier = set(local_set)
    for _ in range(ancestor_steps):
        parents = {
            parent
            for child in frontier
            for parent in ops[child].parents
            if old[parent] >= cutoff - early_slack
        }
        frontier = parents - local_set
        if not frontier:
            break
        local_set.update(frontier)
    fixed_engines = frozenset(
        engine
        for engine in os.environ.get("FIX_ENGINES", "").split(",")
        if engine
    )
    if fixed_engines:
        local_set = {
            i
            for i in local_set
            if ops[i].engine not in fixed_engines or old[i] >= horizon
        }
    local = sorted(local_set)
    print(f"old={max(old)+1} cutoff={cutoff} target={horizon} local={len(local)}", flush=True)

    counts = Counter(op.engine for i, op in enumerate(ops) if i in local_set)
    print("tail counts", counts, flush=True)
    for engine, count in counts.items():
        print(engine, "lower", (count + SLOT_LIMITS[engine] - 1) // SLOT_LIMITS[engine], flush=True)

    model = cp_model.CpModel()
    domain_start = max(0, cutoff - early_slack)
    hint_radius = int(os.environ.get("HINT_RADIUS", "0"))
    dag_earliest = [0] * len(ops)
    dag_tail = [0] * len(ops)
    children: list[list[tuple[int, int]]] = [[] for _ in ops]
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            dag_earliest[child] = max(
                dag_earliest[child], dag_earliest[parent] + lag
            )
            children[parent].append((child, lag))
    for parent in reversed(range(len(ops))):
        dag_tail[parent] = max(
            (lag + dag_tail[child] for child, lag in children[parent]),
            default=0,
        )
    domains = {
        i: (
            max(domain_start, dag_earliest[i], old[i] - hint_radius)
            if hint_radius
            else max(domain_start, dag_earliest[i]),
            min(horizon - 1 - dag_tail[i], old[i] + hint_radius)
            if hint_radius
            else horizon - 1 - dag_tail[i],
        )
        for i in local
    }
    starts = {
        i: model.new_int_var(lower, upper, f"s{i}")
        for i, (lower, upper) in domains.items()
    }
    time_indexed = bool(int(os.environ.get("TIME_INDEXED", "0")))
    if time_indexed:
        time_indexed_engines = frozenset(
            engine for engine in SLOT_LIMITS if engine != "debug"
        )
    else:
        time_indexed_engines = frozenset(
            engine
            for engine in os.environ.get("TIME_INDEXED_ENGINES", "").split(",")
            if engine
        )
    assignments: dict[tuple[int, int], cp_model.IntVar] = {}
    if time_indexed_engines:
        for i, (lower, upper) in domains.items():
            if ops[i].engine not in time_indexed_engines:
                continue
            choices = []
            for cycle in range(lower, upper + 1):
                choice = model.new_bool_var(f"x{i}_{cycle}")
                assignments[i, cycle] = choice
                choices.append(choice)
                model.add_hint(
                    choice,
                    int(cycle == min(upper, max(lower, old[i]))),
                )
            model.add_exactly_one(choices)
            model.add(
                starts[i]
                == sum(
                    cycle * assignments[i, cycle]
                    for cycle in range(lower, upper + 1)
                )
            )
    for child, op in enumerate(ops):
        child_local = child in local_set
        for parent, lag in op.parents.items():
            parent_local = parent in local_set
            if child_local and parent_local:
                model.add(starts[child] >= starts[parent] + lag)
            elif child_local:
                model.add(starts[child] >= old[parent] + lag)
            elif parent_local:
                model.add(old[child] >= starts[parent] + lag)

    ordered_engines = frozenset(
        engine
        for engine in os.environ.get("ORDER_ENGINES", "").split(",")
        if engine
    )
    for engine in ordered_engines:
        sequence = sorted(
            (i for i, op in enumerate(ops) if op.engine == engine),
            key=lambda i: (old[i], i),
        )
        for previous, current in zip(sequence, sequence[1:]):
            previous_local = previous in local_set
            current_local = current in local_set
            if previous_local and current_local:
                model.add(starts[current] >= starts[previous] + 1)
            elif previous_local:
                model.add(old[current] >= starts[previous] + 1)
            elif current_local:
                model.add(starts[current] >= old[previous] + 1)
            elif old[current] < old[previous] + 1:
                raise ValueError(f"fixed {engine} order is not serial")

    if "ORDER_PROJECTED" in os.environ:
        payload = json.loads(Path(os.environ["ORDER_PROJECTED"]).read_text())
        engine = payload["engine"]
        capacity = SLOT_LIMITS[engine]
        projected_cycles = {
            int(index): int(cycle)
            for index, cycle in payload["cycles"].items()
        }
        sequence = sorted(
            projected_cycles,
            key=lambda index: (projected_cycles[index], index),
        )
        for previous, current in zip(sequence, sequence[capacity:]):
            previous_local = previous in local_set
            current_local = current in local_set
            if previous_local and current_local:
                model.add(starts[current] >= starts[previous] + 1)
            elif previous_local:
                model.add(old[current] >= starts[previous] + 1)
            elif current_local:
                model.add(starts[current] >= old[previous] + 1)
            elif old[current] < old[previous] + 1:
                raise ValueError("fixed projected order is invalid")

    if bool(int(os.environ.get("PACK_BRANCH_TRACES", "1"))):
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

        def add_exact_distance(left: int, right: int, distance: int) -> None:
            left_local = left in local_set
            right_local = right in local_set
            if left_local and right_local:
                model.add(starts[left] == starts[right] + distance)
            elif left_local:
                model.add(starts[left] == old[right] + distance)
            elif right_local:
                model.add(old[left] == starts[right] + distance)
            elif old[left] != old[right] + distance:
                raise ValueError("fixed branch trace is not packetized")

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
                    add_exact_distance(child, jump_parent, 1)
            elif op.tag in target_tags:
                for parent in op.parents:
                    if ops[parent].tag in copy_tags:
                        add_exact_distance(child, parent, 0)

    fixed_use: dict[str, Counter[int]] = defaultdict(Counter)
    for i, op in enumerate(ops):
        if i not in local_set and domain_start <= old[i] < horizon:
            fixed_use[op.engine][old[i]] += 1

    ignored_engines = frozenset(
        engine
        for engine in os.environ.get("IGNORE_ENGINES", "").split(",")
        if engine
    )
    for engine, capacity in SLOT_LIMITS.items():
        if engine == "debug":
            continue
        if engine in ignored_engines:
            continue
        if engine in time_indexed_engines:
            for cycle in range(domain_start, horizon):
                choices = [
                    assignments[i, cycle]
                    for i in local
                    if ops[i].engine == engine
                    and domains[i][0] <= cycle <= domains[i][1]
                ]
                if choices:
                    model.add(
                        sum(choices)
                        <= capacity - fixed_use[engine][cycle]
                    )
                elif fixed_use[engine][cycle] > capacity:
                    model.add_bool_or([])
        else:
            intervals = []
            demands = []
            for i in local:
                if ops[i].engine == engine:
                    intervals.append(
                        model.new_fixed_size_interval_var(starts[i], 1, f"i{i}")
                    )
                    demands.append(1)
            for cycle, demand in fixed_use[engine].items():
                intervals.append(
                    model.new_fixed_size_interval_var(cycle, 1, f"f_{engine}_{cycle}")
                )
                demands.append(demand)
            model.add_cumulative(intervals, demands, capacity)

    hint_cycles = old
    old_span = max(hint_cycles) - cutoff or 1
    new_span = horizon - 1 - cutoff
    for i in local:
        hint = max(
            domain_start,
            cutoff + round((hint_cycles[i] - cutoff) * new_span / old_span),
        )
        model.add_hint(starts[i], hint)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = limit
    solver.parameters.num_workers = int(os.environ.get("WORKERS", "8"))
    solver.parameters.repair_hint = bool(int(os.environ.get("REPAIR_HINT", "0")))
    solver.parameters.hint_conflict_limit = 200_000
    solver.parameters.random_seed = int(os.environ.get("RANDOM_SEED", "1"))
    solver.parameters.randomize_search = bool(int(os.environ.get("RANDOMIZE", "0")))
    solver.parameters.use_lns_only = bool(int(os.environ.get("LNS_ONLY", "0")))
    solver.parameters.use_rins_lns = bool(int(os.environ.get("RINS_LNS", "0")))
    solver.parameters.use_lb_relax_lns = bool(int(os.environ.get("LB_RELAX_LNS", "0")))
    solver.parameters.diversify_lns_params = bool(
        int(os.environ.get("DIVERSIFY_LNS", "0"))
    )
    solver.parameters.cp_model_presolve = True
    solver.parameters.log_search_progress = bool(int(os.environ.get("LOG", "0")))
    makespan = None
    if bool(int(os.environ.get("OPTIMIZE", "0"))):
        makespan = model.new_int_var(domain_start, horizon - 1, "makespan")
        model.add_max_equality(makespan, [starts[i] for i in local])
        model.minimize(makespan)
    status = solver.solve(model)
    print("status", solver.status_name(status), flush=True)
    if makespan is not None:
        print(
            "objective",
            solver.objective_value,
            "best_bound",
            solver.best_objective_bound,
            flush=True,
        )
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        result = old.copy()
        for i in local:
            result[i] = solver.value(starts[i])
        output = Path(os.environ.get("OUT", "/tmp/aopt-tail-schedule.json"))
        output.write_text(json.dumps({"makespan": max(result) + 1, "cycles": result}))
        print(
            "makespan",
            (solver.value(makespan) + 1 if makespan is not None else max(result) + 1),
            "output",
            output,
            flush=True,
        )


if __name__ == "__main__":
    main()
