"""Joint instruction-selection and scheduling model for the final hash wave.

The production compiler exposes several semantically equivalent scalar/VALU
lowerings.  Searching those switches independently misses combinations where
one lowering creates the exact hole needed by another group.  This model keeps
the proven prefix fixed and lets CP-SAT choose all final-wave lowerings at once.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path

from ortools.sat.python import cp_model

import codex.perf_takehome_under1000 as kernel
from problem import SLOT_LIMITS


@dataclass
class Task:
    name: str
    engine: str
    start: cp_model.IntVar
    present: cp_model.IntVar
    interval: cp_model.IntervalVar | None
    choices: tuple[tuple[int, cp_model.IntVar], ...]


def main() -> None:
    if "PAIRED_FLOW_SELECT" in os.environ:
        kernel.PAIRED_FLOW_SELECT = bool(int(os.environ["PAIRED_FLOW_SELECT"]))
    source = Path(os.environ.get("SOURCE", "/tmp/tail-valu5895-opt991.json"))
    old = json.loads(source.read_text())["cycles"]
    target = int(os.environ.get("TARGET", "990"))
    first_group = int(os.environ.get("FIRST_GROUP", "16"))
    groups = tuple(range(first_group, kernel.N_GROUPS))
    group_set = frozenset(groups)

    builder = kernel.KernelBuilder()
    builder.build_kernel(10, 2047, 256, 16)
    ops = builder.dag_ops
    if len(old) != len(ops):
        raise ValueError("source schedule does not match production DAG")

    removed = {
        i
        for i, op in enumerate(ops)
        if (
            op.group in group_set
            and op.round == 15
            and op.tag.startswith("hash_")
        )
        or (op.group in group_set and op.tag == "output_store")
    }
    fixed_use: dict[str, Counter[int]] = defaultdict(Counter)
    for i, op in enumerate(ops):
        if i in removed or op.engine == "debug":
            continue
        if old[i] >= target:
            raise ValueError(
                f"fixed operation exceeds target: {i=} {op.tag=} {old[i]=}"
            )
        fixed_use[op.engine][old[i]] += 1

    releases = {}
    for group in groups:
        hash0 = next(
            i
            for i, op in enumerate(ops)
            if op.group == group and op.round == 15 and op.tag == "hash_0"
        )
        releases[group] = max(
            old[parent] + lag
            for parent, lag in ops[hash0].parents.items()
            if parent not in removed
        )

    model = cp_model.CpModel()
    tasks: list[Task] = []
    mode_vars: dict[tuple[int, str], cp_model.IntVar] = {}
    time_indexed = bool(int(os.environ.get("TIME_INDEXED", "1")))
    domain_start = min(releases.values())

    def task(name: str, engine: str, present: cp_model.IntVar) -> Task:
        start = model.new_int_var(0, target - 1, f"start_{name}")
        if time_indexed:
            choices = tuple(
                (cycle, model.new_bool_var(f"at_{name}_{cycle}"))
                for cycle in range(domain_start, target)
            )
            model.add(sum(choice for _, choice in choices) == present)
            model.add(
                start == sum(cycle * choice for cycle, choice in choices)
            )
            interval = None
        else:
            choices = ()
            interval = model.new_optional_fixed_size_interval_var(
                start, 1, present, f"interval_{name}"
            )
            # Optional intervals otherwise leave their start integer
            # unconstrained when absent, creating meaningless symmetry.
            model.add(start == 0).only_enforce_if(present.Not())
        result = Task(name, engine, start, present, interval, choices)
        tasks.append(result)
        return result

    def always(name: str, engine: str) -> list[Task]:
        present = model.new_bool_var(f"present_{name}")
        model.add(present == 1)
        return [task(name, engine, present)]

    def selectable(group: int, name: str) -> list[Task]:
        scalar = model.new_bool_var(f"scalar_g{group}_{name}")
        vector = model.new_bool_var(f"vector_g{group}_{name}")
        model.add(scalar + vector == 1)
        mode_vars[group, name] = scalar
        result = [task(f"g{group}_{name}_v", "valu", vector)]
        scalar_tasks = [
            task(f"g{group}_{name}_s{lane}", "alu", scalar)
            for lane in range(kernel.VLEN)
        ]
        for left, right in zip(scalar_tasks, scalar_tasks[1:]):
            model.add(left.start <= right.start).only_enforce_if(scalar)
        result.extend(scalar_tasks)
        return result

    def after(children: list[Task], parents: list[Task]) -> None:
        for child in children:
            for parent in parents:
                model.add(child.start >= parent.start + 1).only_enforce_if(
                    [child.present, parent.present]
                )

    stages_by_group: dict[int, dict[str, list[Task]]] = {}
    early_release_vars: dict[int, cp_model.IntVar] = {}
    early_release_max = int(os.environ.get("EARLY_RELEASE_MAX", "0"))
    for group in groups:
        stages: dict[str, list[Task]] = {}
        stages["h0"] = always(f"g{group}_h0", "valu")
        stages["h1shift"] = always(f"g{group}_h1shift", "valu")
        stages["h1const"] = selectable(group, "h1const")
        stages["h1join"] = selectable(group, "h1join")
        stages["h23add"] = always(f"g{group}_h23add", "valu")
        stages["h23shift"] = always(f"g{group}_h23shift", "valu")
        stages["h23join"] = selectable(group, "h23join")

        # Hash stage 4 is either one vector MADD or two scalar eight-lane
        # waves.  Both scalar waves share the same selection bit.
        h4_scalar = model.new_bool_var(f"scalar_g{group}_h4")
        h4_vector = model.new_bool_var(f"vector_g{group}_h4")
        model.add(h4_scalar + h4_vector == 1)
        mode_vars[group, "h4"] = h4_scalar
        stages["h4v"] = [task(f"g{group}_h4_v", "valu", h4_vector)]
        stages["h4mul"] = [
            task(f"g{group}_h4_mul_s{lane}", "alu", h4_scalar)
            for lane in range(kernel.VLEN)
        ]
        stages["h4add"] = [
            task(f"g{group}_h4_add_s{lane}", "alu", h4_scalar)
            for lane in range(kernel.VLEN)
        ]
        for scalar_stage in (stages["h4mul"], stages["h4add"]):
            for left, right in zip(scalar_stage, scalar_stage[1:]):
                model.add(left.start <= right.start).only_enforce_if(h4_scalar)

        stages["h5shift"] = selectable(group, "h5shift")
        stages["h5const"] = selectable(group, "h5const")
        stages["h5join"] = selectable(group, "h5join")
        # A vector join naturally feeds one vstore.  A scalar join can instead
        # stream lanes independently: each completed lane gets its own scalar
        # address constant and store, avoiding a barrier on the slowest lane.
        vector_join = stages["h5join"][0]
        scalar_joins = stages["h5join"][1:]
        vector_store = task(
            f"g{group}_store_v", "store", vector_join.present
        )
        pointer_loads = [
            task(f"g{group}_store_ptr_s{lane}", "load", join.present)
            for lane, join in enumerate(scalar_joins)
        ]
        scalar_stores = [
            task(f"g{group}_store_s{lane}", "store", join.present)
            for lane, join in enumerate(scalar_joins)
        ]
        stages["store"] = [vector_store] + scalar_stores
        stages_by_group[group] = stages

        if early_release_max:
            early_release = model.new_int_var(
                0, early_release_max, f"early_release_g{group}"
            )
            early_release_vars[group] = early_release
            for item in stages["h0"]:
                model.add(item.start >= releases[group] - early_release)
        else:
            for item in stages["h0"]:
                model.add(item.start >= releases[group])
        after(stages["h1shift"], stages["h0"])
        after(stages["h1const"], stages["h0"])
        after(stages["h1join"], stages["h1shift"] + stages["h1const"])
        after(stages["h23add"], stages["h1join"])
        after(stages["h23shift"], stages["h1join"])
        after(stages["h23join"], stages["h23add"] + stages["h23shift"])
        after(stages["h4v"], stages["h23join"])
        after(stages["h4mul"], stages["h23join"])
        after(stages["h4add"], stages["h4mul"])
        h4_outputs = stages["h4v"] + stages["h4add"]
        after(stages["h5shift"], h4_outputs)
        after(stages["h5const"], h4_outputs)
        after(stages["h5join"], stages["h5shift"] + stages["h5const"])
        after([vector_store], [vector_join])
        for lane_store, lane_join, pointer_load in zip(
            scalar_stores, scalar_joins, pointer_loads
        ):
            after([lane_store], [lane_join, pointer_load])

    allow_slack = bool(int(os.environ.get("ALLOW_SLACK", "0")))
    slack_vars: dict[tuple[str, int], cp_model.IntVar] = {}
    for engine, capacity in SLOT_LIMITS.items():
        if engine == "debug":
            continue
        if time_indexed:
            for cycle in range(domain_start, target):
                choices = [
                    choice
                    for item in tasks
                    if item.engine == engine
                    for choice_cycle, choice in item.choices
                    if choice_cycle == cycle
                ]
                if allow_slack:
                    slack = model.new_int_var(0, capacity, f"slack_{engine}_{cycle}")
                    slack_vars[engine, cycle] = slack
                    model.add(
                        sum(choices)
                        <= capacity - fixed_use[engine][cycle] + slack
                    )
                else:
                    model.add(
                        sum(choices) <= capacity - fixed_use[engine][cycle]
                    )
        else:
            intervals = [
                item.interval for item in tasks
                if item.engine == engine and item.interval is not None
            ]
            demands = [1] * len(intervals)
            for cycle, demand in fixed_use[engine].items():
                fixed = model.new_interval_var(
                    cycle, 1, cycle + 1, f"fixed_{engine}_{cycle}"
                )
                intervals.append(fixed)
                demands.append(demand)
            model.add_cumulative(intervals, demands, capacity)

    incumbent_modes = {
        "h1const": {
            group
            for group in groups
            if (group + 15) % kernel.HASH_SCALAR_MOD == 0
            or (group, 15) in kernel.HASH_SCALAR_EXTRA
        },
        "h1join": {
            group for group in groups if (group, 15) in kernel.SCALAR_HASH1_JOIN_SET
        },
        "h23join": set(kernel.SCALAR_FINAL_HASH23_JOIN_SET) & group_set,
        "h4": set(kernel.SCALAR_FINAL_HASH4_SET) & group_set,
        "h5shift": set(kernel.SCALAR_FINAL_SHIFT_SET) & group_set,
        "h5const": set(kernel.SCALAR_FINAL_C5_SET) & group_set,
        "h5join": set(kernel.SCALAR_FINAL_JOIN_SET) & group_set,
    }
    for (group, name), variable in mode_vars.items():
        model.add_hint(variable, int(group in incumbent_modes[name]))
    fixed_vector_modes = frozenset(
        name for name in os.environ.get("FIX_VECTOR", "").split(",") if name
    )
    fixed_scalar_modes = frozenset(
        name for name in os.environ.get("FIX_SCALAR", "").split(",") if name
    )
    for (group, name), variable in mode_vars.items():
        if name in fixed_vector_modes:
            model.add(variable == 0)
        elif name in fixed_scalar_modes:
            model.add(variable == 1)

    tag_matchers = {
        "h0": lambda tag: tag == "hash_0",
        "h1shift": lambda tag: tag.startswith("hash_1_shift"),
        "h1const": lambda tag: tag == "hash_1_const",
        "h1join": lambda tag: tag.startswith("hash_1_join"),
        "h23add": lambda tag: tag == "hash_23_add",
        "h23shift": lambda tag: tag == "hash_23_shift",
        "h23join": lambda tag: tag.startswith("hash_23_join"),
        "h4v": lambda tag: tag == "hash_4",
        "h4mul": lambda tag: tag == "hash_4_scalar_mul",
        "h4add": lambda tag: tag == "hash_4_scalar_add",
        "h5shift": lambda tag: tag == "hash_5_shift",
        "h5const": lambda tag: tag.startswith("hash_5_const"),
        "h5join": lambda tag: tag.startswith("hash_5_join"),
        "store": lambda tag: tag == "output_store",
    }
    for group, stages in stages_by_group.items():
        for stage_name, stage_tasks in stages.items():
            actual = sorted(
                (old[i], op.engine)
                for i, op in enumerate(ops)
                if op.group == group
                and (op.round == 15 or stage_name == "store")
                and tag_matchers[stage_name](op.tag)
            )
            if not actual:
                continue
            engine = actual[0][1]
            incumbent_tasks = [item for item in stage_tasks if item.engine == engine]
            if len(incumbent_tasks) != len(actual):
                continue
            for item, (cycle, _) in zip(incumbent_tasks, actual):
                model.add_hint(item.start, min(target - 1, cycle))

    # Optional secondary objective.  Feasibility mode is substantially faster
    # for discovering whether a 990-cycle lowering exists at all.
    slack_budget = os.environ.get("SLACK_BUDGET")
    early_release_budget = os.environ.get("EARLY_RELEASE_BUDGET")
    if early_release_budget is not None:
        model.add(sum(early_release_vars.values()) <= int(early_release_budget))
    if allow_slack and slack_budget is not None:
        model.add(sum(slack_vars.values()) <= int(slack_budget))
    if allow_slack and slack_budget is None:
        model.minimize(
            10_000 * sum(slack_vars.values()) + sum(mode_vars.values())
        )
    elif bool(int(os.environ.get("OPTIMIZE", "0"))):
        model.minimize(sum(mode_vars.values()))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(os.environ.get("TIME_LIMIT", "300"))
    solver.parameters.num_workers = int(os.environ.get("WORKERS", "8"))
    solver.parameters.log_search_progress = bool(int(os.environ.get("LOG", "0")))
    status = solver.solve(model)
    print("status", solver.status_name(status), flush=True)
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return
    if early_release_vars:
        print(
            "early_release",
            [
                (group, solver.value(variable))
                for group, variable in early_release_vars.items()
                if solver.value(variable)
            ],
            flush=True,
        )
    if allow_slack:
        used_slack = [
            (engine, cycle, solver.value(variable))
            for (engine, cycle), variable in slack_vars.items()
            if solver.value(variable)
        ]
        print("slack", used_slack, flush=True)
    for name in (
        "h1const",
        "h1join",
        "h23join",
        "h4",
        "h5shift",
        "h5const",
        "h5join",
    ):
        selected = [
            group for group in groups if solver.value(mode_vars[group, name])
        ]
        print(name, ",".join(map(str, selected)), flush=True)
    for group in groups:
        completion = max(
            solver.value(item.start)
            for item in stages_by_group[group]["store"]
            if solver.value(item.present)
        )
        print(f"group={group} store={completion}", flush=True)


if __name__ == "__main__":
    main()
