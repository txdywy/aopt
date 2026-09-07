"""Exact whole-kernel CP-SAT feasibility model for the VLIW scheduler.

Unlike :mod:`solve_projected_engine` and :mod:`solve_projected_engines`, this
tool never contracts, removes, or transitively reduces a DAG edge.  It is
intended as the correctness oracle for local-search candidates: one start
variable per emitted operation, every original precedence edge, and a capacity
constraint for every physical execution engine are present in the model.

The complete model is deliberately expensive (roughly 20k unit jobs at a
959-cycle horizon).  Per-engine schedules therefore act only as CP-SAT hints;
they never become assumptions unless ``FIX_HINT_CYCLES`` is explicitly set.
``HINT_RADIUS`` turns the same model into a bounded exact LNS neighborhood.

Typical 959-cycle smoke run for the current balanced candidate::

    source /private/tmp/aopt-vrestore1-p6input-m4m23-959.env
    HINTS=/private/tmp/aopt-vrestore1-p6input-keep_m4_m23-alu959.json,\
    /private/tmp/aopt-vrestore1-p6input-keep_m4_m23-valu959.json,\
    /private/tmp/aopt-vrestore1-p6input-keep_m4_m23-load-bubble56-959.json,\
    /private/tmp/aopt-vrestore1-p6input-keep_m4_m23-flow959.json,\
    /private/tmp/aopt-vrestore1-p6input-m4m23-store959.json \
    TARGET=959 HINT_RADIUS=64 TIME_LIMIT=30 WORKERS=4 \
    python3 -m codex.solve_full_dag

Set ``MICROSLOT_ENGINES=alu,valu`` to replace the two corresponding Cumulative
constraints by an exact lane/microslot ``AllDifferent`` representation.  For a
unit-capacity engine, ``ALL_DIFFERENT_ENGINES=flow`` uses ``AllDifferent`` on
the start variables directly.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import heapq
import json
import os
from pathlib import Path
from typing import Iterable

from ortools.sat.python import cp_model

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from problem import SLOT_LIMITS
from codex.partial_order_hints import load_partial_order_hints


def csv_set(name: str) -> frozenset[str]:
    """Read a comma-separated environment variable as a set."""
    return frozenset(value for value in os.environ.get(name, "").split(",") if value)


def group_round_set(name: str) -> frozenset[tuple[int, int]]:
    """Parse ``group:round`` diagnostic selectors."""
    result: set[tuple[int, int]] = set()
    for item in os.environ.get(name, "").split(","):
        if not item:
            continue
        try:
            group, rnd = (int(value) for value in item.split(":"))
        except ValueError as error:
            raise ValueError(f"{name} item must be group:round: {item!r}") from error
        if not 0 <= group < kernel.N_GROUPS or rnd < -1:
            raise ValueError(f"{name} contains invalid group:round {item!r}")
        result.add((group, rnd))
    return frozenset(result)


def integer_set(name: str) -> frozenset[int]:
    """Parse a comma-separated integer diagnostic selector."""
    try:
        return frozenset(
            int(value) for value in os.environ.get(name, "").split(",") if value
        )
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated integer set") from error


def tag_group_round_set(name: str) -> frozenset[tuple[str, int, int]]:
    """Parse ``tag@group:round`` capacity-exemption selectors."""
    result: set[tuple[str, int, int]] = set()
    for item in os.environ.get(name, "").split(","):
        if not item:
            continue
        try:
            tag, group_round = item.split("@", 1)
            group, rnd = (int(value) for value in group_round.split(":"))
        except ValueError as error:
            raise ValueError(
                f"{name} item must be tag@group:round: {item!r}"
            ) from error
        if not tag or not 0 <= group < kernel.N_GROUPS or rnd < -1:
            raise ValueError(f"{name} contains invalid selector {item!r}")
        result.add((tag, group, rnd))
    return frozenset(result)


def apply_diagnostic_lowering_experiments() -> dict[str, tuple[int, ...]]:
    """Apply model-only compiler-lowering experiments.

    These switches intentionally live under the ``EXPERIMENT_`` namespace and
    are consumed only by this strict oracle.  That keeps a speculative
    resource-balance experiment out of the submission/configuration path while
    allowing the exact V/Flow model to screen a semantic rewrite before it is
    promoted into the production compiler.
    """
    selected: dict[str, tuple[int, ...]] = {}
    for environment, attribute in (
        (
            "EXPERIMENT_SCALAR_SECOND_PATH_GROUPS_ADD",
            "SCALAR_SECOND_PATH_GROUPS",
        ),
        (
            "EXPERIMENT_SCALAR_SECOND_PATH_DEPTH2_GROUPS_ADD",
            "SCALAR_SECOND_PATH_DEPTH2_GROUPS",
        ),
        (
            "EXPERIMENT_SCALAR_SECOND_PATH_DEPTH3_GROUPS_ADD",
            "SCALAR_SECOND_PATH_DEPTH3_GROUPS",
        ),
        (
            "EXPERIMENT_SCALAR_LEVEL4_CONDITION_GROUPS_ADD",
            "SCALAR_LEVEL4_CONDITION_GROUPS",
        ),
    ):
        raw = os.environ.get(environment)
        if raw is None:
            continue
        groups = tuple(sorted({int(value) for value in raw.split(",") if value}))
        if any(group < 0 or group >= kernel.N_GROUPS for group in groups):
            raise ValueError(
                f"{environment} contains a group outside 0..{kernel.N_GROUPS - 1}"
            )
        current = frozenset(getattr(kernel, attribute))
        setattr(kernel, attribute, frozenset(current | set(groups)))
        selected[attribute] = groups
    raw_pairs = os.environ.get("EXPERIMENT_HYBRID_MADD_PAIRS")
    if raw_pairs is not None:
        pairs = int(raw_pairs)
        if not 0 <= pairs <= 8:
            raise ValueError("EXPERIMENT_HYBRID_MADD_PAIRS must be in 0..8")
        kernel.HYBRID_MADD_PAIRS = pairs
        selected["HYBRID_MADD_PAIRS"] = (pairs,)
    return selected


def topological_data(ops: list[kernel._Op]) -> tuple[
    list[int], list[list[tuple[int, int]]], list[int], list[int]
]:
    """Return topological order, children, exact DAG release and tail bounds."""
    children: list[list[tuple[int, int]]] = [[] for _ in ops]
    indegree = [0] * len(ops)
    for child, op in enumerate(ops):
        indegree[child] = len(op.parents)
        for parent, lag in op.parents.items():
            children[parent].append((child, lag))

    ready = [index for index, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    topological: list[int] = []
    while ready:
        parent = heapq.heappop(ready)
        topological.append(parent)
        for child, _ in children[parent]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(topological) != len(ops):
        raise ValueError("kernel DAG is cyclic")

    earliest = [0] * len(ops)
    for child in topological:
        for parent, lag in ops[child].parents.items():
            earliest[child] = max(earliest[child], earliest[parent] + lag)
    tail = [0] * len(ops)
    for parent in reversed(topological):
        tail[parent] = max(
            (lag + tail[child] for child, lag in children[parent]), default=0
        )
    return topological, children, earliest, tail


def weighted_transitive_reduction(
    ops: list[kernel._Op],
) -> tuple[list[kernel._Op], int, int]:
    """Remove only edges implied by another weighted precedence path.

    The compiler emits many lane-wise edges whose timing is already implied
    by another parent and a retained path.  This reduction is weighted: an
    edge ``p -> c`` with lag ``d`` is removed only when the retained graph
    proves a path from ``p`` to ``c`` with distance at least ``d``.  Thus the
    earliest/latest bounds are unchanged.  Callers must retain the original
    operations and validate a returned witness against them.
    """
    children: list[list[tuple[int, int]]] = [[] for _ in ops]
    indegree = [len(op.parents) for op in ops]
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            children[parent].append((child, lag))
    ready = [index for index, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    topo: list[int] = []
    while ready:
        parent = heapq.heappop(ready)
        topo.append(parent)
        for child, _ in children[parent]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(topo) != len(ops):
        raise ValueError("kernel DAG is cyclic")
    position = {index: rank for rank, index in enumerate(topo)}

    reduced: list[dict[int, int]] = [{} for _ in ops]
    ancestor_distance: list[dict[int, int]] = [{} for _ in ops]
    raw_edges = sum(len(op.parents) for op in ops)
    kept_edges = 0
    for child in topo:
        implied: dict[int, int] = {}
        parents = sorted(
            ops[child].parents.items(),
            key=lambda item: position[item[0]],
            reverse=True,
        )
        for parent, lag in parents:
            if implied.get(parent, -1) >= lag:
                continue
            reduced[child][parent] = lag
            kept_edges += 1
            implied[parent] = max(implied.get(parent, -1), lag)
            for ancestor, distance in ancestor_distance[parent].items():
                implied[ancestor] = max(implied.get(ancestor, -1), distance + lag)
        ancestor_distance[child] = implied

    reduced_ops = [
        replace(op, parents=reduced[index])
        for index, op in enumerate(ops)
    ]
    return reduced_ops, raw_edges, kept_edges


def schedule_from_payload(
    path: Path, ops: list[kernel._Op]
) -> tuple[str | None, dict[int, int]]:
    """Load either a per-engine hint or a whole-schedule ``cycles`` payload."""
    payload = json.loads(path.read_text())
    raw_cycles = payload.get("cycles")
    if isinstance(raw_cycles, list):
        if len(raw_cycles) != len(ops):
            raise ValueError(f"{path}: full hint has {len(raw_cycles)} operations")
        cycles = {index: int(cycle) for index, cycle in enumerate(raw_cycles)}
    elif isinstance(raw_cycles, dict):
        cycles = {int(index): int(cycle) for index, cycle in raw_cycles.items()}
    else:
        raise ValueError(f"{path}: missing cycles")

    engine = payload.get("engine")
    if engine is not None and engine not in SLOT_LIMITS:
        raise ValueError(f"{path}: unknown engine {engine!r}")
    for index in cycles:
        if not 0 <= index < len(ops):
            raise ValueError(f"{path}: invalid operation index {index}")
        if engine is not None and ops[index].engine != engine:
            raise ValueError(
                f"{path}: op {index} is {ops[index].engine}, not {engine}"
            )
    if engine is not None:
        expected = {i for i, op in enumerate(ops) if op.engine == engine}
        if set(cycles) != expected:
            missing = len(expected - set(cycles))
            extra = len(set(cycles) - expected)
            raise ValueError(
                f"{path}: incomplete {engine} hint missing={missing} extra={extra}"
            )
    return engine, cycles


def load_hints(
    raw_paths: Iterable[str], ops: list[kernel._Op]
) -> tuple[dict[int, int], dict[str, int]]:
    """Load hints, rejecting contradictory duplicate issue times."""
    result: dict[int, int] = {}
    by_engine: Counter[str] = Counter()
    for raw_path in raw_paths:
        if not raw_path:
            continue
        path = Path(raw_path)
        engine, cycles = schedule_from_payload(path, ops)
        for index, cycle in cycles.items():
            old = result.get(index)
            if old is not None and old != cycle:
                raise ValueError(
                    f"conflicting hints for op {index}: {old} vs {cycle} ({path})"
                )
            result[index] = cycle
            by_engine[ops[index].engine] += 1
        print(
            f"hint path={path} engine={engine or 'full'} jobs={len(cycles)}",
            flush=True,
        )
    return result, dict(by_engine)


def baseline_cycles(builder: kernel.KernelBuilder, ops: list[kernel._Op]) -> list[int]:
    """Get a legal complete list-schedule hint without changing any constraints."""
    _, cycles = builder._schedule(
        builder.dag_ops,
        int(os.environ.get("BASELINE_POLICY", str(kernel.SCHEDULE_POLICIES[0]))),
        return_cycles=True,
    )
    if len(cycles) != len(ops):
        raise AssertionError("baseline schedule does not match effective DAG")
    return cycles


def validate_solution(
    ops: list[kernel._Op],
    cycles: list[int],
    horizon: int,
    *,
    capacity_engines: frozenset[str] | None = None,
    capacity_exempt_indices: frozenset[int] = frozenset(),
) -> None:
    """Validate every original edge and the selected engine capacities.

    The default is deliberately the full physical machine.  A selected subset
    is only for resource-isolation diagnostics, where checking an omitted
    capacity would turn a useful satisfiable witness into a false assertion.
    """
    if capacity_engines is None:
        capacity_engines = frozenset(SLOT_LIMITS)
    if len(cycles) != len(ops):
        raise AssertionError("solution operation count mismatch")
    if min(cycles, default=0) < 0 or max(cycles, default=-1) >= horizon:
        raise AssertionError("solution lies outside requested horizon")
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            if cycles[child] < cycles[parent] + lag:
                raise AssertionError(
                    f"precedence {parent}->{child} lag={lag} "
                    f"cycles={cycles[parent]}->{cycles[child]}"
                )
    usage = Counter(
        (cycle, op.engine)
        for index, (cycle, op) in enumerate(zip(cycles, ops))
        if index not in capacity_exempt_indices
    )
    for (cycle, engine), demand in usage.items():
        if engine in capacity_engines and demand > SLOT_LIMITS[engine]:
            raise AssertionError(
                f"capacity {engine}@{cycle}: {demand}>{SLOT_LIMITS[engine]}"
            )


def main() -> None:
    configure_target()
    # Diagnostic-only capacity perturbation.  The production simulator and
    # kernel keep the frozen machine limits; this opt-in knob lets the
    # offline compiler ask which physical resource is responsible for an
    # UNSAT horizon without editing problem.py.  It is intentionally applied
    # in this process after importing the target module, so normal runs are
    # byte-for-byte unchanged.
    raw_capacity_overrides = os.environ.get("CAPACITY_OVERRIDES", "")
    if raw_capacity_overrides:
        for item in raw_capacity_overrides.split(","):
            if not item:
                continue
            try:
                engine, raw_value = item.split(":", 1)
                value = int(raw_value)
            except ValueError as error:
                raise ValueError(
                    "CAPACITY_OVERRIDES must be engine:integer pairs"
                ) from error
            if engine not in SLOT_LIMITS or value <= 0:
                raise ValueError(f"invalid capacity override {item!r}")
            SLOT_LIMITS[engine] = value
        print(f"capacity_overrides={raw_capacity_overrides}", flush=True)
    experimental_lowerings = apply_diagnostic_lowering_experiments()
    # The default module includes an embedded schedule for a different graph.
    # A strict model must expose the target graph without attempting to replay
    # that incumbent during construction.
    kernel.SCHEDULE_EXACT_CYCLES = None
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        if not hasattr(builder, "dag_ops"):
            raise
    # Keep a byte-for-byte copy for the final strict witness audit.  The
    # optional solver-side reduction only changes the internal constraint
    # representation; it never changes the emitted program or validation.
    raw_ops = real_tail_ops(builder.dag_ops)
    ops = raw_ops
    reduction_report = ""
    if bool(int(os.environ.get("TRANSITIVE_REDUCE", "0"))):
        ops, raw_edge_count, reduced_edge_count = weighted_transitive_reduction(
            raw_ops
        )
        reduction_report = (
            f" reduction=weighted:{raw_edge_count}->{reduced_edge_count}"
        )
    _, children, earliest, tail = topological_data(ops)

    horizon = int(os.environ.get("TARGET", os.environ.get("HORIZON", "959")))
    if horizon <= 0:
        raise ValueError("TARGET must be positive")
    counts = Counter(op.engine for op in ops)
    edge_count = sum(len(op.parents) for op in ops)
    floors = {
        engine: (counts[engine] + capacity - 1) // capacity
        for engine, capacity in SLOT_LIMITS.items()
        if counts[engine]
    }
    dag_lb = max(earliest, default=-1) + 1
    impossible = [
        index
        for index in range(len(ops))
        if earliest[index] > horizon - 1 - tail[index]
    ]
    print(
        f"strict_full_dag=1 ops={len(ops)} edges={edge_count} horizon={horizon} "
        f"dag_lb={dag_lb} counts={dict(counts)} floors={floors} "
        f"experimental_lowerings={experimental_lowerings}{reduction_report}",
        flush=True,
    )
    if impossible:
        print(
            f"status=INFEASIBLE reason=dag_window empty_jobs={len(impossible)} "
            f"first={impossible[0]}",
            flush=True,
        )
        return

    external_hints, hint_counts = load_hints(
        os.environ.get("HINTS", "").split(","), ops
    )
    use_baseline = bool(int(os.environ.get("BASELINE_HINT", "1")))
    base = baseline_cycles(builder, ops) if use_baseline else None
    if base is not None:
        # The baseline is guaranteed legal for the effective DAG; this catches
        # accidental drift between the production graph and this exact model.
        validate_solution(ops, base, max(base) + 1)
        print(f"baseline_hint_makespan={max(base) + 1}", flush=True)

    hint_radius = int(os.environ.get("HINT_RADIUS", "-1"))
    radius_on_baseline = bool(int(os.environ.get("RADIUS_ON_BASELINE", "0")))
    fix_baseline = bool(int(os.environ.get("FIX_BASELINE", "0")))
    if fix_baseline and base is None:
        raise ValueError("FIX_BASELINE requires BASELINE_HINT=1")
    fixed_hint_engines = csv_set("FIX_HINT_CYCLES")
    unknown_fixed = fixed_hint_engines - set(SLOT_LIMITS)
    if unknown_fixed:
        raise ValueError(f"unknown FIX_HINT_CYCLES engines: {sorted(unknown_fixed)}")
    microslot_engines = csv_set("MICROSLOT_ENGINES")
    all_different_engines = csv_set("ALL_DIFFERENT_ENGINES")
    full_capacity_engines = frozenset(SLOT_LIMITS)
    capacity_engines = csv_set("CAPACITY_ENGINES") or full_capacity_engines
    unknown_capacity_engines = capacity_engines - set(SLOT_LIMITS)
    if unknown_capacity_engines:
        raise ValueError(
            f"unknown CAPACITY_ENGINES: {sorted(unknown_capacity_engines)}"
        )
    unknown_capacity_modes = (microslot_engines | all_different_engines) - set(
        SLOT_LIMITS
    )
    if unknown_capacity_modes:
        raise ValueError(f"unknown capacity-mode engines: {sorted(unknown_capacity_modes)}")
    overlap = microslot_engines & all_different_engines
    if overlap:
        raise ValueError(
            f"engine cannot use both microslot and direct AllDifferent: {sorted(overlap)}"
        )
    raw_capacity_exempt_ids = os.environ.get("DIAGNOSTIC_CAPACITY_EXEMPT_IDS", "")
    explicit_capacity_exempt_ids = {
        int(value) for value in raw_capacity_exempt_ids.split(",") if value
    }
    if any(index < 0 or index >= len(ops) for index in explicit_capacity_exempt_ids):
        raise ValueError("DIAGNOSTIC_CAPACITY_EXEMPT_IDS contains an invalid op id")
    exempt_group_rounds = group_round_set(
        "DIAGNOSTIC_CAPACITY_EXEMPT_GROUP_ROUNDS"
    )
    exempt_tags = csv_set("DIAGNOSTIC_CAPACITY_EXEMPT_TAGS")
    scoped_exempt_tags = csv_set(
        "DIAGNOSTIC_CAPACITY_EXEMPT_TAGS_AT_ROUNDS"
    )
    scoped_exempt_rounds = integer_set(
        "DIAGNOSTIC_CAPACITY_EXEMPT_ROUNDS"
    )
    if bool(scoped_exempt_tags) != bool(scoped_exempt_rounds):
        raise ValueError(
            "DIAGNOSTIC_CAPACITY_EXEMPT_TAGS_AT_ROUNDS and "
            "DIAGNOSTIC_CAPACITY_EXEMPT_ROUNDS must be set together"
        )
    exempt_tag_group_rounds = tag_group_round_set(
        "DIAGNOSTIC_CAPACITY_EXEMPT_TAG_GROUP_ROUNDS"
    )
    exempt_engines = (
        csv_set("DIAGNOSTIC_CAPACITY_EXEMPT_ENGINES") or capacity_engines
    )
    unknown_exempt_engines = exempt_engines - set(SLOT_LIMITS)
    if unknown_exempt_engines:
        raise ValueError(
            "unknown DIAGNOSTIC_CAPACITY_EXEMPT_ENGINES: "
            f"{sorted(unknown_exempt_engines)}"
        )
    selected_capacity_exempt_ids = {
        index
        for index, op in enumerate(ops)
        if op.engine in exempt_engines
        and (
            (op.group, op.round) in exempt_group_rounds
            or op.tag in exempt_tags
            or (
                op.tag in scoped_exempt_tags
                and op.round in scoped_exempt_rounds
            )
            or (op.tag, op.group, op.round) in exempt_tag_group_rounds
        )
    }
    capacity_exempt_indices = frozenset(
        explicit_capacity_exempt_ids | selected_capacity_exempt_ids
    )

    model = cp_model.CpModel()
    starts: list[cp_model.IntVar] = []
    bounds: list[tuple[int, int]] = []
    external_outside_bounds = 0
    total_domain = 0
    for index, op in enumerate(ops):
        lower = earliest[index]
        upper = horizon - 1 - tail[index]
        radius_hint = external_hints.get(index)
        if radius_hint is None and radius_on_baseline and base is not None:
            radius_hint = base[index]
        if hint_radius >= 0 and radius_hint is not None:
            if not lower <= radius_hint <= upper:
                external_outside_bounds += 1
            lower = max(lower, radius_hint - hint_radius)
            upper = min(upper, radius_hint + hint_radius)
        if lower > upper:
            raise ValueError(
                f"empty local domain op={index} engine={op.engine} "
                f"bounds={lower}:{upper} hint={radius_hint}"
            )
        bounds.append((lower, upper))
        total_domain += upper - lower + 1
        starts.append(model.new_int_var(lower, upper, f"s{index}"))

    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            model.add(starts[child] >= starts[parent] + lag)
    partial_paths = tuple(
        value for value in os.environ.get("PARTIAL_ORDER_HINTS", "").split(",")
        if value
    )
    partial_edges = ()
    if partial_paths:
        partial = load_partial_order_hints(partial_paths, ops)
        partial_edges = partial.effective_edges
        stride = int(os.environ.get("PARTIAL_ORDER_STRIDE", "1"))
        if stride < 1:
            raise ValueError("PARTIAL_ORDER_STRIDE must be positive")
        if stride > 1:
            partial_edges = tuple(
                edge for rank, edge in enumerate(partial_edges) if rank % stride == 0
            )
        for parent, child in partial_edges:
            model.add(starts[child] >= starts[parent] + 1)
        print(
            f"partial_order_hints paths={len(partial_paths)} "
            f"effective_edges={len(partial_edges)}",
            flush=True,
        )

    # Research-only symmetry breaking.  Many LOAD tree-gather lanes are
    # byte-for-byte identical even in the *full* DAG (same opcode, tag,
    # group/round and exact parent map).  Their permutations are semantically
    # interchangeable, but the cumulative propagator otherwise explores all
    # 8! lane orders.  Ordering only these proven-identical classes is safe;
    # projected-parent classes are intentionally not used here because their
    # hidden non-LOAD parents can differ.
    symmetry_mode = os.environ.get("SYMMETRY_ENGINES", "")
    if bool(int(os.environ.get("SYMMETRY_LOAD", "0"))):
        symmetry_mode = "load" if not symmetry_mode else symmetry_mode + ",load"
    symmetry_engines = {
        value for value in symmetry_mode.split(",") if value
    }
    if symmetry_engines:
        from collections import defaultdict

        classes: dict[tuple[object, ...], list[int]] = defaultdict(list)
        for index, op in enumerate(ops):
            if op.engine not in symmetry_engines:
                continue
            key = (
                op.engine,
                op.slot[0] if op.slot else None,
                op.tag,
                op.group,
                op.round,
                tuple(sorted((int(parent), int(lag)) for parent, lag in op.parents.items())),
            )
            if bool(int(os.environ.get("SYMMETRY_STRICT", "0"))):
                key += (tuple(sorted((int(child), int(lag)) for child, lag in children[index])),)
            classes[key].append(index)
        symmetry_edges = 0
        for members in classes.values():
            if len(members) < 2:
                continue
            members.sort()
            for previous, current in zip(members, members[1:]):
                model.add(starts[previous] <= starts[current])
                symmetry_edges += 1
            if bool(int(os.environ.get("SYMMETRY_PACK", "0"))):
                engine = ops[members[0]].engine
                capacity = SLOT_LIMITS[engine]
                if engine == "load":
                    anchor = starts[members[0]]
                    for position, member in enumerate(members[1:], 1):
                        model.add(starts[member] == anchor + position // capacity)
                elif len(members) <= capacity:
                    anchor = starts[members[0]]
                    for member in members[1:]:
                        model.add(starts[member] == anchor)
        print(
            f"symmetry_classes={sum(len(v) > 1 for v in classes.values())} "
            f"symmetry_edges={symmetry_edges} engines={sorted(symmetry_engines)}",
            flush=True,
        )

    intervals_by_engine: dict[str, list[cp_model.IntervalVar]] = {
        engine: [] for engine in SLOT_LIMITS
    }
    for index, op in enumerate(ops):
        if index not in capacity_exempt_indices:
            intervals_by_engine[op.engine].append(
                model.new_fixed_size_interval_var(starts[index], 1, f"i{index}")
            )

    capacity_forms: dict[str, str] = {}
    microslot_variables = 0
    for engine, capacity in SLOT_LIMITS.items():
        indices = [
            index
            for index, op in enumerate(ops)
            if op.engine == engine and index not in capacity_exempt_indices
        ]
        if engine not in capacity_engines:
            capacity_forms[engine] = "disabled_diagnostic"
            continue
        if not indices:
            continue
        if engine in microslot_engines:
            microslots: list[cp_model.IntVar] = []
            for index in indices:
                lower, upper = bounds[index]
                lane = model.new_int_var(0, capacity - 1, f"lane_{engine}_{index}")
                microslot = model.new_int_var(
                    capacity * lower,
                    capacity * upper + capacity - 1,
                    f"microslot_{engine}_{index}",
                )
                model.add(microslot == capacity * starts[index] + lane)
                microslots.append(microslot)
                microslot_variables += 2
            model.add_all_different(microslots)
            capacity_forms[engine] = "microslot_all_different"
        elif engine in all_different_engines:
            if capacity != 1:
                raise ValueError(
                    f"ALL_DIFFERENT_ENGINES only supports capacity-one engines; "
                    f"{engine} has capacity {capacity}. Use MICROSLOT_ENGINES."
                )
            model.add_all_different([starts[index] for index in indices])
            capacity_forms[engine] = "all_different"
        else:
            model.add_cumulative(
                intervals_by_engine[engine], [1] * len(indices), capacity
            )
            capacity_forms[engine] = "cumulative"

    fixed = 0
    for index, cycle in external_hints.items():
        engine = ops[index].engine
        lower, upper = bounds[index]
        clamped = min(upper, max(lower, cycle))
        model.add_hint(starts[index], clamped)
        if engine in fixed_hint_engines:
            if not lower <= cycle <= upper:
                raise ValueError(
                    f"fixed hint outside domain op={index} {engine} "
                    f"cycle={cycle} bounds={lower}:{upper}"
                )
            model.add(starts[index] == cycle)
            fixed += 1
    if base is not None:
        for index, cycle in enumerate(base):
            if index not in external_hints:
                lower, upper = bounds[index]
                model.add_hint(starts[index], min(upper, max(lower, cycle)))
            if fix_baseline:
                lower, upper = bounds[index]
                if not lower <= cycle <= upper:
                    raise ValueError(
                        f"baseline outside domain op={index} cycle={cycle} "
                        f"bounds={lower}:{upper}"
                    )
                model.add(starts[index] == cycle)
                fixed += 1

    # Optional exact optimization mode.  The normal tool is a feasibility
    # oracle at a caller-supplied horizon; when a complete incumbent is
    # available, minimizing one shared makespan variable lets CP-SAT search
    # for a strictly shorter calendar without changing the production graph.
    # Keep this opt-in because the extra 20k precedence bounds are expensive
    # for the usual diagnostic feasibility runs.
    minimize_makespan = bool(int(os.environ.get("MINIMIZE_MAKESPAN", "0")))
    makespan_var = None
    if minimize_makespan:
        makespan_var = model.new_int_var(0, horizon, "makespan")
        for start in starts:
            model.add(makespan_var >= start + 1)
        model.minimize(makespan_var)

    proto = model.proto
    print(
        f"model starts={len(starts)} precedence={edge_count} "
        f"intervals={sum(map(len, intervals_by_engine.values()))} "
        f"capacity_forms={capacity_forms} microslot_vars={microslot_variables} "
        f"domain_values={total_domain} external_hints={len(external_hints)} "
        f"hint_counts={hint_counts} hint_radius={hint_radius} "
        f"outside_exact_bounds={external_outside_bounds} fixed={fixed} "
        f"capacity_exempt_ids={len(capacity_exempt_indices)} "
        f"capacity_exempt_groups={sorted(exempt_group_rounds)} "
        f"capacity_exempt_tags={sorted(exempt_tags)} "
        f"capacity_exempt_scoped_tags={sorted(scoped_exempt_tags)} "
        f"capacity_exempt_scoped_rounds={sorted(scoped_exempt_rounds)} "
        f"capacity_exempt_tag_group_rounds={sorted(exempt_tag_group_rounds)} "
        f"fix_baseline={int(fix_baseline)} "
        f"minimize_makespan={int(minimize_makespan)} "
        f"proto_vars={len(proto.variables)} proto_constraints={len(proto.constraints)}",
        flush=True,
    )
    validation_error = model.validate()
    if validation_error:
        raise ValueError(f"model validation error: {validation_error}")
    if bool(int(os.environ.get("MODEL_ONLY", "0"))):
        print("status=MODEL_OK", flush=True)
        return

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(os.environ.get("TIME_LIMIT", "60"))
    solver.parameters.num_workers = int(os.environ.get("WORKERS", "8"))
    solver.parameters.random_seed = int(os.environ.get("RANDOM_SEED", "1"))
    solver.parameters.randomize_search = bool(int(os.environ.get("RANDOMIZE", "0")))
    solver.parameters.repair_hint = bool(int(os.environ.get("REPAIR_HINT", "1")))
    solver.parameters.hint_conflict_limit = int(
        os.environ.get("HINT_CONFLICT_LIMIT", "500000")
    )
    # Expose the solver knobs used by the offline compiler search.  The
    # default CP-SAT portfolio is excellent for ordinary models, but these
    # near-saturated VLIW calendars benefit from deterministic probing of a
    # fixed phase order and, in some runs, from disabling the expensive
    # linear relaxation entirely.
    if "LINEARIZATION_LEVEL" in os.environ:
        solver.parameters.linearization_level = int(
            os.environ["LINEARIZATION_LEVEL"]
        )
    if "CP_MODEL_PRESOLVE" in os.environ:
        solver.parameters.cp_model_presolve = bool(
            int(os.environ["CP_MODEL_PRESOLVE"])
        )
    if "SYMMETRY_LEVEL" in os.environ:
        solver.parameters.symmetry_level = int(os.environ["SYMMETRY_LEVEL"])
    if "SEARCH_BRANCHING" in os.environ:
        raw_branching = os.environ["SEARCH_BRANCHING"]
        branch_names = {
            "AUTOMATIC_SEARCH": cp_model.AUTOMATIC_SEARCH,
            "FIXED_SEARCH": cp_model.FIXED_SEARCH,
            "PORTFOLIO_SEARCH": cp_model.PORTFOLIO_SEARCH,
            "LP_SEARCH": cp_model.LP_SEARCH,
            "PSEUDO_COST_SEARCH": cp_model.PSEUDO_COST_SEARCH,
            "PORTFOLIO_WITH_QUICK_RESTART_SEARCH": cp_model.PORTFOLIO_WITH_QUICK_RESTART_SEARCH,
        }
        solver.parameters.search_branching = branch_names.get(
            raw_branching,
            getattr(cp_model, raw_branching, cp_model.AUTOMATIC_SEARCH),
        )
    if "MAX_NUM_CUTS" in os.environ:
        solver.parameters.max_num_cuts = int(os.environ["MAX_NUM_CUTS"])
    solver.parameters.log_search_progress = bool(int(os.environ.get("LOG", "0")))
    status = solver.solve(model)
    result = (
        [solver.value(start) for start in starts]
        if status in (cp_model.FEASIBLE, cp_model.OPTIMAL)
        else []
    )
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        solved_makespan = (
            solver.value(makespan_var)
            if minimize_makespan and makespan_var is not None
            else max(result) + 1
        )
    else:
        solved_makespan = None
    print(
        f"status={solver.status_name(status)} wall={solver.wall_time:.3f}s "
        f"conflicts={solver.num_conflicts} branches={solver.num_branches} "
        f"objective={solved_makespan if solved_makespan is not None else '-'}",
        flush=True,
    )
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return

    validate_solution(
        raw_ops,
        result,
        horizon,
        capacity_engines=capacity_engines,
        capacity_exempt_indices=capacity_exempt_indices,
    )
    if capacity_engines != full_capacity_engines or capacity_exempt_indices:
        diagnostic_out = os.environ.get("DIAGNOSTIC_OUT")
        if diagnostic_out:
            Path(diagnostic_out).write_text(
                json.dumps(
                    {
                        "makespan": solved_makespan,
                        "cycles": result,
                        "strict_precedence": True,
                        "diagnostic_capacity_engines": sorted(capacity_engines),
                        "diagnostic_capacity_exempt_ids": sorted(
                            capacity_exempt_indices
                        ),
                        "horizon": horizon,
                    }
                )
            )
        print(
            "diagnostic_solution_selected_capacities="
            f"{','.join(sorted(capacity_engines))} "
            f"capacity_exempt_ids={len(capacity_exempt_indices)} "
            f"makespan={max(result) + 1} "
            f"diagnostic_output={diagnostic_out or '-'}",
            flush=True,
        )
        return
    output = Path(os.environ.get("OUT", "/tmp/aopt-full-dag.json"))
    output.write_text(
        json.dumps(
            {
                "makespan": solved_makespan,
                "cycles": result,
                "strict_full_dag": True,
                "horizon": horizon,
            }
        )
    )
    print(f"validated_makespan={solved_makespan} output={output}", flush=True)


if __name__ == "__main__":
    main()
