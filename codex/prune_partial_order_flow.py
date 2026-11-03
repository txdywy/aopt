"""Capacity-aware pruning of strict partial resource-order hints.

``filter_partial_order_hints`` guarantees that a retained partial order is a
directed acyclic graph whose precedence-only span fits the requested horizon.
That is necessary but not sufficient for this kernel: Flow has 955 unit jobs
in a 959-cycle horizon, so foreign ALU/VALU order arcs can consume its four
available bubbles without extending the ordinary critical path.

This tool uses an exact *necessary-condition* oracle.  It keeps every raw DAG
edge, models a start variable for every operation, enforces the capacity of
one selected engine (normally Flow), and makes each optional partial resource
arc an assumption literal.  When CP-SAT proves the set infeasible, it returns
an assumption core.  The tool drops one or a configurable small batch from
that core and repeats.  A written result therefore has two strict guarantees:

* original DAG + retained arcs is acyclic and its precedence span fits TARGET;
* the selected engine's exact capacity is feasible at TARGET.

It deliberately does *not* claim whole-machine feasibility: all other engine
capacities remain for the final full CP model.  It is a safe way to turn a
large cross-resource edge pool into a capacity-aware anchor or LNS seed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import heapq
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

from ortools.sat.python import cp_model

from codex.partial_order_hints import (
    PartialOrderHints,
    _build_target_ops,
    audit_augmented_dag,
    load_partial_order_hints,
)
from problem import SLOT_LIMITS


@dataclass(frozen=True)
class RawDagData:
    topological: tuple[int, ...]
    earliest: tuple[int, ...]
    tail: tuple[int, ...]


def raw_dag_data(ops: Sequence[object]) -> RawDagData:
    """Compute exact raw-DAG time windows used by the capacity oracle."""
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
        raise ValueError("raw kernel DAG is cyclic")
    earliest = [0] * len(ops)
    for child in topological:
        for parent, lag in ops[child].parents.items():
            earliest[child] = max(earliest[child], earliest[parent] + lag)
    tail = [0] * len(ops)
    for parent in reversed(topological):
        tail[parent] = max(
            (lag + tail[child] for child, lag in children[parent]), default=0
        )
    return RawDagData(tuple(topological), tuple(earliest), tuple(tail))


def load_hint_cycles(
    paths: Iterable[str], ops: Sequence[object]
) -> dict[int, int]:
    """Load optional soft start hints, fail-closed on malformed endpoints."""
    result: dict[int, int] = {}
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path)
        payload = json.loads(path.read_text())
        raw_cycles = payload.get("cycles")
        if isinstance(raw_cycles, list):
            items = enumerate(raw_cycles)
        elif isinstance(raw_cycles, dict):
            items = ((int(index), cycle) for index, cycle in raw_cycles.items())
        else:
            raise ValueError(f"{path}: missing cycles for HINTS")
        for index, cycle in items:
            if not 0 <= index < len(ops):
                raise ValueError(f"{path}: invalid hint operation {index}")
            if isinstance(cycle, bool):
                raise ValueError(f"{path}: boolean issue cycle for {index}")
            cycle = int(cycle)
            previous = result.get(index)
            if previous is not None and previous != cycle:
                raise ValueError(
                    f"{path}: conflicting hint for operation {index}: "
                    f"{previous} vs {cycle}"
                )
            result[index] = cycle
    return result


def _literal_variable_index(literal: int) -> int:
    """Map a CP-SAT signed literal to its BoolVar proto index."""
    return literal if literal >= 0 else -literal - 1


def choose_drops(
    core_indices: Sequence[int],
    edges: Sequence[tuple[int, int]],
    ops: Sequence[object],
    count: int,
    policy: str,
) -> list[int]:
    """Choose deterministic arc removals from one infeasibility core."""
    if count <= 0:
        raise ValueError("CORE_DROP_COUNT must be positive")
    core = list(core_indices)
    if policy == "latest":
        ordered = sorted(core, reverse=True)
    elif policy == "earliest":
        ordered = sorted(core)
    elif policy == "foreign-flow-first":
        # Preserve the selected engine's own calendar when possible; for a
        # nearly saturated Flow oracle, foreign ALU/VALU order is normally the
        # least trustworthy source of cross-engine timing restrictions.
        rank = {"alu": 0, "valu": 1, "load": 2, "store": 3, "flow": 4}
        ordered = sorted(
            core,
            key=lambda index: (rank.get(ops[edges[index][1]].engine, 5), -index),
        )
    else:
        raise ValueError(
            f"unknown CORE_DROP_POLICY {policy!r}; choose latest, earliest, "
            "or foreign-flow-first"
        )
    return ordered[: min(count, len(ordered))]


def partial_with_active_edges(
    ops: Sequence[object], partial: PartialOrderHints, active: set[int]
) -> PartialOrderHints:
    """Serialize active optional arcs plus all redundant direct-DAG provenance."""
    active_effective = tuple(
        edge for index, edge in enumerate(partial.effective_edges) if index in active
    )
    effective_set = set(partial.effective_edges)
    active_set = set(active_effective)
    raw = tuple(
        edge
        for edge in partial.resource_edges
        if edge not in effective_set or edge in active_set
    )
    return PartialOrderHints(
        paths=partial.paths,
        resource_edges=raw,
        effective_edges=active_effective,
        by_engine=dict(Counter(ops[child].engine for _, child in raw)),
        effective_by_engine=dict(
            Counter(ops[child].engine for _, child in active_effective)
        ),
        direct_dag_redundant=len(raw) - len(active_effective),
    )


def source_payloads(paths: Sequence[Path]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        engine = payload.get("engine")
        if not isinstance(engine, str):
            raise AssertionError(f"validated input missing engine: {path}")
        result[engine] = payload
    return result


def write_output(
    ops: Sequence[object],
    partial: PartialOrderHints,
    output_prefix: Path,
    metadata: dict[str, object],
) -> tuple[list[Path], Path]:
    """Write filtered payloads and independently audit the exact files."""
    payloads = source_payloads(partial.paths)
    by_engine: dict[str, list[tuple[int, int]]] = {}
    for edge in partial.resource_edges:
        by_engine.setdefault(ops[edge[1]].engine, []).append(edge)
    outputs: list[Path] = []
    for engine, payload in payloads.items():
        result = dict(payload)
        result["resource_edges"] = [
            [int(parent), int(child)] for parent, child in by_engine.get(engine, [])
        ]
        result["capacity_oracle_seed"] = {
            **metadata,
            "engine": engine,
            "resource_edges": len(by_engine.get(engine, [])),
            "effective_resource_edges": partial.effective_by_engine.get(engine, 0),
        }
        output = Path(f"{output_prefix}-{engine}.json")
        output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
        outputs.append(output)
    replay = load_partial_order_hints(outputs, ops)
    replay_audit = audit_augmented_dag(ops, replay)
    if replay.resource_edges != partial.resource_edges:
        raise AssertionError("output resource arcs changed under strict replay")
    if replay_audit.longest_path_span != metadata["precedence_span"]:
        raise AssertionError("output span differs from capacity-oracle seed")
    report = {
        **metadata,
        "output_paths": [str(path) for path in outputs],
        "resource_edges": len(replay.resource_edges),
        "effective_edges": len(replay.effective_edges),
        "direct_dag_redundant": replay.direct_dag_redundant,
        "effective_by_engine": replay.effective_by_engine,
    }
    report_path = Path(f"{output_prefix}.report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return outputs, report_path


def main() -> None:
    raw_paths = [
        path for path in os.environ.get("PARTIAL_ORDER_HINTS", "").split(",") if path
    ]
    if not raw_paths:
        raise ValueError("PARTIAL_ORDER_HINTS must name strict partial JSON files")
    horizon = int(os.environ.get("TARGET", os.environ.get("HORIZON", "959")))
    check_engine = os.environ.get("CHECK_ENGINE", "flow")
    if check_engine not in SLOT_LIMITS:
        raise ValueError(f"unknown CHECK_ENGINE {check_engine!r}")
    if SLOT_LIMITS[check_engine] != 1:
        raise ValueError(
            "this compact oracle currently requires a capacity-one CHECK_ENGINE; "
            f"{check_engine} has capacity {SLOT_LIMITS[check_engine]}"
        )
    max_rounds = int(os.environ.get("MAX_CORE_ROUNDS", "256"))
    drop_count = int(os.environ.get("CORE_DROP_COUNT", "1"))
    drop_policy = os.environ.get("CORE_DROP_POLICY", "foreign-flow-first")
    output_prefix = Path(
        os.environ.get("OUT_PREFIX", "/tmp/aopt-capacity-pruned-partial")
    )

    ops = _build_target_ops()
    raw = raw_dag_data(ops)
    if max(raw.earliest, default=-1) + 1 > horizon:
        raise ValueError("raw DAG already exceeds requested TARGET")
    partial = load_partial_order_hints(raw_paths, ops)
    input_audit = audit_augmented_dag(ops, partial)
    if input_audit.longest_path_span > horizon:
        raise ValueError(
            "PARTIAL_ORDER_HINTS must first pass precedence filtering: "
            f"span={input_audit.longest_path_span} TARGET={horizon}"
        )

    model = cp_model.CpModel()
    starts: list[cp_model.IntVar] = []
    for index, op in enumerate(ops):
        lower = raw.earliest[index]
        upper = horizon - 1 - raw.tail[index]
        if lower > upper:
            raise ValueError(f"raw DAG has an empty target window at operation {index}")
        starts.append(model.new_int_var(lower, upper, f"s{index}"))
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            model.add(starts[child] >= starts[parent] + lag)
    checked = [index for index, op in enumerate(ops) if op.engine == check_engine]
    model.add_all_different([starts[index] for index in checked])

    optional: list[cp_model.BoolVar] = []
    for index, (parent, child) in enumerate(partial.effective_edges):
        keep = model.new_bool_var(f"keep_{index}")
        model.add(starts[child] >= starts[parent] + 1).only_enforce_if(keep)
        optional.append(keep)
    hint_cycles = load_hint_cycles(os.environ.get("HINTS", "").split(","), ops)
    for index, cycle in hint_cycles.items():
        lower = raw.earliest[index]
        upper = horizon - 1 - raw.tail[index]
        model.add_hint(starts[index], min(upper, max(lower, cycle)))
    validation = model.validate()
    if validation:
        raise ValueError(f"capacity oracle model validation error: {validation}")

    active = set(range(len(optional)))
    literal_to_edge = {keep.index: index for index, keep in enumerate(optional)}
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(os.environ.get("TIME_LIMIT", "30"))
    solver.parameters.num_workers = int(os.environ.get("WORKERS", "1"))
    solver.parameters.random_seed = int(os.environ.get("RANDOM_SEED", "1"))
    solver.parameters.randomize_search = bool(int(os.environ.get("RANDOMIZE", "0")))
    solver.parameters.repair_hint = bool(int(os.environ.get("REPAIR_HINT", "1")))
    solver.parameters.hint_conflict_limit = int(
        os.environ.get("HINT_CONFLICT_LIMIT", "500000")
    )

    status: int | None = None
    last_core: list[int] = []
    for round_index in range(max_rounds + 1):
        model.clear_assumptions()
        model.add_assumptions([optional[index] for index in sorted(active)])
        status = solver.solve(model)
        print(
            f"capacity_oracle round={round_index} engine={check_engine} "
            f"active={len(active)} status={solver.status_name(status)} "
            f"wall={solver.wall_time:.3f}s conflicts={solver.num_conflicts} "
            f"branches={solver.num_branches}",
            flush=True,
        )
        if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            break
        if status != cp_model.INFEASIBLE:
            raise RuntimeError(
                "capacity oracle did not prove feasibility or infeasibility; "
                "increase TIME_LIMIT or simplify the input seed"
            )
        core = [
            literal_to_edge[_literal_variable_index(int(literal))]
            for literal in solver.sufficient_assumptions_for_infeasibility()
            if _literal_variable_index(int(literal)) in literal_to_edge
        ]
        core = sorted(set(core))
        if not core:
            raise AssertionError("infeasibility core contains no optional resource arc")
        last_core = core
        drop = choose_drops(
            core, partial.effective_edges, ops, drop_count, drop_policy
        )
        active.difference_update(drop)
        print(
            f"capacity_oracle_core size={len(core)} "
            f"by_engine={dict(Counter(ops[partial.effective_edges[index][1]].engine for index in core))} "
            f"drop={drop} "
            f"drop_by_engine={dict(Counter(ops[partial.effective_edges[index][1]].engine for index in drop))}",
            flush=True,
        )
    else:
        raise RuntimeError(f"capacity oracle exceeded MAX_CORE_ROUNDS={max_rounds}")

    assert status in (cp_model.FEASIBLE, cp_model.OPTIMAL)
    result = partial_with_active_edges(ops, partial, active)
    audit = audit_augmented_dag(ops, result)
    if audit.longest_path_span > horizon:
        raise AssertionError("capacity-pruned result unexpectedly exceeds TARGET")
    flow_cycles = {
        str(index): solver.value(starts[index]) for index in checked
    }
    flow_usage = Counter(flow_cycles.values())
    if max(flow_usage.values(), default=0) > 1:
        raise AssertionError("capacity oracle emitted overlapping checked-engine jobs")
    metadata = {
        "strict_capacity_oracle_seed": True,
        "target": horizon,
        "checked_engine": check_engine,
        "checked_engine_jobs": len(checked),
        "checked_engine_capacity": SLOT_LIMITS[check_engine],
        "checked_engine_feasible": True,
        "input_effective_edges": len(partial.effective_edges),
        "retained_effective_edges": len(result.effective_edges),
        "dropped_effective_edges": len(partial.effective_edges) - len(result.effective_edges),
        "retained_effective_by_engine": result.effective_by_engine,
        "precedence_span": audit.longest_path_span,
        "core_drop_policy": drop_policy,
        "core_drop_count": drop_count,
        "last_core_size": len(last_core),
    }
    outputs, report = write_output(ops, result, output_prefix, metadata)
    flow_output = Path(f"{output_prefix}-{check_engine}-oracle-cycles.json")
    flow_output.write_text(
        json.dumps(
            {
                "engine": check_engine,
                "horizon": horizon,
                "cycles": flow_cycles,
                "capacity_oracle": True,
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    print(
        f"capacity_oracle=FEASIBLE retained={len(result.effective_edges)} "
        f"dropped={len(partial.effective_edges) - len(result.effective_edges)} "
        f"span={audit.longest_path_span} outputs={','.join(map(str, outputs))} "
        f"flow_oracle={flow_output} report={report}",
        flush=True,
    )


if __name__ == "__main__":
    main()
