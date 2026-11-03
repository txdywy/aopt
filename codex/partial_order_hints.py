"""Strict parsing and auditing for partial resource-order hints.

The per-engine solvers can produce a very useful *partial* ordering of the
operations assigned to an engine.  Such an ordering is not a schedule: the
remaining resource choices must still be made by a global model.  It is safe
to add the retained arcs to that global model only when all of the following
hold:

* every arc endpoint belongs to the declared engine and ``matched_indices``;
* an input cannot silently omit or redirect an endpoint;
* the original DAG plus the retained arcs is acyclic.

This module deliberately does not interpret ``cycles`` or synthesize lane
edges from an ``order`` field.  ``PARTIAL_ORDER_HINTS`` means exactly the
explicit ``resource_edges`` carried by ``break_resource_cycles.py`` output.
That distinction prevents independently feasible engine calendars from being
mistaken for a strict global schedule.

``solve_full_dag.py`` can import :func:`load_partial_order_hints` and add
``effective_edges`` as hard ``+1`` precedence constraints.  The module also
has a small CLI for independently auditing a seed before it is given to a
large CP-SAT run::

    PARTIAL_ORDER_HINTS=/tmp/seed-alu.json,/tmp/seed-valu.json \\
      python3 -m codex.partial_order_hints
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import heapq
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from problem import SLOT_LIMITS


class _OpLike(Protocol):
    """The small portion of a kernel operation needed by this module."""

    engine: str
    parents: Mapping[int, int]


@dataclass(frozen=True)
class PartialOrderHints:
    """Validated retained resource arcs and useful accounting information.

    ``resource_edges`` contains every explicit arc present in the hint files.
    ``effective_edges`` excludes arcs already implied by an original direct
    precedence of lag at least one; adding those again to CP-SAT would be
    redundant.  Both sets remain strict constraints of the augmented graph.
    """

    paths: tuple[Path, ...]
    resource_edges: tuple[tuple[int, int], ...]
    effective_edges: tuple[tuple[int, int], ...]
    by_engine: dict[str, int]
    effective_by_engine: dict[str, int]
    direct_dag_redundant: int


@dataclass(frozen=True)
class AugmentedDagAudit:
    """Acyclicity and critical-span summary for a strict partial seed."""

    topological: tuple[int, ...]
    raw_edges: int
    effective_resource_edges: int
    longest_path_span: int


def _index(value: object, context: str) -> int:
    """Parse one JSON operation index without accepting booleans or floats."""
    if isinstance(value, bool):
        raise ValueError(f"{context}: boolean is not an operation index")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as error:
            raise ValueError(f"{context}: invalid operation index {value!r}") from error
    raise ValueError(f"{context}: invalid operation index {value!r}")


def _require_object(payload: object, path: Path) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: partial-order payload must be a JSON object")
    return payload


def _validate_endpoint(
    index: int,
    *,
    ops: Sequence[_OpLike],
    engine: str,
    path: Path,
    context: str,
) -> None:
    if not 0 <= index < len(ops):
        raise ValueError(f"{path}: {context} has invalid operation index {index}")
    if ops[index].engine != engine:
        raise ValueError(
            f"{path}: {context} op {index} is {ops[index].engine}, not {engine}"
        )


def load_partial_order_hints(
    raw_paths: Iterable[str | Path],
    ops: Sequence[_OpLike],
    *,
    slot_limits: Mapping[str, int] = SLOT_LIMITS,
) -> PartialOrderHints:
    """Load explicit, engine-local resource arcs with fail-closed validation.

    A partial hint must provide all three fields ``engine``,
    ``matched_indices`` and ``resource_edges``.  In particular, this function
    refuses the convenient but unsafe fallback of reconstructing edges from
    independently solved issue cycles.  A repeated engine or repeated arc is
    rejected rather than silently merged, so a seed always has one clear
    provenance.
    """
    paths = tuple(Path(raw_path) for raw_path in raw_paths if str(raw_path))
    if not paths:
        return PartialOrderHints((), (), (), {}, {}, 0)

    seen_engines: set[str] = set()
    seen_edges: set[tuple[int, int]] = set()
    resource_edges: list[tuple[int, int]] = []
    effective_edges: list[tuple[int, int]] = []
    by_engine: Counter[str] = Counter()
    effective_by_engine: Counter[str] = Counter()
    direct_dag_redundant = 0

    for path in paths:
        try:
            payload = _require_object(json.loads(path.read_text()), path)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: invalid JSON: {error.msg}") from error

        engine = payload.get("engine")
        if not isinstance(engine, str) or engine not in slot_limits:
            raise ValueError(f"{path}: unknown or missing engine {engine!r}")
        if engine in seen_engines:
            raise ValueError(f"{path}: duplicate partial-order engine {engine}")
        seen_engines.add(engine)

        raw_matched = payload.get("matched_indices")
        if not isinstance(raw_matched, list):
            raise ValueError(f"{path}: missing list matched_indices")
        matched: set[int] = set()
        for position, raw_index in enumerate(raw_matched):
            index = _index(raw_index, f"{path}: matched_indices[{position}]")
            _validate_endpoint(
                index,
                ops=ops,
                engine=engine,
                path=path,
                context="matched_indices",
            )
            if index in matched:
                raise ValueError(f"{path}: duplicate matched index {index}")
            matched.add(index)
        if not matched:
            raise ValueError(f"{path}: matched_indices is empty")

        raw_edges = payload.get("resource_edges")
        if not isinstance(raw_edges, list):
            raise ValueError(f"{path}: missing list resource_edges")
        local_edges: set[tuple[int, int]] = set()
        for position, raw_edge in enumerate(raw_edges):
            if not isinstance(raw_edge, (list, tuple)) or len(raw_edge) != 2:
                raise ValueError(
                    f"{path}: resource_edges[{position}] must be [parent, child]"
                )
            parent = _index(
                raw_edge[0], f"{path}: resource_edges[{position}][0]"
            )
            child = _index(
                raw_edge[1], f"{path}: resource_edges[{position}][1]"
            )
            _validate_endpoint(
                parent,
                ops=ops,
                engine=engine,
                path=path,
                context=f"resource_edges[{position}] parent",
            )
            _validate_endpoint(
                child,
                ops=ops,
                engine=engine,
                path=path,
                context=f"resource_edges[{position}] child",
            )
            if parent not in matched or child not in matched:
                raise ValueError(
                    f"{path}: resource_edges[{position}] endpoint is not matched"
                )
            if parent == child:
                raise ValueError(f"{path}: resource_edges[{position}] is a self-edge")
            edge = (parent, child)
            if edge in local_edges or edge in seen_edges:
                raise ValueError(f"{path}: duplicate resource edge {parent}->{child}")
            local_edges.add(edge)
            seen_edges.add(edge)
            resource_edges.append(edge)
            by_engine[engine] += 1

            # A pre-existing stronger direct dependency already provides the
            # requested +1 constraint.  Do not inflate the global model with
            # a duplicate; a lag-0 direct relation, should one exist, is not
            # enough and intentionally remains an effective +1 edge.
            if ops[child].parents.get(parent, -1) >= 1:
                direct_dag_redundant += 1
            else:
                effective_edges.append(edge)
                effective_by_engine[engine] += 1

    return PartialOrderHints(
        paths=paths,
        resource_edges=tuple(resource_edges),
        effective_edges=tuple(effective_edges),
        by_engine=dict(by_engine),
        effective_by_engine=dict(effective_by_engine),
        direct_dag_redundant=direct_dag_redundant,
    )


def audit_augmented_dag(
    ops: Sequence[_OpLike], partial: PartialOrderHints
) -> AugmentedDagAudit:
    """Fail if original DAG plus strict effective arcs contains a cycle.

    The returned longest-path span uses the original edge lag and the fixed
    unit lag of each effective resource edge.  It is a precedence-only lower
    bound, not a claim that the partially ordered capacity problem fits in a
    given horizon.
    """
    parents: list[dict[int, int]] = [dict(op.parents) for op in ops]
    for parent, child in partial.effective_edges:
        parents[child][parent] = max(parents[child].get(parent, -1), 1)

    children: list[list[tuple[int, int]]] = [[] for _ in ops]
    indegree = [0] * len(ops)
    for child, child_parents in enumerate(parents):
        indegree[child] = len(child_parents)
        for parent, lag in child_parents.items():
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
        residual = [index for index, degree in enumerate(indegree) if degree]
        raise ValueError(
            "partial resource-order hints introduce a cycle into the full DAG "
            f"(residual_jobs={len(residual)} first={residual[0]})"
        )

    earliest = [0] * len(ops)
    for child in topological:
        for parent, lag in parents[child].items():
            earliest[child] = max(earliest[child], earliest[parent] + lag)
    return AugmentedDagAudit(
        topological=tuple(topological),
        raw_edges=sum(len(op.parents) for op in ops),
        effective_resource_edges=len(partial.effective_edges),
        longest_path_span=max(earliest, default=-1) + 1,
    )


def _build_target_ops() -> list[kernel._Op]:
    configure_target()
    kernel.SCHEDULE_EXACT_CYCLES = None
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        if not hasattr(builder, "dag_ops"):
            raise
    return real_tail_ops(builder.dag_ops)


def main() -> None:
    """Audit ``PARTIAL_ORDER_HINTS`` against the current target DAG."""
    raw_paths = [
        value for value in os.environ.get("PARTIAL_ORDER_HINTS", "").split(",") if value
    ]
    if not raw_paths:
        raise ValueError("PARTIAL_ORDER_HINTS must name one or more JSON files")
    ops = _build_target_ops()
    partial = load_partial_order_hints(raw_paths, ops)
    audit = audit_augmented_dag(ops, partial)
    print(
        f"partial_order_hints paths={len(partial.paths)} "
        f"resource_edges={len(partial.resource_edges)} "
        f"effective_edges={len(partial.effective_edges)} "
        f"direct_dag_redundant={partial.direct_dag_redundant} "
        f"by_engine={partial.by_engine} "
        f"effective_by_engine={partial.effective_by_engine}",
        flush=True,
    )
    print(
        f"partial_order_augmented_dag=ACYCLIC ops={len(ops)} "
        f"raw_edges={audit.raw_edges} "
        f"effective_resource_edges={audit.effective_resource_edges} "
        f"precedence_span={audit.longest_path_span}",
        flush=True,
    )


if __name__ == "__main__":
    main()
