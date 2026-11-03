"""Turn per-engine calendars into a legal full-DAG schedule.

This is a lightweight order oracle for research hints: it adds the usual
capacity-k order edges to the exact kernel DAG and computes the longest-path
calendar.  Unlike a CP hint, every emitted cycle is legal by construction when
the combined order graph is acyclic.
"""

from __future__ import annotations

from collections import Counter
import heapq
import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from problem import SLOT_LIMITS


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
    n = len(ops)
    children = [[] for _ in ops]
    indegree = [len(op.parents) for op in ops]
    edges: list[tuple[int, int, int, str]] = []
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            edges.append((parent, child, lag, "dag"))
            children[parent].append(child)
    paths = [Path(x) for x in os.environ.get("HINTS", "").split(",") if x]
    if not paths:
        raise ValueError("HINTS must contain per-engine JSON files")
    for path in paths:
        payload = json.loads(path.read_text())
        engine = payload.get("engine")
        if engine not in SLOT_LIMITS:
            raise ValueError(f"{path}: missing/invalid engine")
        cycles = {int(i): int(c) for i, c in payload["cycles"].items()}
        selected = [i for i, op in enumerate(ops) if op.engine == engine]
        if set(cycles) != set(selected):
            raise ValueError(f"{path}: incomplete {engine} hint")
        order = sorted(selected, key=lambda i: (cycles[i], i))
        k = SLOT_LIMITS[engine]
        raw_edges = payload.get("resource_edges")
        order_edges = (
            [(int(a), int(b)) for a, b in raw_edges]
            if raw_edges is not None
            else list(zip(order, order[k:]))
        )
        for a, b in order_edges:
            edges.append((a, b, 1, f"resource:{engine}"))
            children[a].append(b)
            indegree[b] += 1
        print(f"order={engine} jobs={len(order)}", flush=True)

    # Kahn topological sort, preserving deterministic operation order.
    ready = [i for i, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    topo = []
    while ready:
        p = heapq.heappop(ready)
        topo.append(p)
        for c in children[p]:
            indegree[c] -= 1
            if indegree[c] == 0:
                heapq.heappush(ready, c)
    if len(topo) != n:
        residual = {i for i, d in enumerate(indegree) if d}
        print(f"cycle residual={len(residual)}")
        # Show one residual walk for debugging order interactions.
        node = next(iter(residual)); seen = {}; walk = []
        walk_kinds = []
        while node not in seen:
            seen[node] = len(walk); walk.append(node)
            found = next(((a, kind) for a, b, _, kind in edges if b == node and a in residual), None)
            parent, kind = found if found is not None else (None, "")
            if parent is None: break
            walk_kinds.append(kind)
            node = parent
        start = seen.get(node, 0)
        print("cycle_nodes", walk[start:][:24])
        print("cycle_kinds", walk_kinds[start:][:24])
        for idx in walk[start:][:24]:
            op = ops[idx]
            print(f"cycle_op {idx} {op.engine} g={op.group} r={op.round} {op.tag}")
        return

    cycles = [0] * n
    parent_edges = [[] for _ in ops]
    for a, b, lag, kind in edges:
        parent_edges[b].append((a, lag))
    for child in topo:
        if parent_edges[child]:
            cycles[child] = max(cycles[p] + lag for p, lag in parent_edges[child])
    makespan = max(cycles) + 1
    usage = Counter((cycles[i], op.engine) for i, op in enumerate(ops))
    violations = [(key, value) for key, value in usage.items() if value > SLOT_LIMITS[key[1]]]
    print(f"status=LEGAL makespan={makespan} violations={len(violations)}")
    if violations:
        print("violation_sample", violations[:10]); return
    out = Path(os.environ.get("OUT", ".search959/fixed-orders-full.json"))
    out.write_text(json.dumps({"cycles": cycles, "horizon": makespan}))
    print(f"output={out}")


if __name__ == "__main__":
    main()
