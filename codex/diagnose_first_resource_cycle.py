"""Print one concrete cycle induced by per-engine teacher orders.

This diagnostic is deliberately small: it combines the semantic DAG with
resource-order arcs from mapped engine calendars and prints the operation tags
around the first cycle.  The cycle is the actionable form of the
ALU<->VALU conflict; Hall counts alone cannot expose it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.break_resource_cycles import find_cycle
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
    edges: list[tuple[int, int, str]] = []
    for child, op in enumerate(ops):
        for parent in op.parents:
            edges.append((parent, child, "dag"))
    payloads = [
        json.loads(Path(path).read_text())
        for path in os.environ.get("HINTS", "").split(",")
        if path
    ]
    for payload in payloads:
        engine = str(payload["engine"])
        cycles = {int(i): int(c) for i, c in payload["cycles"].items()}
        order = payload.get("order")
        if order is None:
            order = sorted(cycles, key=lambda i: (cycles[i], i))
        capacity = SLOT_LIMITS[engine]
        if len(order) != len(cycles):
            raise ValueError(f"{engine} order does not cover its cycles")
        for parent, child in zip(order, order[capacity:]):
            edges.append((int(parent), int(child), engine))
    cycle = find_cycle(len(ops), edges)
    print(f"ops={len(ops)} edges={len(edges)} cycle={'yes' if cycle else 'no'}")
    if cycle is None:
        return
    # The cycle edge list is oriented parent -> child.  Reverse it so the
    # displayed operation sequence follows the directed cycle.
    chosen = [edges[index] for index in cycle]
    for parent, child, kind in reversed(chosen):
        op_parent = ops[parent]
        op_child = ops[child]
        print(
            f"{kind:4s} {parent:5d} {op_parent.engine:5s} "
            f"g={op_parent.group:2d} r={op_parent.round:2d} "
            f"{op_parent.tag:34s} -> {child:5d} {op_child.engine:5s} "
            f"g={op_child.group:2d} r={op_child.round:2d} {op_child.tag}",
        )


if __name__ == "__main__":
    main()
