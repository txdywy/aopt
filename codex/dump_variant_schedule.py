"""Dump the deterministic list schedule for an environment-configured DAG."""

from __future__ import annotations

import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops, validate


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
    earliest = [0] * len(ops)
    reason = [-1] * len(ops)
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            candidate = earliest[parent] + lag
            if candidate > earliest[child]:
                earliest[child] = candidate
                reason[child] = parent
    endpoint = max(range(len(ops)), key=earliest.__getitem__)
    chain = []
    node = endpoint
    while node >= 0:
        chain.append(node)
        node = reason[node]
    chain.reverse()
    _, cycles = builder._schedule(
        builder.dag_ops,
        int(os.environ.get("POLICY", str(kernel.SCHEDULE_POLICIES[0]))),
        return_cycles=True,
    )
    validate(ops, cycles)
    output = Path(os.environ.get("OUT", "/tmp/aopt-variant-list.json"))
    output.write_text(json.dumps({"makespan": max(cycles) + 1, "cycles": cycles}))
    print(
        f"makespan={max(cycles) + 1} dag_lb={earliest[endpoint] + 1} "
        f"ops={len(cycles)} output={output}"
    )
    if bool(int(os.environ.get("PRINT_DAG_CHAIN", "0"))):
        for index in chain:
            op = ops[index]
            print(
                f"i={index:5d} e={earliest[index]:4d} {op.engine:5s} "
                f"g={op.group:2d} r={op.round:2d} {op.tag}"
            )


if __name__ == "__main__":
    main()
