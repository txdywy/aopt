"""Search virtual execution-lane matchings for an exact VLIW schedule.

Slots in one engine bundle are interchangeable.  Assigning each cycle's ops
to persistent virtual lanes yields a compact precedence certificate, but a
naive index-order assignment can manufacture an unnecessarily long resource
chain.  This tool randomizes those matchings and minimizes their DAG span.
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import random

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops, validate
from problem import SLOT_LIMITS


def main() -> None:
    schedule = json.loads(Path(os.environ["SCHEDULE"]).read_text())
    cycles = schedule["cycles"]
    configure_target()
    kernel.SCHEDULE_EXACT_CYCLES = cycles
    builder = kernel.KernelBuilder()
    builder.build_kernel(10, 2047, 256, 16)
    ops = real_tail_ops(builder.dag_ops)
    validate(ops, cycles)

    order = sorted(range(len(ops)), key=lambda i: (cycles[i], i))
    blocks: dict[str, list[list[int]]] = {}
    horizon = max(cycles) + 1
    for engine in SLOT_LIMITS:
        if engine == "debug":
            continue
        by_cycle: list[list[int]] = [[] for _ in range(horizon)]
        for i, op in enumerate(ops):
            if op.engine == engine:
                by_cycle[cycles[i]].append(i)
        blocks[engine] = by_cycle

    rng = random.Random(int(os.environ.get("SEED", "1")))
    trials = int(os.environ.get("TRIALS", "1000"))
    best_span = 1 << 60
    best_parent: list[int] | None = None

    for trial in range(trials):
        resource_parent = [-1] * len(ops)
        for engine, by_cycle in blocks.items():
            capacity = SLOT_LIMITS[engine]
            lane_tail = [-1] * capacity
            for cycle_ops in by_cycle:
                if not cycle_ops:
                    continue
                current = cycle_ops.copy()
                if trial:
                    rng.shuffle(current)
                    lanes = rng.sample(range(capacity), len(current))
                else:
                    lanes = list(range(len(current)))
                for lane, op_index in zip(lanes, current):
                    resource_parent[op_index] = lane_tail[lane]
                    lane_tail[lane] = op_index

        earliest = [0] * len(ops)
        for child in order:
            ready = 0
            for parent, lag in ops[child].parents.items():
                ready = max(ready, earliest[parent] + lag)
            parent = resource_parent[child]
            if parent >= 0:
                ready = max(ready, earliest[parent] + 1)
            earliest[child] = ready
        span = max(earliest) + 1
        if span < best_span:
            best_span = span
            best_parent = resource_parent
            print(f"trial={trial} span={span}", flush=True)
            Path(os.environ.get("OUT", "/tmp/aopt-lane-order.json")).write_text(
                json.dumps(
                    {
                        "span": span,
                        "resource_parent": resource_parent,
                        "earliest": earliest,
                    }
                )
            )

    if best_parent is None:
        raise AssertionError("no lane assignment generated")
    print(f"best_span={best_span} trials={trials}")


if __name__ == "__main__":
    main()
