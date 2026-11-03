"""Analyze dependency/resource critical chains in an exact variant schedule."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops, validate
from problem import SLOT_LIMITS


def main() -> None:
    schedule = json.loads(Path(os.environ["SCHEDULE"]).read_text())
    cycles = schedule["cycles"]
    configure_target()
    kernel.SCHEDULE_EXACT_CYCLES = cycles
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        if not hasattr(builder, "dag_ops"):
            raise
    ops = real_tail_ops(builder.dag_ops)
    validate(ops, cycles)

    # Fix the incumbent's per-engine order using k parallel resource lanes.
    # The resulting DAG captures which ordering decisions create its current
    # span while retaining the real instruction dependencies.
    parents = [dict(op.parents) for op in ops]
    edge_kind: dict[tuple[int, int], str] = {
        (parent, child): "dag"
        for child, op in enumerate(ops)
        for parent in op.parents
    }
    for engine, capacity in SLOT_LIMITS.items():
        if engine == "debug":
            continue
        sequence = sorted(
            (i for i, op in enumerate(ops) if op.engine == engine),
            key=lambda i: (cycles[i], i),
        )
        for position in range(capacity, len(sequence)):
            parent = sequence[position - capacity]
            child = sequence[position]
            # Capacity guarantees the two operations are in distinct cycles.
            if cycles[parent] >= cycles[child]:
                raise AssertionError((engine, parent, child, cycles[parent]))
            if parents[child].get(parent, 0) < 1:
                parents[child][parent] = 1
                edge_kind[parent, child] = f"res-{engine}"

    order = sorted(range(len(ops)), key=lambda i: (cycles[i], i))
    earliest = [0] * len(ops)
    reason = [-1] * len(ops)
    for child in order:
        for parent, lag in parents[child].items():
            candidate = earliest[parent] + lag
            if candidate > earliest[child]:
                earliest[child] = candidate
                reason[child] = parent

    endpoint = max(range(len(ops)), key=lambda i: (earliest[i], cycles[i]))
    latest = [max(cycles)] * len(ops)
    for child in reversed(order):
        for parent, lag in parents[child].items():
            latest[parent] = min(latest[parent], latest[child] - lag)
    slack = [latest[i] - earliest[i] for i in range(len(ops))]
    chain = []
    node = endpoint
    while node >= 0:
        chain.append(node)
        node = reason[node]
    chain.reverse()
    if "OUT_CHAIN" in os.environ:
        Path(os.environ["OUT_CHAIN"]).write_text(json.dumps({"indices": chain}))
    if "OUT_EXPANDED" in os.environ:
        radius = int(os.environ.get("RESOURCE_RADIUS", "1"))
        expanded = set(chain)
        chain_set = set(chain)
        for engine, capacity in SLOT_LIMITS.items():
            if engine == "debug":
                continue
            sequence = sorted(
                (i for i, op in enumerate(ops) if op.engine == engine),
                key=lambda i: (cycles[i], i),
            )
            positions = {index: position for position, index in enumerate(sequence)}
            width = capacity * radius
            for index in chain_set:
                if index not in positions:
                    continue
                position = positions[index]
                expanded.update(
                    sequence[max(0, position - width) : position + width + 1]
                )
        Path(os.environ["OUT_EXPANDED"]).write_text(
            json.dumps({"indices": sorted(expanded)})
        )
        print(f"expanded_radius_{radius}={len(expanded)}")
    if "OUT_SLACK" in os.environ:
        slack_max = int(os.environ.get("SLACK_MAX", "0"))
        slack_indices = [i for i, value in enumerate(slack) if value <= slack_max]
        Path(os.environ["OUT_SLACK"]).write_text(
            json.dumps({"indices": slack_indices})
        )
        print(f"slack_le_{slack_max}={len(slack_indices)}")
    if bool(int(os.environ.get("PRINT_PARITY", "0"))):
        parity = defaultdict(list)
        for i, op in enumerate(ops):
            if op.tag.startswith("mirror_bit"):
                parity[op.group, op.round].append(i)
        ranked_parity = sorted(
            (
                min(slack[i] for i in indices),
                max(cycles[i] for i in indices),
                group,
                rnd,
                len(indices),
                max(earliest[i] for i in indices),
            )
            for (group, rnd), indices in parity.items()
        )
        print("parity_by_slack:")
        for item in ranked_parity[: int(os.environ.get("PARITY_LIMIT", "80"))]:
            print(
                f"slack={item[0]:3d} cycle={item[1]:3d} g={item[2]:2d} "
                f"r={item[3]:2d} ops={item[4]} earliest={item[5]:3d}"
            )
    if "PRINT_CYCLES" in os.environ:
        selected_cycles = {
            int(value) for value in os.environ["PRINT_CYCLES"].split(",") if value
        }
        print("selected_cycles:")
        for i, op in enumerate(ops):
            if cycles[i] in selected_cycles:
                print(
                    f"c={cycles[i]:3d} {op.engine:5s} i={i:5d} g={op.group:2d} "
                    f"r={op.round:2d} {op.tag}"
                )
    kinds = Counter(
        edge_kind.get((left, right), "dag")
        for left, right in zip(chain, chain[1:])
    )
    counts = Counter(op.engine for op in ops)
    floors = {
        engine: (count + SLOT_LIMITS[engine] - 1) // SLOT_LIMITS[engine]
        for engine, count in counts.items()
        if engine != "debug"
    }
    print(
        f"exact_span={max(cycles) + 1} ordered_lb={earliest[endpoint] + 1} "
        f"endpoint={endpoint}"
    )
    print(f"counts={dict(counts)} floors={floors}")
    print(f"critical_edges={dict(kinds)} chain_nodes={len(chain)}")

    tail_count = int(os.environ.get("TAIL", "100"))
    print("critical_chain_tail:")
    for index in chain[-tail_count:]:
        op = ops[index]
        parent = reason[index]
        kind = edge_kind.get((parent, index), "root" if parent < 0 else "dag")
        print(
            f"{index:5d} c={cycles[index]:3d} e={earliest[index]:3d} "
            f"via={kind:9s} {op.engine:5s} g={op.group:2d} "
            f"r={op.round:2d} {op.tag}"
        )

    horizon = max(cycles) + 1
    usage = defaultdict(Counter)
    for i, op in enumerate(ops):
        usage[op.engine][cycles[i]] += 1
    print("tail_holes:")
    tail_start = max(0, horizon - int(os.environ.get("HOLE_TAIL", "120")))
    for engine in ("valu", "alu", "load", "flow"):
        holes = [
            cycle
            for cycle in range(tail_start, horizon)
            if usage[engine][cycle] < SLOT_LIMITS[engine]
        ]
        print(f"{engine}: {len(holes)} {holes}")


if __name__ == "__main__":
    main()
