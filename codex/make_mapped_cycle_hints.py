"""Map an exact source calendar into soft per-operation target hints.

Structural lowering changes operation counts and occasionally moves an
operation between ALU/VALU.  Hard resource-order transfer is too brittle for
those rewrites, but the old exact issue cycle remains an excellent phase
teacher.  This utility maps semantic operation keys (with a relaxed
engine-independent fallback) and emits one complete hint file for the list
scheduler.  It never claims legality; the resulting schedule is validated by
the normal full-DAG verifier.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, op_key, real_tail_ops


def build_source(cycles: list[int]) -> list[kernel._Op]:
    kernel.SCHEDULE_EXACT_CYCLES = cycles
    builder = kernel.KernelBuilder()
    builder.build_kernel(10, 2047, 256, 16)
    return real_tail_ops(builder.dag_ops)


def build_target() -> list[kernel._Op]:
    kernel.SCHEDULE_EXACT_CYCLES = None
    configure_target()
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        if not hasattr(builder, "dag_ops"):
            raise
    return real_tail_ops(builder.dag_ops)


def relaxed_key(key: tuple[object, ...]) -> tuple[object, ...] | None:
    if len(key) < 5 or key[1] == "identity":
        return None
    tag = str(key[2])
    if tag.endswith("_vector"):
        tag = tag[:-7]
    return (key[1], tag, key[3], key[4])


def main() -> None:
    source_path = Path(os.environ["SOURCE"])
    source_payload = json.loads(source_path.read_text())
    raw_cycles = source_payload["cycles"]
    source_cycles = (
        [int(value) for value in raw_cycles]
        if isinstance(raw_cycles, list)
        else [int(raw_cycles[str(index)]) for index in range(len(raw_cycles))]
    )
    source_ops = build_source(source_cycles)
    if len(source_ops) != len(source_cycles):
        raise ValueError(
            f"source cycles={len(source_cycles)} but source DAG={len(source_ops)}"
        )
    target_ops = build_target()

    exact: dict[tuple[object, ...], deque[int]] = defaultdict(deque)
    relaxed: dict[tuple[object, ...], list[int]] = defaultdict(list)
    tag_round: dict[tuple[object, ...], list[int]] = defaultdict(list)
    group_round: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, op in enumerate(source_ops):
        key = op_key(op)
        exact[key].append(source_cycles[index])
        if (relaxed_key_value := relaxed_key(key)) is not None:
            relaxed[relaxed_key_value].append(source_cycles[index])
        tag_round[(op.tag, op.group, op.round)].append(source_cycles[index])
        group_round[(op.group, op.round)].append(source_cycles[index])

    def median(values: list[int]) -> int:
        ordered = sorted(values)
        return ordered[len(ordered) // 2]

    result: list[int] = []
    stats = Counter()
    for op in target_ops:
        key = op_key(op)
        if exact[key]:
            cycle = exact[key].popleft()
            stats["exact"] += 1
        elif (relaxed_values := relaxed.get(relaxed_key(key))):
            cycle = median(relaxed_values)
            stats["relaxed"] += 1
        elif (tag_values := tag_round.get((op.tag, op.group, op.round))):
            cycle = median(tag_values)
            stats["tag"] += 1
        elif (group_values := group_round.get((op.group, op.round))):
            cycle = median(group_values)
            stats["round"] += 1
        else:
            cycle = 0
            stats["fallback"] += 1
        result.append(cycle)

    output = Path(os.environ.get("OUT", "/tmp/aopt-mapped-cycle-hints.json"))
    output.write_text(
        json.dumps(
            {
                "horizon": int(os.environ.get("TARGET", "959")),
                "cycles": {str(index): cycle for index, cycle in enumerate(result)},
                "source": str(source_path),
                "stats": dict(stats),
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    print(
        f"source_ops={len(source_ops)} target_ops={len(target_ops)} "
        f"mapped={dict(stats)} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
