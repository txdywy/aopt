"""Print matching operation regions from source and target kernel variants."""

from __future__ import annotations

import os

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops


def build() -> list[kernel._Op]:
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
    source_config = {
        key.removeprefix("SOURCE_CONFIG_"): value
        for key, value in os.environ.items()
        if key.startswith("SOURCE_CONFIG_")
    }
    target_config = {key: os.environ.get(key) for key in source_config}
    for key, value in source_config.items():
        os.environ[key] = value
    source = build()
    for key, value in target_config.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    target = build()

    filters = {
        tuple(map(int, item.split(":")))
        for item in os.environ["GROUP_ROUNDS"].split(",")
        if item
    }
    radius = int(os.environ.get("RADIUS", "0"))
    for label, ops in (("source", source), ("target", target)):
        selected = [
            i
            for i, op in enumerate(ops)
            if (op.group, op.round) in filters
            and (
                not os.environ.get("TAGS")
                or op.tag in os.environ["TAGS"].split(",")
            )
        ]
        expanded = {
            i
            for center in selected
            for i in range(max(0, center - radius), min(len(ops), center + radius + 1))
        }
        print(f"[{label}] count={len(ops)} selected={len(selected)}")
        for i in sorted(expanded if radius else selected):
            op = ops[i]
            print(
                f"{i:5d} {op.engine:5s} {op.slot[0]:16s} "
                f"g={op.group:2d} r={op.round:2d} tag={op.tag:30s} "
                f"reads={op.reads} writes={op.writes} slot={op.slot}"
            )


if __name__ == "__main__":
    main()
