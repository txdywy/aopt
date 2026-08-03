"""Dump the effective DAG operation metadata for cross-variant mapping."""

from __future__ import annotations

import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops


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

    def jsonable(value):
        if isinstance(value, tuple):
            return [jsonable(item) for item in value]
        if isinstance(value, list):
            return [jsonable(item) for item in value]
        return value

    payload = {
        "ops": [
            {
                "engine": op.engine,
                "slot": jsonable(op.slot),
                "tag": op.tag,
                "group": op.group,
                "round": op.round,
                "identity": jsonable(op.identity),
                "parents": {str(parent): lag for parent, lag in op.parents.items()},
            }
            for op in ops
        ]
    }
    Path(os.environ.get("OUT", "/tmp/aopt-op-metadata.json")).write_text(
        json.dumps(payload)
    )


if __name__ == "__main__":
    main()
