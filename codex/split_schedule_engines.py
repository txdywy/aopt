"""Split a validated complete cycle assignment into per-engine order hints."""

from __future__ import annotations

import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops, validate
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
    payload = json.loads(Path(os.environ["SOURCE"]).read_text())
    raw = payload["cycles"]
    cycles = (
        [int(raw[str(index)]) for index in range(len(ops))]
        if isinstance(raw, dict)
        else [int(value) for value in raw]
    )
    if len(cycles) != len(ops):
        raise ValueError("source schedule length does not match target DAG")
    if not bool(int(os.environ.get("SKIP_VALIDATE", "0"))):
        validate(ops, cycles)
    prefix = Path(os.environ.get("OUT_PREFIX", "/tmp/aopt-engine-split"))
    horizon = max(cycles) + 1
    for engine in SLOT_LIMITS:
        order = sorted(
            (index for index, op in enumerate(ops) if op.engine == engine),
            key=lambda index: (cycles[index], index),
        )
        if not order:
            continue
        output = Path(f"{prefix}-{engine}.json")
        output.write_text(
            json.dumps(
                {
                    "engine": engine,
                    "horizon": horizon,
                    "cycles": {str(index): cycles[index] for index in order},
                    "order": order,
                }
            )
        )
        print(f"engine={engine} jobs={len(order)} output={output}", flush=True)


if __name__ == "__main__":
    main()
