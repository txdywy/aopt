"""Project a full cycle list onto one engine of the current DAG."""

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
    source = Path(os.environ["SOURCE"])
    payload = json.loads(source.read_text())
    raw_cycles = payload.get("cycles")
    if not isinstance(raw_cycles, list):
        raise ValueError("SOURCE must contain a full cycles list")
    if len(raw_cycles) != len(ops):
        raise ValueError(f"source cycles={len(raw_cycles)} target ops={len(ops)}")
    engine = os.environ["ENGINE"]
    selected = [i for i, op in enumerate(ops) if op.engine == engine]
    result = {
        "engine": engine,
        "horizon": max(int(raw_cycles[i]) for i in selected) + 1,
        "cycles": {str(i): int(raw_cycles[i]) for i in selected},
        "order": sorted(selected, key=lambda i: (int(raw_cycles[i]), i)),
    }
    output = Path(os.environ.get("OUT", f".search959/{engine}-hint.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result))
    print(
        f"source={source} engine={engine} jobs={len(selected)} "
        f"span={result['horizon']} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
