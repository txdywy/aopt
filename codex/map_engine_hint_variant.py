"""Map per-engine projected calendars across a small lowering variant.

Research utility: builds a source and target semantic DAG without replaying
the source schedule, then transfers issue cycles by ``op_key``.  Operations
whose engine changes are intentionally left unconstrained; the strict model
can place those few retimed jobs itself.
"""
from __future__ import annotations

from collections import defaultdict, deque
import importlib
import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, op_key, real_tail_ops


def build(env_overrides: dict[str, str]):
    old = {key: os.environ.get(key) for key in env_overrides}
    try:
        for key, value in env_overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        importlib.reload(kernel)
        configure_target()
        kernel.SCHEDULE_EXACT_CYCLES = None
        builder = kernel.KernelBuilder()
        try:
            builder.build_kernel(10, 2047, 256, 16)
        except (AssertionError, StopIteration, ValueError):
            if not hasattr(builder, "dag_ops"):
                raise
        return real_tail_ops(builder.dag_ops)
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> None:
    source_restore = os.environ.get("SOURCE_VECTOR_RESTORE_LAYOUT_SET", "1")
    source_join = os.environ.get("SOURCE_SCALAR_HASH5_JOIN", "")
    source_gather = os.environ.get("SOURCE_GATHER_LEVEL4", "")
    source_final_join = os.environ.get("SOURCE_SCALAR_FINAL_JOIN", "")
    target_restore = os.environ.get("TARGET_VECTOR_RESTORE_LAYOUT_SET", "1,15")
    target_join = os.environ.get("TARGET_SCALAR_HASH5_JOIN", "31:14")
    target_gather = os.environ.get("TARGET_GATHER_LEVEL4", "")
    target_final_join = os.environ.get("TARGET_SCALAR_FINAL_JOIN")
    if target_final_join is None:
        target_final_join = os.environ.get("SCALAR_FINAL_JOIN", "")
    source = build({"VECTOR_RESTORE_LAYOUT_SET": source_restore,
                    "SCALAR_HASH5_JOIN": source_join,
                    "GATHER_LEVEL4": source_gather,
                    "SCALAR_FINAL_JOIN": source_final_join})
    target = build({"VECTOR_RESTORE_LAYOUT_SET": target_restore,
                    "SCALAR_HASH5_JOIN": target_join,
                    "GATHER_LEVEL4": target_gather,
                    "SCALAR_FINAL_JOIN": target_final_join})
    by_key: dict[tuple, deque[int]] = defaultdict(deque)
    source_index_by_engine: dict[str, set[int]] = defaultdict(set)
    for index, op in enumerate(source):
        by_key[op_key(op)].append(index)
        source_index_by_engine[op.engine].add(index)
    payloads = {
        str(path): json.loads(Path(path).read_text())
        for path in os.environ["SOURCE_HINTS"].split(",") if path
    }
    by_engine = {str(data["engine"]): data for data in payloads.values()}
    source_cycles = {
        engine: {int(i): int(c) for i, c in data["cycles"].items()}
        for engine, data in by_engine.items()
    }
    mapped: dict[str, dict[str, int]] = defaultdict(dict)
    stats = defaultdict(int)
    fallback_cycle = int(os.environ.get("FALLBACK_CYCLE", "480"))
    for target_index, op in enumerate(target):
        key = op_key(op)
        matches = by_key.get(key)
        if not matches:
            stats["missing"] += 1
            mapped[op.engine][str(target_index)] = fallback_cycle
            continue
        source_index = matches.popleft()
        cycle = source_cycles.get(op.engine, {}).get(source_index)
        if cycle is None:
            stats["engine_moved"] += 1
            mapped[op.engine][str(target_index)] = fallback_cycle
            continue
        mapped[op.engine][str(target_index)] = cycle
        stats["mapped"] += 1
    out_dir = Path(os.environ.get("OUT_DIR", "/tmp/mapped-variant"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for engine, cycles in mapped.items():
        (out_dir / f"{engine}.json").write_text(
            json.dumps({"engine": engine, "horizon": 959, "cycles": cycles})
        )
    print(f"source={len(source)} target={len(target)} stats={dict(stats)} out={out_dir}")


if __name__ == "__main__":
    main()
