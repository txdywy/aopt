"""Place engine-swapped critical operations in each other's freed slots.

The swapcrit DAG exchanges eight g1:r9 scalar dynamic transforms with one
g17:r9 vector hash shift.  Semantic cross-engine mapping preserves the old
operation times, which creates opposite resource overflows.  Exchanging the
two source cycle multisets instead keeps both calendars exactly feasible and
also respects the new same-engine chains.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops


def source_cycles(path: Path, tag: str, group: int, rnd: int) -> list[int]:
    payload = json.loads(path.read_text())
    result = []
    for entry in payload["entries"]:
        key = entry["id"][0]
        if len(key) >= 5 and key[2:] == [tag, group, rnd]:
            result.append(int(entry["cycle"]))
    if not result:
        raise ValueError(f"missing source operation {tag} g={group} r={rnd}")
    return sorted(result)


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

    alu_path = Path(os.environ["ALU"])
    valu_path = Path(os.environ["VALU"])
    alu_payload = json.loads(alu_path.read_text())
    valu_payload = json.loads(valu_path.read_text())
    alu_cycles = {int(i): int(c) for i, c in alu_payload["cycles"].items()}
    valu_cycles = {int(i): int(c) for i, c in valu_payload["cycles"].items()}

    freed_alu = source_cycles(
        Path(os.environ["SOURCE_ALU_ID"]),
        "dynamic_node_transform_scalar",
        1,
        9,
    )
    freed_valu = source_cycles(
        Path(os.environ["SOURCE_VALU_ID"]),
        "hash_1_shift_vector",
        17,
        9,
    )
    new_alu = [
        i
        for i, op in enumerate(ops)
        if op.engine == "alu"
        and op.tag == "hash_1_shift_scalar"
        and op.group == 17
        and op.round == 9
    ]
    new_valu = [
        i
        for i, op in enumerate(ops)
        if op.engine == "valu"
        and op.tag == "dynamic_node_transform_vector"
        and op.group == 1
        and op.round == 9
    ]
    if len(new_alu) != len(freed_alu) or len(new_valu) != len(freed_valu):
        raise ValueError(
            f"cycle swap cardinality mismatch: {len(new_alu)=} "
            f"{len(freed_alu)=} {len(new_valu)=} {len(freed_valu)=}"
        )
    for index, cycle in zip(new_alu, freed_alu):
        alu_cycles[index] = cycle
    for index, cycle in zip(new_valu, freed_valu):
        valu_cycles[index] = cycle

    prefix = Path(os.environ.get("OUT_PREFIX", "/tmp/aopt-swapcrit"))
    for engine, payload, cycles in (
        ("alu", alu_payload, alu_cycles),
        ("valu", valu_payload, valu_cycles),
    ):
        payload.update(
            {
                "engine": engine,
                "horizon": 959,
                "cycles": {str(i): cycle for i, cycle in cycles.items()},
            }
        )
        Path(f"{prefix}-{engine}.json").write_text(json.dumps(payload))
    print(
        f"critical_cycle_swap alu={freed_alu} valu={freed_valu} "
        f"output_prefix={prefix}",
        flush=True,
    )


if __name__ == "__main__":
    main()
