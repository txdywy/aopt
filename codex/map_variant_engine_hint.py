"""Transfer a resource order between structural kernel variants.

Run once with ``MODE=dump`` under the source variant, then with ``MODE=map``
under the target variant.  Operations are identified by the same semantic
key and construction-order occurrence used by the full schedule mapper, so
virtual-register renaming does not destroy correspondence.
"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, op_key, real_tail_ops
from problem import SLOT_LIMITS


def freeze(value: object) -> object:
    if isinstance(value, list):
        return tuple(freeze(item) for item in value)
    return value


def cross_engine_key(key: tuple[object, ...]) -> tuple[object, ...] | None:
    """Match an operation whose implementation engine changed.

    Structural retiming may replace eight scalar operations with one vector
    operation (or the reverse).  Engine and the optional ``_vector`` tag
    suffix are implementation details; opcode/group/round identify the same
    semantic step closely enough to transfer its old issue time.
    """
    if len(key) < 5 or key[1] == "identity":
        return None
    tag = str(key[2])
    if tag.endswith("_vector"):
        tag = tag[:-7]
    return (key[1], tag, key[3], key[4])


def operation_ids(ops: list[kernel._Op]) -> list[tuple[tuple[object, ...], int]]:
    occurrences: Counter[tuple[object, ...]] = Counter()
    result = []
    for op in ops:
        key = op_key(op)
        result.append((key, occurrences[key]))
        occurrences[key] += 1
    return result


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
    ids = operation_ids(ops)
    engine = os.environ.get("ENGINE", "flow")
    mode = os.environ.get("MODE", "dump")
    source = json.loads(Path(os.environ["SOURCE"]).read_text())
    output = Path(os.environ.get("OUT", "/tmp/aopt-mapped-engine.json"))

    if mode == "dump":
        cycles = {int(i): int(c) for i, c in source["cycles"].items()}
        order = [
            int(i)
            for i in source.get(
                "order", sorted(cycles, key=lambda i: (cycles[i], i))
            )
        ]
        expected = {i for i, op in enumerate(ops) if op.engine == engine}
        if set(order) != expected:
            raise ValueError("source hint does not match source variant")
        payload = {
            "engine": engine,
            "entries": [
                {"id": [list(ids[i][0]), ids[i][1]], "cycle": cycles[i]}
                for i in order
            ],
        }
    elif mode == "map":
        source_entries = source["entries"]
        source_rank = {
            (freeze(entry["id"][0]), int(entry["id"][1])): rank
            for rank, entry in enumerate(source_entries)
        }
        source_cycle = {
            (freeze(entry["id"][0]), int(entry["id"][1])): int(entry["cycle"])
            for entry in source_entries
        }
        target = [i for i, op in enumerate(ops) if op.engine == engine]
        target_by_id = {ids[i]: i for i in target}
        missing = [i for i in target if ids[i] not in source_rank]
        allow_missing = bool(int(os.environ.get("ALLOW_MISSING", "0")))
        if missing and not allow_missing:
            raise ValueError(
                f"target has {len(missing)} unmatched {engine} operations"
            )
        relaxed_tag_cycles: dict[tuple[object, ...], list[int]] = {}
        relaxed_round_cycles: dict[tuple[object, ...], list[int]] = {}
        cross_cycles: dict[tuple[object, ...], list[int]] = {}
        for raw_cross_source in os.environ.get("CROSS_SOURCES", "").split(","):
            if not raw_cross_source:
                continue
            cross_payload = json.loads(Path(raw_cross_source).read_text())
            for entry in cross_payload["entries"]:
                cross_key = cross_engine_key(freeze(entry["id"][0]))
                if cross_key is not None:
                    cross_cycles.setdefault(cross_key, []).append(
                        int(entry["cycle"])
                    )
        for entry in source_entries:
            key = freeze(entry["id"][0])
            cycle = int(entry["cycle"])
            if len(key) >= 5:
                relaxed_tag_cycles.setdefault(
                    (key[0], key[2], key[3], key[4]), []
                ).append(cycle)
                relaxed_round_cycles.setdefault(
                    (key[0], key[3], key[4]), []
                ).append(cycle)

        def median(values: list[int]) -> int:
            ordered = sorted(values)
            return ordered[len(ordered) // 2]

        mapped_cycle: dict[int, int] = {}
        matched_indices: list[int] = []
        exact = 0
        cross_fallback = 0
        tag_fallback = 0
        round_fallback = 0
        for i in target:
            if ids[i] in source_cycle:
                mapped_cycle[i] = source_cycle[ids[i]]
                matched_indices.append(i)
                exact += 1
                continue
            key = op_key(ops[i])
            cross_values = cross_cycles.get(cross_engine_key(key))
            tag_values = (
                relaxed_tag_cycles.get((key[0], key[2], key[3], key[4]))
                if len(key) >= 5
                else None
            )
            round_values = (
                relaxed_round_cycles.get((key[0], key[3], key[4]))
                if len(key) >= 5
                else None
            )
            if cross_values:
                mapped_cycle[i] = median(cross_values)
                cross_fallback += 1
            elif tag_values:
                mapped_cycle[i] = median(tag_values)
                tag_fallback += 1
            elif round_values:
                mapped_cycle[i] = median(round_values)
                round_fallback += 1
            else:
                mapped_cycle[i] = 0
        order = sorted(target, key=lambda i: (mapped_cycle[i], i))
        payload = {
            "engine": engine,
            "horizon": int(os.environ.get("TARGET", "959")),
            "cycles": {str(i): mapped_cycle[i] for i in target},
            "order": order,
            "mapped": len(target),
            "exact": exact,
            "cross_fallback": cross_fallback,
            "tag_fallback": tag_fallback,
            "round_fallback": round_fallback,
            "source_entries": len(source_entries),
            # Consumers can preserve the proven resource order for unchanged
            # operations while leaving structurally new operations free to be
            # inserted by CP-SAT.
            "matched_indices": matched_indices,
            "resource_edges": [
                [target_by_id[parent_id], target_by_id[child_id]]
                for parent_entry, child_entry in zip(
                    source_entries,
                    source_entries[SLOT_LIMITS[engine]:],
                )
                if (parent_id := (
                    freeze(parent_entry["id"][0]),
                    int(parent_entry["id"][1]),
                )) in target_by_id
                and (child_id := (
                    freeze(child_entry["id"][0]),
                    int(child_entry["id"][1]),
                )) in target_by_id
            ],
        }
    else:
        raise ValueError("MODE must be dump or map")

    output.write_text(json.dumps(payload))
    print(
        f"mode={mode} engine={engine} jobs={len(payload.get('order', payload.get('entries', [])))} "
        f"output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
