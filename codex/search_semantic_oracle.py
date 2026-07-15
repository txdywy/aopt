"""Use a known global schedule as a soft semantic phase oracle."""

from __future__ import annotations

from collections import defaultdict, deque
import json
import os
from pathlib import Path
import statistics

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops, validate


def key(op: kernel._Op) -> tuple[object, ...]:
    opcode = op.slot[0] if op.slot else None
    return (op.engine, opcode, op.tag, op.group, op.round)


def main() -> None:
    source = json.loads(Path(os.environ["SOURCE"]).read_text())["cycles"]
    kernel.SCHEDULE_EXACT_CYCLES = source
    source_builder = kernel.KernelBuilder()
    source_builder.build_kernel(10, 2047, 256, 16)
    source_ops = real_tail_ops(source_builder.dag_ops)

    by_key: dict[tuple[object, ...], deque[int]] = defaultdict(deque)
    phase_values: dict[tuple[int, int], list[int]] = defaultdict(list)
    setup_values: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, op in enumerate(source_ops):
        by_key[key(op)].append(source[i])
        if op.group is not None and op.group >= 0 and op.round is not None:
            phase_values[op.group, op.round].append(source[i])
        else:
            setup_values[op.engine, op.tag].append(source[i])
    phase_median = {
        phase: round(statistics.median(values))
        for phase, values in phase_values.items()
    }
    setup_median = {
        phase: round(statistics.median(values))
        for phase, values in setup_values.items()
    }

    kernel.SCHEDULE_EXACT_CYCLES = None
    configure_target()
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        if not hasattr(builder, "dag_ops"):
            raise
    ops = real_tail_ops(builder.dag_ops)
    _, fallback = builder._schedule(builder.dag_ops, 4, return_cycles=True)
    fallback_span = max(fallback) + 1
    source_span = max(source) + 1

    oracle = []
    matched = 0
    phased = 0
    for i, op in enumerate(ops):
        matches = by_key[key(op)]
        if matches:
            oracle.append(matches.popleft())
            matched += 1
        elif (
            op.group is not None
            and op.group >= 0
            and op.round is not None
            and (op.group, op.round) in phase_median
        ):
            oracle.append(phase_median[op.group, op.round])
            phased += 1
        elif (op.engine, op.tag) in setup_median:
            oracle.append(setup_median[op.engine, op.tag])
            phased += 1
        else:
            oracle.append(
                round(fallback[i] * (source_span - 1) / max(1, fallback_span - 1))
            )

    cycle_weights = tuple(
        int(value)
        for value in os.environ.get("CYCLE_WEIGHTS", "1,2,4,8,16,32,64").split(",")
    )
    height_weights = tuple(
        int(value)
        for value in os.environ.get("HEIGHT_WEIGHTS", "1,2,4,8,16,32").split(",")
    )
    best: tuple[int, int, int, int, list[int]] | None = None
    for cycle_weight in cycle_weights:
        external = [-cycle_weight * value for value in oracle]
        for height_weight in height_weights:
            for policy in range(4):
                _, cycles = builder._schedule(
                    builder.dag_ops,
                    policy,
                    return_cycles=True,
                    external_scores=external,
                    height_weight=height_weight,
                    tie_scores=[-value for value in oracle],
                )
                validate(ops, cycles)
                candidate = (
                    max(cycles) + 1,
                    cycle_weight,
                    height_weight,
                    policy,
                    cycles,
                )
                if best is None or candidate[:4] < best[:4]:
                    best = candidate
                    print(
                        f"score={candidate[0]} cycle_weight={cycle_weight} "
                        f"height_weight={height_weight} policy={policy}",
                        flush=True,
                    )
    assert best is not None
    output = Path(os.environ.get("OUT", "/tmp/aopt-semantic-oracle.json"))
    output.write_text(
        json.dumps(
            {
                "makespan": best[0],
                "cycle_weight": best[1],
                "height_weight": best[2],
                "policy": best[3],
                "matched": matched,
                "phased": phased,
                "cycles": best[4],
            }
        )
    )
    print(
        f"best={best[0]} matched={matched} phased={phased} output={output}"
    )


if __name__ == "__main__":
    main()
