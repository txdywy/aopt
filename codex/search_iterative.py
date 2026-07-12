"""Iterative resource-constrained critical-path scheduling experiments."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from problem import SLOT_LIMITS
import codex.perf_takehome as kernel
from codex.analyze_schedule import configure


def resource_height(ops, cycles, *, include_anti: bool, reverse_ties: bool) -> list[int]:
    children: list[dict[int, int]] = [dict() for _ in ops]
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            if include_anti or lag:
                children[parent][child] = max(children[parent].get(child, 0), lag)

    for engine, capacity in SLOT_LIMITS.items():
        if engine == "debug":
            continue
        nodes = [i for i, op in enumerate(ops) if op.engine == engine]
        nodes.sort(key=lambda i: (cycles[i], -i if reverse_ties else i))
        for before, after in zip(nodes, nodes[capacity:]):
            if cycles[before] < cycles[after]:
                children[before][after] = max(children[before].get(after, 0), 1)

    height = [0] * len(ops)
    order = sorted(
        range(len(ops)),
        key=lambda i: (cycles[i], -i if reverse_ties else i),
        reverse=True,
    )
    for node in order:
        if children[node]:
            height[node] = max(lag + height[child] for child, lag in children[node].items())
    return height


def main() -> None:
    configure()
    kernel.SCHEDULE_POLICIES = (16,)
    kernel.BACKWARD_POLICIES = ()
    builder = kernel.KernelBuilder()
    builder.build_kernel(10, 2047, 256, 16)
    ops = builder.dag_ops
    best = (len(builder.instrs), 16, True, False, 0, 0)
    print("start", best, flush=True)

    for initial in (0, 4, 16, 17, 18, 19, 20, 80, 81, 86, 92, 104):
        for include_anti in (False, True):
            for reverse_ties in (False, True):
                _, cycles = builder._schedule(ops, initial, return_cycles=True)
                seen = set()
                for iteration in range(12):
                    score0 = max(cycles) + 1
                    signature = tuple(cycles)
                    if signature in seen:
                        break
                    seen.add(signature)
                    heights = resource_height(
                        ops,
                        cycles,
                        include_anti=include_anti,
                        reverse_ties=reverse_ties,
                    )
                    for height_weight in (0, 1, 2, 4):
                        schedule, candidate = builder._schedule(
                            ops,
                            initial,
                            return_cycles=True,
                            external_scores=heights,
                            height_weight=height_weight,
                        )
                        row = (
                            len(schedule), initial, include_anti, reverse_ties,
                            iteration + 1, height_weight,
                        )
                        if row < best:
                            best = row
                            print("best", best, flush=True)
                    _, cycles = builder._schedule(
                        ops,
                        initial,
                        return_cycles=True,
                        external_scores=heights,
                        height_weight=best[-1] if best[1:4] == (initial, include_anti, reverse_ties) else 0,
                    )
                    if max(cycles) + 1 == score0:
                        break
    print("BEST", best)


if __name__ == "__main__":
    main()
