"""Temporary lane trace for the overlapped address experiment."""

from __future__ import annotations

from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from frozen_problem import Input, Machine, Tree, build_mem_image, reference_kernel2
import codex.perf_takehome as kernel
from codex.analyze_schedule import configure
from codex.search_offsets import color


OFFSETS = (4, 0, 2, 10, 4, 17, 0, 18, 5, 17, 0, 2, 19, 16, 0, 18,
           13, 15, 23, 24, 5, 13, 18, 5, 8, 24, 13, 12, 11, 24, 3, 23)


def main() -> None:
    configure()
    kernel.FIRST_CACHE_SET = frozenset((7, 12, 15, 18, 19, 22, 25, 29, 31))
    kernel.FINAL_CACHE_SET = frozenset(range(9))
    kernel.HYBRID_MADD_PAIRS = 8
    kernel.OVERLAP_DEEP_ADDRESS = True
    kernel.OVERLAP_SHALLOW_ADDRESS = True
    kernel.INDEPENDENT_ROOT_CACHE = True
    kernel.TAIL_EMISSION_MODE = "full_offset"
    kernel.FULL_ROUND_OFFSETS = OFFSETS
    kernel.WORKSPACE_ASSIGNMENT = color(OFFSETS)
    kernel.SECOND_WORKSPACE_FIXED = 8
    kernel.SCHEDULE_POLICIES = (36,)
    kernel.BACKWARD_POLICIES = ()
    builder = kernel.KernelBuilder()
    builder.build_kernel(10, 2047, 256, 16)
    _, cycles = builder._schedule(builder.dag_ops, 36, return_cycles=True)

    group = 9
    interesting = {}
    for i, op in enumerate(builder.dag_ops):
        if op.group == group and op.tag in {
            "mirror_build_3", "depth4_address_double", "depth4_address_affine",
            "depth4_address_parity", "next_address_affine",
            "next_address_parity", "tree_gather", "hash_5_join",
        }:
            interesting.setdefault(cycles[i], []).append(op.tag)
    print(interesting)

    random.seed(0)
    tree = Tree.generate(10)
    inp = Input.generate(tree, 256, 16)
    mem = build_mem_image(tree, inp)
    trace = {}
    ref = mem.copy()
    for _ in reference_kernel2(ref, trace):
        pass
    print("expected", [(r, trace[(r, group * 8, "idx")], trace[(r, group * 8, "hashed_val")]) for r in range(16)])

    machine = Machine(mem, builder.instrs, builder.debug_info())
    mirror = builder.scratch[f"mirror_{group}"]
    value = builder.scratch[f"value_{group}"]
    core = machine.cores[0]
    for pc, bundle in enumerate(builder.instrs):
        machine.step(bundle, core)
        if pc in interesting:
            print(pc, interesting[pc], "m", core.scratch[mirror], "v", core.scratch[value])


if __name__ == "__main__":
    main()
