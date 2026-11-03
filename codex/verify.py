"""Verify the independent kernel against the frozen simulator/reference."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS))

from frozen_problem import Input, Machine, N_CORES, Tree, build_mem_image, reference_kernel2
from codex.perf_takehome_under1000 import KernelBuilder


@lru_cache(maxsize=1)
def optimized_builder() -> KernelBuilder:
    kb = KernelBuilder()
    kb.build_kernel(10, 2047, 256, 16)
    return kb


def run_once(seed: int) -> int:
    random.seed(seed)
    forest = Tree.generate(10)
    inp = Input.generate(forest, 256, 16)
    mem = build_mem_image(forest, inp)

    kb = optimized_builder()
    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()

    for reference in reference_kernel2(mem):
        pass
    values_p = reference[6]
    expected = reference[values_p : values_p + 256]
    actual = machine.mem[values_p : values_p + 256]
    if actual != expected:
        mismatch = next(i for i, (a, b) in enumerate(zip(actual, expected)) if a != b)
        raise AssertionError(
            f"seed={seed} mismatch at input {mismatch}: got {actual[mismatch]}, "
            f"expected {expected[mismatch]}"
        )
    return machine.cycle


def main() -> None:
    cycles = [run_once(seed) for seed in range(8)]
    kb = optimized_builder()
    slots = Counter(
        engine
        for bundle in kb.instrs
        for engine, engine_slots in bundle.items()
        for _ in engine_slots
    )
    print(f"cycles={cycles[0]} seeds={len(cycles)} scratch={kb.scratch_ptr}")
    print("slots=" + " ".join(f"{key}:{slots[key]}" for key in sorted(slots)))


if __name__ == "__main__":
    main()
