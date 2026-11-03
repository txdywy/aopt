"""Validate the standalone submission with the unchanged frozen simulator.

The official contract compares final values. Consumed tree/header/index memory
may hold temporaries, so it is intentionally not compared to an older kernel.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from frozen_problem import (  # noqa: E402
    Input, Machine, N_CORES, SCRATCH_SIZE, SLOT_LIMITS, Tree,
    VLEN, build_mem_image, reference_kernel2,
)
from perf_takehome import EXPECTED_CYCLES, KernelBuilder, PROGRAM_SHA256  # noqa: E402


def scratch_accesses(engine, slot):
    """Include every architectural lane, including unused vector padding."""
    op, *args = slot
    words = lambda base: set(range(base, base + VLEN))
    if engine == "alu":
        dest, a, b = args
        return {a, b}, {dest}
    if engine == "valu":
        dest, *sources = args
        reads = {sources[0]} if op == "vbroadcast" else set().union(
            *(words(source) for source in sources))
        return reads, words(dest)
    if engine == "load":
        dest, source, *extra = args
        if op == "const":
            return set(), {dest}
        if op == "load_offset":
            return {source + extra[0]}, {dest + extra[0]}
        if op in ("load", "vload"):
            return {source}, words(dest) if op == "vload" else {dest}
    if engine == "store":
        addr, source = args
        return {addr} | (words(source) if op == "vstore" else {source}), set()
    if engine == "flow":
        if op in ("select", "vselect"):
            dest, cond, a, b = args
            if op == "vselect":
                return words(cond) | words(a) | words(b), words(dest)
            return {cond, a, b}, {dest}
        if op == "add_imm":
            dest, source, _ = args
            return {source}, {dest}
        if op == "jump_indirect":
            return {args[0]}, set()
        if op in ("halt", "jump"):
            return set(), set()
    raise ValueError((engine, slot))


def validate_isa(program):
    counts = Counter()
    maximum = -1
    for pc, packet in enumerate(program):
        assert packet, (pc, "empty packet")
        written = set()
        for engine, slots in packet.items():
            assert engine in ("alu", "valu", "load", "store", "flow")
            assert len(slots) <= SLOT_LIMITS[engine], (pc, engine)
            counts[engine] += len(slots)
            for slot in slots:
                if engine == "flow" and slot[0] == "jump":
                    assert 0 <= slot[1] < len(program)
                reads, writes = scratch_accesses(engine, slot)
                assert all(0 <= address < SCRATCH_SIZE for address in reads | writes), (pc, slot)
                assert not written & writes, (pc, slot, "scratch write collision")
                written |= writes
                maximum = max(maximum, max(reads | writes, default=-1))
    return dict(counts), maximum


class BoundedMemory(list):
    """Reject negative addresses that a normal Python list would accept."""
    def __getitem__(self, address):
        if not isinstance(address, slice):
            assert 0 <= address < len(self), ("memory read", address, len(self))
        return super().__getitem__(address)

    def __setitem__(self, address, value):
        assert isinstance(address, int) and 0 <= address < len(self), (
            "memory write", address, len(self))
        super().__setitem__(address, value)


def execute(builder, program, seed, pattern):
    rng = random.Random(seed)
    tree = Tree(10, [rng.getrandbits(30) for _ in range(2047)])
    inp = Input([0] * 256, [rng.getrandbits(30) for _ in range(256)], 16)
    if pattern == "full32":
        tree.values = [rng.getrandbits(32) for _ in tree.values]
        inp.values = [rng.getrandbits(32) for _ in inp.values]
    elif pattern != "normal":
        words = {
            "zero": (0,), "ones": (0xFFFFFFFF,),
            "alternating": (0xAAAAAAAA, 0x55555555),
            "high_bit": (0, 0x80000000), "low_bit": (0, 1),
            "extremes": (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF),
        }[pattern]
        tree.values = [words[i % len(words)] for i in range(2047)]
        inp.values = [words[(i + seed) % len(words)] for i in range(256)]
    memory = build_mem_image(tree, inp)
    machine = Machine(memory, program, builder.debug_info(), n_cores=N_CORES)
    machine.mem = BoundedMemory(machine.mem)
    machine.enable_pause = machine.enable_debug = False
    machine.run()
    for expected in reference_kernel2(memory.copy()):
        pass
    base = expected[6]
    assert machine.mem[base:base + 256] == expected[base:base + 256], (pattern, seed)
    assert machine.cycle == EXPECTED_CYCLES, (pattern, seed, machine.cycle)
    return machine.cycle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=256)
    args = parser.parse_args()
    if args.seeds < 1: parser.error("seeds must be positive")
    builder = KernelBuilder()
    builder.build_kernel(10, 2047, 256, 16)
    counts, maximum = validate_isa(builder.instrs)
    assert maximum < builder.scratch_ptr <= SCRATCH_SIZE
    raw = json.dumps(builder.instrs, separators=(",", ":")).encode()
    assert hashlib.sha256(raw).hexdigest() == PROGRAM_SHA256
    # Browser transport changes tuples to lists. Execute that representation.
    program = json.loads(raw)
    cases = [(seed, "full32") for seed in range(args.seeds)]
    cases += [(seed, "normal") for seed in range(8)]
    cases += [(i, pattern) for i, pattern in enumerate(
        ("zero", "ones", "alternating", "high_bit", "low_bit", "extremes"))]
    scores = [execute(builder, program, seed, pattern) for seed, pattern in cases]
    report = {"cycles": sorted(set(scores)), "cases": len(cases),
              "full32_seeds": args.seeds, "normal_seeds": 8, "boundary_patterns": 6,
              "scratch_words": builder.scratch_ptr, "packets": len(builder.instrs),
              "max_scratch_address": maximum, "operations": counts,
              "total_operations": sum(counts.values()),
              "json_round_trip": True, "runtime_memory_bounds_checked": True,
              "instruction_json_bytes": len(json.dumps(builder.instrs).encode()),
              "source_bytes": (ROOT / "perf_takehome.py").stat().st_size,
              "program_sha256": PROGRAM_SHA256}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
