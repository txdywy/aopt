"""Exact-incumbent tail experiment; never modifies the embedded baseline.

Move the final node XOR before selection, then retime the terminal hashes.
All edits assert their original instructions, and all branch table variants
receive identical semantic edits. This is deliberately not a generic pass.
"""

from copy import deepcopy
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from codex.perf_takehome_under1000 import KernelBuilder
from frozen_problem import Input, Machine, N_CORES, Tree, build_mem_image, reference_kernel2


def transform(original):
    fingerprint = hashlib.sha256(json.dumps(original, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if fingerprint != "62fd0cfa456526bcf9816eae8e8becc053f4130173a5cf4960ed902a6af34a04":
        raise ValueError("tail transform requires the verified 987-cycle program layout")
    program = deepcopy(original)

    def remove(pc, engine, slot):
        assert slot in program[pc].get(engine, []), (pc, engine, slot)
        program[pc][engine].remove(slot)

    def add(pc, engine, slot):
        program[pc].setdefault(engine, []).append(slot)

    def move(src, dst, engine, slot):
        remove(src, engine, slot)
        add(dst, engine, slot)

    # First six candidate lanes are ready after table three. Last two lanes
    # are overwritten by table four after the speculative vector XORs.
    add(974, "valu", ("^", 687, 687, 679))
    add(974, "valu", ("^", 943, 943, 679))
    entries = [pc for pc in range(987, len(program))
               if program[pc].get("flow") == [("jump", 976)]]
    assert len(entries) == 64
    for pc in entries:
        slots = program[pc]["alu"]
        count = 0
        for i, (op, dst, a, b) in enumerate(slots):
            if dst in (693, 694, 949, 950):
                assert op == "|" and a == b
                lane = dst - (687 if dst < 943 else 943)
                slots[i] = ("^", dst, a, 679 + lane)
                count += 1
        assert count == 4
    remove(976, "flow", ("vselect", 1191, 1191, 687, 943))
    add(976, "flow", ("vselect", 679, 1191, 687, 943))
    remove(977, "valu", ("^", 679, 679, 1191))

    # The last group's complete hash shifts one cycle earlier.
    for pc, slots in (
        (978, [("multiply_add", 679, 679, 127, 71)]),
        (979, [(">>", 1191, 679, 55), ("^", 679, 679, 79)]),
        (980, [("^", 679, 679, 1191)]),
        (981, [("multiply_add", 1191, 679, 135, 87), ("multiply_add", 679, 679, 95, 103)]),
        (983, [("multiply_add", 679, 679, 143, 111)]),
        (984, [(">>", 1191, 679, 63), ("^", 679, 679, 119)]),
        (985, [("^", 679, 679, 1191)]),
    ):
        for slot in slots:
            move(pc, pc-1, "valu", slot)
    for lane in range(8):
        move(982, 981, "alu", ("^", 679+lane, 679+lane, 1191+lane))
    # Free one VALU slot in cycle 978 by delaying an independent output.
    move(978, 979, "valu", ("^", 559, 559, 1071))
    move(979, 980, "store", ("vstore", 14, 559))

    # Other terminal hashes have split scalar vector operations. Rejoin them
    # only after all operands are ready; no partial-lane result is discarded.
    for lane in range(8):
        pc = 982 if lane in (1, 2, 4, 6) else 983
        remove(pc, "alu", ("^", 599+lane, 599+lane, 1111+lane))
    add(982, "valu", ("^", 599, 599, 1111))
    for lane in range(8):
        remove(983 if lane in (2, 3, 4, 5, 7) else 984,
               "alu", (">>", 1119+lane, 607+lane, 4))
        remove(983 if lane == 5 else 984,
               "alu", ("^", 607+lane, 607+lane, 12))
        remove(984 if lane == 5 else 985,
               "alu", ("^", 607+lane, 607+lane, 1119+lane))
    add(983, "valu", (">>", 1119, 607, 63))
    add(983, "valu", ("^", 607, 607, 119))
    add(984, "valu", ("^", 607, 607, 1119))

    # Retiming stores preserves each stream's pointer value at packet entry.
    for src, dst, slot in (
        (983, 982, ("vstore", 15, 583)),
        (983, 982, ("vstore", 14, 591)),
        (984, 983, ("vstore", 15, 599)),
        (984, 983, ("vstore", 5, 623)),
        (985, 984, ("vstore", 15, 615)),
        (986, 985, ("vstore", 14, 607)),
        (986, 985, ("vstore", 8, 679)),
    ):
        move(src, dst, "store", slot)
    move(983, 982, "alu", ("+", 15, 15, 4))
    move(984, 983, "alu", ("+", 15, 15, 4))
    # Keep the old terminal packet valid; it becomes unreachable.
    assert program[986]["flow"] == [("halt",)]
    add(985, "flow", ("halt",))
    limits = {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1}
    for pc, packet in enumerate(program):
        for engine, cap in limits.items():
            assert len(packet.get(engine, [])) <= cap, (pc, engine, packet)
    return program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--full32", action="store_true")
    parser.add_argument("--pattern", choices=("random", "zero", "ones", "alternating"), default="random")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("seeds must be positive")
    builder = KernelBuilder()
    builder.build_kernel(10, 2047, 256, 16)
    program = transform(builder.instrs)
    cycles = []
    for seed in range(args.seeds):
        random.seed(seed)
        tree = Tree.generate(10)
        inp = Input.generate(tree, 256, 16)
        if args.full32:
            tree.values = [random.getrandbits(32) for _ in tree.values]
            inp.values = [random.getrandbits(32) for _ in inp.values]
        if args.pattern != "random":
            values = {"zero": [0], "ones": [0xFFFFFFFF], "alternating": [0xAAAAAAAA, 0x55555555]}[args.pattern]
            tree.values = [values[i % len(values)] for i in range(len(tree.values))]
            inp.values = [values[(i+seed) % len(values)] for i in range(len(inp.values))]
        mem = build_mem_image(tree, inp)
        expected = mem.copy()
        for _ in reference_kernel2(expected):
            pass
        machine = Machine(mem, program, builder.debug_info(), n_cores=N_CORES)
        machine.run()
        start = mem[6]
        wrong = [i for i in range(256) if machine.mem[start+i] != expected[start+i]]
        assert not wrong, (seed, wrong)
        cycles.append(machine.cycle)
    print(json.dumps({"seeds": args.seeds, "full32": args.full32, "pattern": args.pattern, "cycles": sorted(set(cycles))}))


if __name__ == "__main__":
    main()
