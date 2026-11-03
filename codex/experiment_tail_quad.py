"""Coarsen the final speculative tables: four pair tables -> two quad tables."""

from copy import deepcopy
import json

from codex.experiment_tail_speculation import KernelBuilder, transform


def quad_transform():
    builder = KernelBuilder()
    builder.build_kernel(10, 2047, 256, 16)
    original = builder.instrs
    p = transform(original)

    def remove(pc, eng, slot):
        p[pc][eng].remove(slot)

    def add(pc, eng, slot):
        p[pc].setdefault(eng, []).append(slot)

    def move(src, dst, eng, slot):
        remove(src, eng, slot)
        add(dst, eng, slot)

    tables = [[packet for packet in original[987:]
               if packet.get("flow") == [("jump", target)]]
              for target in (970, 972, 974, 976)]
    assert all(len(table) == 64 for table in tables)
    bases = [len(p), len(p) + 4096]
    for pair, template_pc, target in ((0, 971, 972), (2, 973, 974)):
        destinations = set(range(687, 695)) | set(range(943, 951))
        for first in tables[pair]:
            for second in tables[pair+1]:
                packet = deepcopy(original[template_pc])
                packet["alu"] = [s for s in packet["alu"] if s[1] not in destinations]
                for candidate in (first, second):
                    packet["alu"].extend(s for s in candidate["alu"] if s[1] in destinations)
                packet["flow"] = [("jump", target)]
                p.append(packet)

    # Raw pair indexes are radix 64 digits; constant addresses are loaded
    # into dead old table-base registers before their new first use.
    add(965, "load", ("const", 696, 64))
    add(966, "load", ("const", 695, bases[0]))
    add(966, "load", ("const", 697, bases[1]))
    for reg, base in ((951, 695), (952, 696), (953, 697)):
        remove(967, "alu", ("+", reg, reg, base))
    add(967, "alu", ("*", 951, 951, 696))
    add(967, "alu", ("*", 953, 953, 696))
    move(968, 967, "alu", ("+", 954, 954, 942))
    add(968, "alu", ("+", 951, 951, 952))
    add(968, "alu", ("+", 953, 953, 954))
    move(968, 969, "alu", ("^", 560, 560, 1072))
    remove(969, "alu", ("+", 954, 954, 698))
    add(969, "alu", ("+", 951, 951, 695))
    add(969, "alu", ("+", 953, 953, 697))
    remove(968, "flow", ("jump_indirect", 951))
    remove(969, "flow", ("add_imm", 951, 951, 2996830211))
    remove(970, "flow", ("jump_indirect", 952))
    add(970, "flow", ("jump_indirect", 951))
    remove(974, "flow", ("jump_indirect", 954))
    remove(975, "flow", ("add_imm", 954, 954, 2996830211))
    for pc, destinations in ((969, (687, 688, 943, 944)), (975, (693, 694, 949, 950))):
        for dst in destinations:
            remove(pc, "alu", ("|", dst, 183, 183))
    move(975, 974, "valu", ("&", 1191, 679, 23))
    move(976, 975, "flow", ("vselect", 679, 1191, 687, 943))
    # Quad entries contain eight candidate copies. Make room by moving
    # independent work out of table one and revectorizing table two's join.
    for lane in range(8):
        remove(970, "alu", ("^", 567+lane, 567+lane, 7))
    add(970, "valu", ("^", 567, 567, 79))
    fillers = [(">>", 1151, 639, 4), (">>", 1153, 641, 4),
               (">>", 1158, 646, 4), ("^", 625, 625, 12)]
    for slot in fillers:
        add(970, "alu", slot)
    for pc in range(bases[0], bases[1]):
        for slot in fillers:
            remove(pc, "alu", slot)
    for pc in range(bases[1], len(p)):
        for lane in (0, 1, 3, 5, 6):
            remove(pc, "alu", ("^", 567+lane, 567+lane, 1079+lane))
        add(pc, "valu", ("^", 567, 567, 1079))
    for lane in (4, 7):
        remove(974, "alu", ("^", 567+lane, 567+lane, 1079+lane))
    remove(975, "alu", ("^", 569, 569, 1081))
    limits = {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1}
    for pc, packet in enumerate(p):
        for eng, limit in limits.items():
            assert len(packet.get(eng, [])) <= limit, (pc, eng, len(packet[eng]))
    return p, builder.debug_info()


def main():
    import argparse
    import random
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", action="store_true")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=9)
    parser.add_argument("--pattern", choices=("full32", "normal", "zero", "ones", "alternating"), default="full32")
    args = parser.parse_args()
    if args.seeds < 1 or not 1 <= args.horizon <= 10:
        parser.error("positive seeds and horizon 1..10 required")
    from codex.experiment_tail_speculation import Input, Machine, N_CORES, Tree, build_mem_image, reference_kernel2
    program, debug = quad_transform()
    if args.schedule:
        from codex.schedule_tail_window import schedule_tail
        program, report = schedule_tail(program, horizon=args.horizon)
        print(json.dumps(report), flush=True)
        if program is None:
            return
    cycles = []
    for seed in range(args.seeds):
        random.seed(seed)
        tree = Tree.generate(10)
        inp = Input.generate(tree, 256, 16)
        if args.pattern == "full32":
            tree.values = [random.getrandbits(32) for _ in tree.values]
            inp.values = [random.getrandbits(32) for _ in inp.values]
        elif args.pattern != "normal":
            words = {"zero": [0], "ones": [0xFFFFFFFF], "alternating": [0xAAAAAAAA, 0x55555555]}[args.pattern]
            tree.values = [words[i % len(words)] for i in range(len(tree.values))]
            inp.values = [words[(i+seed) % len(words)] for i in range(len(inp.values))]
        mem = build_mem_image(tree, inp)
        expected = mem.copy()
        for _ in reference_kernel2(expected):
            pass
        machine = Machine(mem, program, debug, n_cores=N_CORES)
        machine.run()
        start = mem[6]
        wrong = [i for i in range(256) if machine.mem[start+i] != expected[start+i]]
        assert not wrong, (seed, wrong)
        cycles.append(machine.cycle)
    print(json.dumps({"cycles": sorted(set(cycles)), "seeds": args.seeds, "pattern": args.pattern, "program_length": len(program)}))


if __name__ == "__main__":
    main()
