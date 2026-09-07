"""Earlier last selection via a dead vector and half-vector speculation."""

from codex.experiment_tail_quad import quad_transform


def early_transform():
    p, debug = quad_transform()
    aliases = {971: [971] + list(range(1312, 5408)),
               973: [973] + list(range(5408, 9504))}

    def remove(pc, engine, slot):
        for at in aliases.get(pc, [pc]):
            p[at][engine].remove(slot)

    def add(pc, engine, slot):
        for at in aliases.get(pc, [pc]):
            p[at].setdefault(engine, []).append(slot)

    def move(src, dst, engine, slot):
        remove(src, engine, slot)
        add(dst, engine, slot)

    # 431..438 are a completed output vector, dead throughout this suffix.
    # A cached leaf mux becomes one MAD with a precomputed node difference.
    add(942, "valu", ("-", 431, 263, 295))
    add(961, "valu", ("multiply_add", 431, 687, 431, 295))
    remove(962, "flow", ("vselect", 1191, 687, 263, 295))
    remove(963, "flow", ("vselect", 1191, 943, 711, 1191))
    add(962, "flow", ("vselect", 431, 943, 711, 431))
    remove(964, "flow", ("vselect", 1191, 703, 719, 1191))
    add(964, "flow", ("vselect", 679, 703, 719, 431))

    # Two extra speculative XORs shorten select -> node-XOR by one packet.
    # Rebalance independent joins to ALU to make the two VALU slots available.
    remove(963, "valu", ("^", 639, 639, 1151))
    for lane in range(8):
        add(962, "alu", ("^", 639+lane, 639+lane, 1151+lane))
    remove(963, "valu", ("^", 583, 583, 1095))
    for lane in range(8):
        add(963, "alu", ("^", 583+lane, 583+lane, 1095+lane))
    for lane in (0, 3):
        move(963, 964, "alu", ("^", 623+lane, 623+lane, 12))
    add(963, "valu", ("^", 719, 719, 679))
    add(963, "valu", ("^", 431, 431, 679))
    for lane in range(8):
        remove(965, "alu", ("^", 679+lane, 679+lane, 1191+lane))

    move(965, 964, "valu", ("^", 551, 551, 1063))
    remove(966, "valu", ("^", 575, 575, 1087))
    for lane in range(8):
        add(965, "alu", ("^", 575+lane, 575+lane, 1087+lane))
    move(966, 965, "valu", ("multiply_add", 679, 679, 127, 71))
    move(967, 966, "valu", (">>", 1191, 679, 55))
    for lane in range(8):
        remove(967, "alu", ("^", 679+lane, 679+lane, 7))
    add(966, "valu", ("^", 679, 679, 79))
    move(968, 967, "valu", ("^", 679, 679, 1191))
    for slot in (("multiply_add", 1191, 679, 135, 87), ("multiply_add", 679, 679, 95, 103)):
        move(969, 968, "valu", slot)
    move(970, 969, "valu", ("^", 679, 679, 1191))
    move(971, 970, "valu", ("multiply_add", 679, 679, 143, 111))
    move(972, 971, "valu", (">>", 1191, 679, 63))
    move(973, 972, "valu", ("^", 679, 679, 1191))

    # Table two now reads the completed q value at packet entry. Its four
    # remaining lanes XOR q while copying. Shifted vector XORs produce the
    # first four lanes, writing dead padding instead of table-written lanes.
    # No destination overlaps: 683..690 / 939..946 versus 691..694 / 947..950.
    for pc in range(5408, 9504):
        slots = p[pc]["alu"]
        for i, slot in enumerate(slots):
            op, dst, a, b = slot
            if dst in (*range(691, 695), *range(947, 951)):
                assert op == "|" and a == b
                lane = dst - (687 if dst < 943 else 943)
                slots[i] = ("^", dst, a, 679+lane)
    remove(974, "valu", ("^", 687, 687, 679))
    remove(974, "valu", ("^", 943, 943, 679))
    add(973, "valu", ("^", 683, 683, 675))
    add(973, "valu", ("^", 939, 939, 675))
    move(974, 973, "valu", ("&", 1191, 679, 23))
    move(975, 974, "flow", ("vselect", 679, 1191, 687, 943))
    # The last streamed output otherwise waits one extra cycle for the
    # previous store's pointer increment. Its address is shape-static;
    # table dispatch register 951 is dead after packet 970.
    add(974, "load", ("const", 951, 7 + 2047 + 256 + (615 - 431)))
    remove(984, "store", ("vstore", 15, 615))
    add(984, "store", ("vstore", 951, 615))
    # This already-completed output's pointer is ready after table two.
    # Store it alongside the earlier selection instead of occupying the
    # otherwise fully saturated terminal store window.
    move(975, 974, "store", ("vstore", 15, 503))
    move(975, 974, "valu", ("^", 639, 639, 119))
    for lane in (1, 3, 4, 5, 6):
        move(975, 974, "alu", (">>", 1151+lane, 639+lane, 4))
    move(976, 974, "valu", ("^", 631, 631, 1143))
    # A second output chain gates the shared store pointer. Advance its
    # preceding hash too, using the slots vacated by the first chain.
    move(971, 970, "valu", ("multiply_add", 559, 559, 127, 71))
    # The last two input lanes originally completed in packet 970. Their
    # gather addresses are ready after 957; pull loads into idle LOAD slots
    # and then complete the node XORs before the advanced hash reads them.
    for lane in (5, 7):
        move(969, 962, "load", ("load_offset", 1071, 1071, lane))
        move(970, 969, "alu", ("^", 559+lane, 559+lane, 1071+lane))
    move(972, 971, "valu", (">>", 1071, 559, 55))
    for lane in range(8):
        remove(972, "alu", ("^", 559+lane, 559+lane, 7))
    add(971, "valu", ("^", 559, 559, 79))
    move(973, 972, "valu", ("^", 559, 559, 1071))
    for slot in (("multiply_add", 1071, 559, 135, 87), ("multiply_add", 559, 559, 95, 103)):
        move(974, 973, "valu", slot)
    move(975, 974, "valu", ("^", 559, 559, 1071))
    limits = {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1}
    for pc, packet in enumerate(p):
        for engine, limit in limits.items():
            assert len(packet.get(engine, [])) <= limit, (pc, engine, len(packet[engine]))
    return p, debug


def main():
    import argparse
    import json
    import random
    from codex.experiment_tail_speculation import Input, Machine, N_CORES, Tree, build_mem_image, reference_kernel2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", action="store_true")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--pattern", choices=("full32", "normal", "zero", "ones", "alternating"), default="full32")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("seeds must be positive")
    p, debug = early_transform()
    if args.schedule:
        from codex.schedule_tail_window import schedule_tail
        p, report = schedule_tail(p, start=975, horizon=9)
        print(json.dumps(report), flush=True)
        if p is None:
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
        machine = Machine(mem, p, debug, n_cores=N_CORES)
        machine.run()
        start = mem[6]
        wrong = [i for i in range(256) if machine.mem[start+i] != expected[start+i]]
        assert not wrong, (seed, wrong)
        cycles.append(machine.cycle)
    print(json.dumps({"cycles": sorted(set(cycles)), "seeds": args.seeds, "pattern": args.pattern}))


if __name__ == "__main__":
    main()
