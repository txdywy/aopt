"""Try a conservative whole-packet prefix compression before the quad tables."""

import argparse
import json
import random

from codex import perf_takehome_984
from codex.experiment_tail_speculation import Input, Machine, N_CORES, Tree, build_mem_image, reference_kernel2
from codex.schedule_tail_window import schedule_tail


def prepare_prefix(start=938, scalarize=4):
    builder = perf_takehome_984.KernelBuilder()
    builder.build_kernel(10, 2047, 256, 16)
    remaining = scalarize
    for pc in reversed(range(start, 970)):
        for slot in reversed(list(builder.instrs[pc].get("valu", []))):
            if remaining and len(slot) == 4 and slot[0] != "vbroadcast":
                op, dst, a, b = slot
                builder.instrs[pc]["valu"].remove(slot)
                builder.instrs[pc].setdefault("alu", []).extend((op, dst+i, a+i, b+i) for i in range(8))
                remaining -= 1
    if remaining or scalarize < 0:
        raise ValueError("invalid scalarization count")
    return builder.instrs, builder.debug_info()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=938)
    parser.add_argument("--save", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--timeout-ms", type=int, default=45000)
    parser.add_argument("--scalarize", type=int, default=0,
                        help="Lower this many late basic vector operations to eight scalar slots")
    parser.add_argument("--certificate")
    parser.add_argument("--pattern", choices=("full32", "normal", "zero", "ones", "alternating"), default="full32")
    args = parser.parse_args()
    if not 900 <= args.start < 969 or args.save not in (0, 1, 2) or args.seeds < 1:
        parser.error("start 900..968, save 0..2, and positive seeds required")
    program, debug = prepare_prefix(args.start, args.scalarize)
    certificate = None
    if args.certificate:
        from pathlib import Path
        certificate = json.loads(Path(args.certificate).read_text())
    program, report = schedule_tail(program, start=args.start, stop=970,
                                    horizon=970-args.start-args.save,
                                    timeout_ms=args.timeout_ms,
                                    allow_data_flow=True, allow_memory=True, exit_pc=970,
                                    certificate=certificate)
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
        assert machine.mem[start:start+256] == expected[start:start+256], seed
        cycles.append(machine.cycle)
    print(json.dumps({"cycles": sorted(set(cycles)), "seeds": args.seeds, "pattern": args.pattern}))


if __name__ == "__main__":
    main()
