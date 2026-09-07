"""Count executed ISA work, excluding unreachable tables, on the frozen VM.

Bounds apply to the measured lowering, not every equivalent implementation.
The profiler observes machine state and never feeds it into kernel creation.
"""

import argparse
from collections import Counter
import json
import math
import random

from codex import perf_takehome_983
from codex.experiment_tail_speculation import Input, Machine, N_CORES, Tree, build_mem_image, reference_kernel2

LIMITS = {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1}


class ProfileMachine(Machine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executed_slots = Counter()
        self.load_classes = Counter()
        self.rows = []

    def step(self, instr, core):
        row = {engine: len(slots) for engine, slots in instr.items()}
        self.executed_slots.update(row)
        self.rows.append(row)
        for slot in instr.get("load", []):
            op, dst, src, *extra = slot
            if op == "const":
                category = "constant"
            else:
                address = core.scratch[src + extra[0]] if op == "load_offset" else core.scratch[src]
                if 7 <= address < 7+2047:
                    depth = (address-7+1).bit_length()-1
                    # For vload this labels the first word only; its eight
                    # words may cross a tree-level boundary.
                    category = f"tree_start_depth_{depth}:{op}"
                elif address < 7:
                    category = f"header:{op}"
                else:
                    category = f"other_memory:{op}"
            self.load_classes[category] += 1
        super().step(instr, core)


def gap(counts, target):
    return {
        "unchanged_work_engine_bounds": {e: math.ceil(counts[e]/cap) for e, cap in LIMITS.items()},
        "relaxed_arithmetic_bound": math.ceil((counts["alu"] + 8*counts["valu"])/60),
        "fixed_lowering_slot_overflow": {e: max(0, counts[e]-target*cap) for e, cap in LIMITS.items()},
        "arithmetic_lane_work_overflow": max(0, counts["alu"] + 8*counts["valu"]-60*target),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=899)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--full32", action="store_true")
    args = parser.parse_args()
    if args.target < 1 or args.seeds < 1:
        parser.error("target and seeds must be positive")
    builder = perf_takehome_983.KernelBuilder()
    builder.build_kernel(10, 2047, 256, 16)
    reports = []
    for seed in range(args.seeds):
        random.seed(seed)
        tree = Tree.generate(10)
        inp = Input.generate(tree, 256, 16)
        if args.full32:
            tree.values = [random.getrandbits(32) for _ in tree.values]
            inp.values = [random.getrandbits(32) for _ in inp.values]
        mem = build_mem_image(tree, inp)
        expected = mem.copy()
        for _ in reference_kernel2(expected):
            pass
        machine = ProfileMachine(mem, builder.instrs, builder.debug_info(), n_cores=N_CORES)
        machine.run()
        start = mem[6]
        assert machine.mem[start:start+256] == expected[start:start+256]
        assert len(machine.rows) == machine.cycle
        windows = []
        for lo in range(0, machine.cycle, 32):
            rows = machine.rows[lo:lo+32]
            counts = Counter()
            for row in rows:
                counts.update(row)
            windows.append({"cycles": [lo, lo+len(rows)], "slots": dict(counts),
                            "saturated_engines": [e for e, cap in LIMITS.items()
                                                  if counts[e] == len(rows)*cap]})
        reports.append({"cycles": machine.cycle, "slots": dict(machine.executed_slots),
                        "load_classes": dict(sorted(machine.load_classes.items())),
                        "windows": windows, **gap(machine.executed_slots, args.target)})
    assert all(report == reports[0] for report in reports), "Input-dependent work; do not use a single profile"
    print(json.dumps({"target": args.target, "seeds": args.seeds, "full32": args.full32,
                      "verified_all_outputs": True, **reports[0]}, indent=2))


if __name__ == "__main__":
    main()
