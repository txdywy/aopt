"""Bounded, deterministic scalar-lowering screen; never omit exit-packet work."""
from copy import deepcopy
import json
import random
import sys
from pathlib import Path

from codex.perf_takehome_983 import KernelBuilder
from codex.schedule_tail_window import schedule_tail


def main():
    builder = KernelBuilder()
    builder.build_kernel(10, 2047, 256, 16)
    original = builder.instrs
    candidates = [(pc, si) for pc in range(938, 969)
                  for si, slot in enumerate(original[pc].get('valu', []))
                  if len(slot) == 4 and slot[0] in ('^', '&', '|', '+', '-', '>>', '<<', '*')]
    rng = random.Random(983)
    choices = [candidates[:4], candidates[-4:]]
    choices += [rng.sample(candidates, 4) for _ in range(6)]
    for selected in choices:
        program = deepcopy(original)
        del program[968]['flow']
        for pc, si in sorted(selected, reverse=True):
            op, dest, left, right = program[pc]['valu'].pop(si)
            program[pc].setdefault('alu', []).extend(
                (op, dest + lane, left + lane, right + lane) for lane in range(8))
        result, report = schedule_tail(program, start=938, stop=969, horizon=30,
                                      allow_data_flow=True, allow_memory=True,
                                      exit_pc=970, timeout_ms=3000)
        summary = {k: v for k, v in report.items()
                   if k not in ('issue_cycles', 'critical_chain', 'store_earliest')}
        summary['lowered'] = selected
        if result is not None:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
            from frozen_problem import Machine, Tree, Input, build_mem_image, reference_kernel2, N_CORES
            scores = []
            for seed in range(23):
                random.seed(seed)
                tree = Tree.generate(10)
                inp = Input.generate(tree, 256, 16)
                def value(i):
                    return random.getrandbits(32) if seed < 20 else (0, 0xffffffff, 0xffffffff if i % 2 else 0)[seed - 20]
                tree.values = [value(i) for i in range(len(tree.values))]
                inp.values = [value(i) for i in range(256)]
                mem = build_mem_image(tree, inp)
                machine = Machine(mem, result, builder.debug_info(), n_cores=N_CORES)
                machine.enable_pause = machine.enable_debug = False
                machine.run()
                for ref in reference_kernel2(mem):
                    pass
                assert machine.mem[ref[6]:ref[6]+256] == ref[ref[6]:ref[6]+256], seed
                scores.append(machine.cycle)
            summary.update(cycles=scores, json_bytes=len(json.dumps(result)), certificate=report)
        print(json.dumps(summary), flush=True)
        if result is not None:
            break


if __name__ == '__main__':
    main()
