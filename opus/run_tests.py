"""
Standalone validation + cycle count for the opus KernelBuilder.

Imports the *frozen* simulator from ../tests and the problem definitions from
the parent directory. Does not modify anything under tests/.
"""

import os
import sys
import random

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
# Order matters: opus/ must shadow the parent's perf_takehome.py, so insert it
# LAST (it ends up at index 0 and wins the import).
sys.path.insert(0, PARENT)                         # ../problem.py
sys.path.insert(0, os.path.join(PARENT, "tests"))  # ../tests/frozen_problem.py
sys.path.insert(0, HERE)                           # opus/perf_takehome.py (priority)

from frozen_problem import (  # noqa: E402
    Machine, build_mem_image, reference_kernel2, Tree, Input, N_CORES,
)
from perf_takehome import KernelBuilder  # noqa: E402

BASELINE = 147734


def run(forest_height=10, rounds=16, batch_size=256, seed=None, trials=20):
    cyc = None
    for t in range(trials):
        if seed is not None:
            random.seed(seed + t)
        forest = Tree.generate(forest_height)
        inp = Input.generate(forest, batch_size, rounds)
        mem = build_mem_image(forest, inp)

        kb = KernelBuilder()
        kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

        machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()

        for ref_mem in reference_kernel2(mem):
            pass
        ivp = ref_mem[6]
        got = machine.mem[ivp:ivp + len(inp.values)]
        exp = ref_mem[ivp:ivp + len(inp.values)]
        assert got == exp, (
            f"MISMATCH trial {t}: first diff at "
            f"{next(i for i in range(len(got)) if got[i] != exp[i])}"
        )
        cyc = machine.cycle
    print(f"OK  h={forest_height} rounds={rounds} batch={batch_size}  "
          f"CYCLES={cyc}  speedup={BASELINE / cyc:.2f}x  instrs={len(kb.instrs)} "
          f"scratch={kb.scratch_ptr}")
    return cyc


if __name__ == "__main__":
    run()
