import os, sys
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("tests"))

from frozen_problem import Machine, build_mem_image, reference_kernel2, Tree, Input, N_CORES
from perf_takehome import KernelBuilder

print("Generating tree...")
forest = Tree.generate(10)
inp = Input.generate(forest, 256, 16)
mem = build_mem_image(forest, inp)

print("Building kernel...")
kb = KernelBuilder()
kb.build_kernel(forest.height, len(forest.values), len(inp.indices), 16)

machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
machine.enable_pause = False
machine.enable_debug = False

print("Running machine...")
machine.run()

print("Verifying correctness...")
for ref_mem in reference_kernel2(mem):
    pass

inp_values_p = ref_mem[6]
output_match = (machine.mem[inp_values_p : inp_values_p + len(inp.values)]
                == ref_mem[inp_values_p : inp_values_p + len(inp.values)])

print(f"Correctness: {output_match}")
print(f"CYCLES: {machine.cycle}")
assert output_match, "Incorrect output values"
