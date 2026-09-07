"""Validate the upload file without __file__ and against frozen tests."""
import json
from pathlib import Path
import random
import runpy
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT))
from frozen_problem import Machine, Tree, Input, build_mem_image, reference_kernel2, N_CORES

source = (ROOT / 'codex/perf_takehome_981.py').read_text()
module = types.ModuleType('perf_takehome')
exec(compile(source, '/perf_takehome.py', 'exec'), module.__dict__)
builder = module.KernelBuilder()
builder.build_kernel(10, 2047, 256, 16)
assert len(builder.instrs) == 4064538
cycles = []
for seed in range(103):
    random.seed(seed)
    tree = Tree.generate(10)
    inp = Input.generate(tree, 256, 16)
    def value(index):
        if seed < 100:
            return random.getrandbits(32)
        return (0, 0xffffffff, 0xffffffff if index % 2 else 0)[seed-100]
    tree.values = [value(i) for i in range(len(tree.values))]
    inp.values = [value(i) for i in range(len(inp.values))]
    mem = build_mem_image(tree, inp)
    machine = Machine(mem, builder.instrs, builder.debug_info(), n_cores=N_CORES)
    machine.enable_pause = machine.enable_debug = False
    machine.run()
    for reference in reference_kernel2(mem):
        pass
    start = reference[6]
    assert machine.mem[start:start+256] == reference[start:start+256], seed
    assert machine.cycle == 981, (seed, machine.cycle)
    cycles.append(machine.cycle)
print(json.dumps({'cycles': sorted(set(cycles)), 'full32_seeds':100,
                  'boundary_patterns':3, 'bytes':len(source.encode()),
                  'no_file_global': '__file__' not in module.__dict__}), flush=True)
# Reuse the already materialized, input-independent ISA list in the unchanged
# official tests, avoiding a second 4-million-packet allocation.
class VerifiedBuilder(module.KernelBuilder):
    def build_kernel(self, *shape):
        assert shape == (10,2047,256,16)
        self.instrs = builder.instrs
module.KernelBuilder = VerifiedBuilder
sys.modules['perf_takehome'] = module
sys.argv = [str(ROOT/'tests/submission_tests.py')]
runpy.run_path(sys.argv[0], run_name='__main__')
