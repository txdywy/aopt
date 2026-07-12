import sys
import time
sys.path.append('.')
sys.path.append('tests')
from tests.submission_tests import do_kernel_test, kernel_builder
from tests.frozen_problem import Tree, Input, build_mem_image
from perf_takehome import KernelBuilder

forest = Tree.generate(10)
inp = Input.generate(forest, 256, 16)
mem = build_mem_image(forest, inp)
print("1. mem built")
kb = KernelBuilder()
print("2. kb created")
kb.build_kernel(10, len(forest.values), len(inp.indices), 16)
print("3. build_kernel finished. ops:", len(kb.instrs))
import tests.frozen_problem as fp
machine = fp.Machine(mem, kb.instrs, kb.debug_info(), n_cores=16)
print("4. Machine created")
machine.enable_pause = False
machine.enable_debug = False
import signal
def handler(signum, frame):
    print("Caught signal! PC:", [c.pc for c in machine.cores])
    sys.exit(1)
signal.signal(signal.SIGALRM, handler)
signal.alarm(5)
print("5. machine running")
machine.run()
print("6. machine finished")
