import sys
sys.path.append('.')
sys.path.append('tests')
from tests.submission_tests import do_kernel_test
import tests.frozen_problem as fp
import signal

orig_alu = fp.Machine.alu

def patched_alu(self, core, op, dest, a1, a2):
    global last_op, last_v1, last_v2
    last_op = op
    last_v1 = core.scratch[a1]
    last_v2 = core.scratch[a2]
    orig_alu(self, core, op, dest, a1, a2)

fp.Machine.alu = patched_alu

def handler(signum, frame):
    print(f"Caught signal! Last op: {last_op}, v1: {last_v1}, v2: {last_v2}")
    sys.exit(1)

signal.signal(signal.SIGALRM, handler)
signal.alarm(5)

do_kernel_test(10, 16, 256)
