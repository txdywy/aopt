import sys
import time
sys.path.append('.')
sys.path.append('tests')
from tests.submission_tests import do_kernel_test
import tests.frozen_problem as fp
import signal

orig_step = fp.Machine.step

def patched_step(self, instr, core):
    global last_pc
    last_pc = core.pc
    orig_step(self, instr, core)

fp.Machine.step = patched_step

def handler(signum, frame):
    print("Caught signal! PC:", last_pc)
    sys.exit(1)

signal.signal(signal.SIGALRM, handler)
signal.alarm(5)

do_kernel_test(10, 16, 256)
