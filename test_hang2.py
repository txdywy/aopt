import sys
import time
sys.path.append('.')
sys.path.append('tests')
from tests.submission_tests import do_kernel_test
import tests.frozen_problem as fp
import signal

orig_alu = fp.Machine.alu

def patched_alu(self, core, op, dest, a1, a2):
    v1 = core.scratch[a1]
    v2 = core.scratch[a2]
    if op == "<<" and v2 > 1000:
        print(f"HANG: << {v2}")
        sys.exit(1)
    orig_alu(self, core, op, dest, a1, a2)

fp.Machine.alu = patched_alu

def handler(signum, frame):
    print("Caught signal!")
    sys.exit(1)

signal.signal(signal.SIGALRM, handler)
signal.alarm(5)

do_kernel_test(10, 16, 256)
