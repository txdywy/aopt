import sys
import time
sys.path.append('.')
sys.path.append('tests')
from tests.submission_tests import kernel_builder
kb = kernel_builder(10, 16, 256, 16)
print("Instrs length:", len(kb.instrs))
print("First 10 instrs:", kb.instrs[:10])
print("Last 10 instrs:", kb.instrs[-10:])
