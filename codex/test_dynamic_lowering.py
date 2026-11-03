import unittest
from unittest.mock import patch

from codex import perf_takehome_under1000 as kernel


class DynamicLoweringTests(unittest.TestCase):
    def test_full_vector_moves_to_free_scalar_slots(self):
        graph = kernel._Graph()
        for i in range(7):
            dest = 128 + i * 8
            graph.emit("valu", ("^", dest, 0, 8), reads=range(16), writes=range(dest, dest + 8))
        graph.emit("alu", ("+", 300, 176, 176), reads=(176,), writes=(300,))
        with patch.object(kernel, "DYNAMIC_VECTOR_TO_ALU", True):
            program, cycles = kernel.KernelBuilder._schedule(graph.ops, 4, return_cycles=True)
        self.assertEqual(len(program[0]["valu"]), 6)
        self.assertEqual(len(program[0]["alu"]), 8)
        self.assertEqual(cycles[:7], [0] * 7)
        self.assertEqual(cycles[7], 1)

    def test_mad_is_not_scalarized(self):
        graph = kernel._Graph()
        for i in range(7):
            graph.emit("valu", ("multiply_add", 128 + i * 8, 0, 8, 16))
        with patch.object(kernel, "DYNAMIC_VECTOR_TO_ALU", True):
            program = kernel.KernelBuilder._schedule(graph.ops, 4)
        self.assertEqual(sum(len(b.get("valu", ())) for b in program), 7)
        self.assertFalse(any(b.get("alu") for b in program))


if __name__ == "__main__":
    unittest.main()
