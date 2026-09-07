"""Small frozen-machine correctness checks for the experimental postpass."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from frozen_problem import DebugInfo, Machine
from codex.fuse_table_returns import fuse_returns


class FusionTests(unittest.TestCase):
    def program(self):
        return [
            {"flow": [("jump", 3)]},
            {"alu": [("+", 4, 2, 3)], "flow": [("jump_indirect", 5)]},
            {"flow": [("halt",)]},
            {"alu": [("+", 0, 2, 3)], "flow": [("jump", 1)]},
            {"alu": [("^", 1, 0, 4)], "flow": [("jump", 2)]},
        ]

    def execute(self, program):
        machine = Machine([0] * 16, program, DebugInfo({}))
        machine.cores[0].scratch[2:6] = [17, 39, 0, 4]
        machine.run()
        return machine.cycle, machine.cores[0].scratch

    def test_fuses_and_preserves_execution(self):
        original = self.program()
        changed = copy.deepcopy(original)
        report = fuse_returns(changed, 3)
        before_cycles, before = self.execute(original)
        after_cycles, after = self.execute(changed)
        self.assertEqual(report["fused_sites"], [1])
        self.assertEqual(before, after)
        self.assertEqual(after_cycles, before_cycles - 1)

    def test_read_after_write_rejected(self):
        program = self.program()
        program[1]["alu"] = [("+", 4, 0, 3)]
        self.assertEqual(fuse_returns(program, 3)["rejected"], {"register_dependency": 1})

    def test_capacity_rejected(self):
        program = self.program()
        program[1]["alu"] = [("+", 20 + i, 2, 3) for i in range(12)]
        self.assertEqual(fuse_returns(program, 3)["rejected"], {"capacity": 1})

    def test_one_unsafe_variant_rejects_entire_site(self):
        program = self.program()
        program.append({"alu": [("+", 2, 6, 7)], "flow": [("jump", 1)]})
        before = copy.deepcopy(program)
        self.assertEqual(fuse_returns(program, 3)["fused_sites"], [])
        self.assertEqual(program, before)


if __name__ == "__main__":
    unittest.main()
