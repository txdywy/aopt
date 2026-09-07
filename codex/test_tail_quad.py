"""Check the quad-table artifact without requiring the optional SMT solver."""

from copy import deepcopy
import json
from pathlib import Path
import random
import unittest

from codex import perf_takehome_985
from codex.experiment_tail_quad import quad_transform
from codex.experiment_tail_speculation import Input, Machine, N_CORES, Tree, build_mem_image, reference_kernel2
from codex.export_tail_submission import render
from codex.schedule_tail_window import schedule_tail


class QuadTailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.program, cls.debug = quad_transform()
        cls.certificate = json.loads(Path(__file__).with_name("tail_quad_schedule.json").read_text())

    def test_certificate_reproduces_exported_program(self):
        expected, _ = schedule_tail(self.program, certificate=self.certificate)
        builder = perf_takehome_985.KernelBuilder()
        builder.build_kernel(10, 2047, 256, 16)
        self.assertEqual(expected, builder.instrs)

    def test_bad_graph_hash_is_rejected(self):
        bad = deepcopy(self.certificate)
        bad["graph_sha256"] = "incorrect"
        with self.assertRaisesRegex(ValueError, "does not match"):
            schedule_tail(self.program, certificate=bad)

    def test_invalid_schedule_is_rejected(self):
        bad = deepcopy(self.certificate)
        bad["issue_cycles"] = [0] * len(bad["issue_cycles"])
        with self.assertRaises(ValueError):
            schedule_tail(self.program, certificate=bad)

    def test_portable_export_is_reproducible(self):
        source = render(quad=True)
        self.assertEqual(source, Path(perf_takehome_985.__file__).read_text())
        scope = {"__name__": "portable_quad_submission"}
        exec(compile(source, "/perf_takehome.py", "exec"), scope)
        builder = scope["KernelBuilder"]()
        builder.build_kernel(10, 2047, 256, 16)
        self.assertEqual(len(builder.instrs), 9504)

    def test_boundary_inputs_on_exported_frozen_machine(self):
        builder = perf_takehome_985.KernelBuilder()
        builder.build_kernel(10, 2047, 256, 16)
        for words in ([0], [0xFFFFFFFF], [0xAAAAAAAA, 0x55555555]):
            for offset in range(3):
                with self.subTest(words=words, offset=offset):
                    random.seed(offset)
                    tree = Tree.generate(10)
                    inp = Input.generate(tree, 256, 16)
                    tree.values = [words[i % len(words)] for i in range(len(tree.values))]
                    inp.values = [words[(i+offset) % len(words)] for i in range(256)]
                    mem = build_mem_image(tree, inp)
                    expected = mem.copy()
                    for _ in reference_kernel2(expected):
                        pass
                    machine = Machine(mem, builder.instrs, builder.debug_info(), n_cores=N_CORES)
                    machine.run()
                    start = mem[6]
                    self.assertEqual(machine.mem[start:start+256], expected[start:start+256])
                    self.assertEqual(machine.cycle, 985)


if __name__ == "__main__":
    unittest.main()
