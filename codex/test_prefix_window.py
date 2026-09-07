"""Prefix export regression and optional conservative-memory scheduling tests."""

import importlib.util
import json
from pathlib import Path
import random
import unittest

from codex import perf_takehome_983
from codex.experiment_prefix_window import prepare_prefix
from codex.experiment_tail_speculation import Input, Machine, N_CORES, Tree, build_mem_image, reference_kernel2
from codex.export_tail_submission import render
from codex.schedule_tail_window import schedule_tail
from problem import DebugInfo


class PrefixWindowTests(unittest.TestCase):
    def test_certificate_and_portable_export(self):
        p, _ = prepare_prefix()
        certificate = json.loads(Path(__file__).with_name("prefix_schedule.json").read_text())
        expected, _ = schedule_tail(p, start=938, stop=970, horizon=31,
                                    allow_data_flow=True, allow_memory=True,
                                    exit_pc=970, certificate=certificate)
        builder = perf_takehome_983.KernelBuilder()
        builder.build_kernel(10, 2047, 256, 16)
        self.assertEqual(builder.instrs, expected)
        self.assertEqual(builder.instrs[968]["flow"], [("jump", 970)])
        source = render(prefix=True)
        self.assertEqual(source, Path(perf_takehome_983.__file__).read_text())
        scope = {"__name__": "portable_prefix_submission"}
        exec(compile(source, "/perf_takehome.py", "exec"), scope)
        scope["KernelBuilder"]().build_kernel(10, 2047, 256, 16)

    def test_exported_boundaries(self):
        builder = perf_takehome_983.KernelBuilder()
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
                    self.assertEqual(machine.cycle, 983)

    @unittest.skipUnless(importlib.util.find_spec("z3"), "optional research dependency")
    def test_aliasing_load_observes_old_memory(self):
        original = [
            {"load": [("const", 0, 0), ("const", 1, 7)]},
            {"load": [("load", 2, 0)], "store": [("store", 0, 1)]},
            {"flow": [("halt",)]},
        ]
        p, report = schedule_tail(original, start=0, stop=3, horizon=2, allow_memory=True)
        self.assertEqual(report["status"], "sat")
        machine = Machine([3], p, DebugInfo({}), n_cores=1)
        machine.run()
        self.assertEqual(machine.mem[0], 7)
        self.assertEqual(machine.cores[0].scratch[2], 3)


if __name__ == "__main__":
    unittest.main()
