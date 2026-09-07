"""Regression checks for the 984-cycle early-selection submission."""

import json
from pathlib import Path
import random
import unittest

from codex import perf_takehome_984
from codex.experiment_tail_early import early_transform
from codex.experiment_tail_speculation import Input, Machine, N_CORES, Tree, build_mem_image, reference_kernel2
from codex.export_tail_submission import render
from codex.fuse_table_returns import accesses, LIMITS
from codex.schedule_tail_window import schedule_tail


class EarlyTailTests(unittest.TestCase):
    def test_export_and_certificate(self):
        p, _ = early_transform()
        certificate = json.loads(Path(__file__).with_name("tail_early_schedule.json").read_text())
        expected, _ = schedule_tail(p, start=975, certificate=certificate)
        builder = perf_takehome_984.KernelBuilder()
        builder.build_kernel(10, 2047, 256, 16)
        self.assertEqual(builder.instrs, expected)
        source = render(early=True)
        self.assertEqual(source, Path(perf_takehome_984.__file__).read_text())
        scope = {"__name__": "portable_early_submission"}
        exec(compile(source, "/perf_takehome.py", "exec"), scope)
        scope["KernelBuilder"]().build_kernel(10, 2047, 256, 16)

    def test_half_vector_destinations_are_disjoint(self):
        p, _ = early_transform()
        for pc in range(5408, 9504):
            writes = set()
            for engine, slots in p[pc].items():
                self.assertLessEqual(len(slots), LIMITS[engine])
                if engine == "flow":
                    continue
                for slot in slots:
                    _, destinations = accesses(engine, slot)
                    self.assertFalse(writes & destinations, (pc, engine, slot))
                    writes |= destinations

    def test_boundary_inputs_and_original_gather_regression(self):
        builder = perf_takehome_984.KernelBuilder()
        builder.build_kernel(10, 2047, 256, 16)
        for words in ([0], [0xFFFFFFFF], [0xAAAAAAAA, 0x55555555], None):
            for offset in range(3):
                with self.subTest(words=words, offset=offset):
                    random.seed(offset)
                    tree = Tree.generate(10)
                    inp = Input.generate(tree, 256, 16)
                    if words is None:
                        tree.values = [random.getrandbits(32) for _ in tree.values]
                        inp.values = [random.getrandbits(32) for _ in inp.values]
                    else:
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
                    self.assertEqual(machine.cycle, 984)


if __name__ == "__main__":
    unittest.main()
