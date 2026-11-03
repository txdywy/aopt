"""Regression checks for the verified 986-cycle artifact and its lowering."""

from copy import deepcopy
from pathlib import Path
import unittest

from codex import perf_takehome_986
from codex.experiment_tail_speculation import KernelBuilder, transform
from codex.export_tail_submission import render


class TailSpeculationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = KernelBuilder()
        cls.baseline.build_kernel(10, 2047, 256, 16)

    def test_transform_does_not_mutate_incumbent(self):
        before = deepcopy(self.baseline.instrs)
        transformed = transform(self.baseline.instrs)
        self.assertEqual(before, self.baseline.instrs)
        self.assertEqual(transformed[985]["flow"], [("halt",)])

    def test_changed_layout_is_rejected(self):
        changed = deepcopy(self.baseline.instrs)
        changed[0] = {}
        with self.assertRaisesRegex(ValueError, "verified 987-cycle"):
            transform(changed)

    def test_exported_program_is_exactly_the_transformation(self):
        builder = perf_takehome_986.KernelBuilder()
        builder.build_kernel(10, 2047, 256, 16)
        self.assertEqual(builder.instrs, transform(self.baseline.instrs))

    def test_export_is_reproducible_and_needs_no_file_global(self):
        source = render()
        self.assertEqual(source, Path(perf_takehome_986.__file__).read_text())
        namespace = {"__name__": "portable_submission"}
        exec(compile(source, "/perf_takehome.py", "exec"), namespace)
        builder = namespace["KernelBuilder"]()
        builder.build_kernel(10, 2047, 256, 16)
        self.assertEqual(builder.instrs, transform(self.baseline.instrs))


if __name__ == "__main__":
    unittest.main()
