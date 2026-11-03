from types import SimpleNamespace
import unittest
from unittest.mock import patch

from codex.audit_dead_values import unconsumed_definitions
from codex.perf_takehome_under1000 import _Graph
from codex import perf_takehome_under1000 as kernel


def op(reads=(), writes=()):
    return SimpleNamespace(reads=reads, writes=writes)


class DeadValuesTests(unittest.TestCase):
    def test_fused_transform_elision_requires_every_lane(self):
        for lanes, fused, expected in (
            (((0, 1, 2, 3), (4, 5, 6, 7)), True, 1),
            (((0, 1, 2, 3),), True, 0),
            (((0, 1, 2, 3), (4, 5, 6, 7)), False, 0),
        ):
            settings = dict(PAIRED_DIRECT_BRANCH_LOOKUPS={3: lanes},
                            DIRECT_BRANCH_ROUND=3, DIRECT_BRANCH_DEPTH=3,
                            DIRECT_BRANCH_ARITY=4, CHAINED_PAIRED_DIRECT_BASE=True,
                            RECYCLE_PAIRED_DIRECT_STRIDE=True,
                            COMPACT_PAIRED_DIRECT_RETURN=True,
                            FUSED_DIRECT_BRANCH_XOR=fused, SCHEDULE_EXACT_CYCLES=[])
            counts = []
            with self.subTest(lanes=lanes, fused=fused), patch.multiple(kernel, **settings):
                for enabled in (False, True):
                    with patch.object(kernel, "OMIT_FUSED_DIRECT_NODE_TRANSFORM", enabled):
                        builder = kernel.KernelBuilder()
                        with self.assertRaisesRegex(ValueError, "exact cycle schedule does not match DAG"):
                            builder.build_kernel(10, 2047, 256, 16)
                        counts.append(sum(op.tag == "dynamic_node_transform_vector"
                                          for op in builder.dag_ops))
                self.assertEqual(counts[0] - counts[1], expected)

    def test_scalar_cache_rejects_incompatible_setup(self):
        for overrides in (
            {"OMIT_LEVEL4_DIFF_SETUP": False},
            {"DIRECT_BRANCH_DEPTH": 5},
            {"REVERSED_RELOCATED_TREE": True},
            {"INDEPENDENT_ROOT_CACHE": False, "DIRECT_MIRROR_PATH": False},
        ):
            settings = {"SCALAR_LEVEL4_TABLE_CACHE": True,
                        "OMIT_LEVEL4_DIFF_SETUP": True, **overrides}
            with self.subTest(overrides=overrides), patch.multiple(kernel, **settings):
                with self.assertRaisesRegex(ValueError, "scalar level4 cache requires"):
                    kernel.KernelBuilder().build_kernel(10, 2047, 256, 16)

    def test_baseline_still_requires_differences(self):
        with patch.object(kernel, "OMIT_LEVEL4_DIFF_SETUP", True):
            with self.assertRaisesRegex(ValueError, "omitted scratch definition consumed"):
                kernel.KernelBuilder().build_kernel(10, 2047, 256, 16)

    def test_omitted_definition_rejects_in_place_read(self):
        graph = _Graph()
        graph.omitted_definitions.add(1)
        with self.assertRaisesRegex(ValueError, "omitted scratch definition"):
            graph.emit("alu", ("+", 1, 1, 2), reads=(1, 2), writes=(1,))
        self.assertEqual(graph.omitted_definitions, {1})
        graph.emit("load", ("const", 1, 7), writes=(1,))
        graph.emit("alu", ("+", 1, 1, 2), reads=(1, 2), writes=(1,))

    def test_overwrite_does_not_consume_previous_definition(self):
        self.assertEqual(unconsumed_definitions([
            op(writes=(1,)), op(writes=(1,)), op(reads=(1,))]), [0])

    def test_in_place_read_consumes_previous_definition(self):
        self.assertEqual(unconsumed_definitions([
            op(writes=(1,)), op(reads=(1,), writes=(1,)), op(reads=(1,))]), [])

    def test_partially_consumed_vector_is_not_dead(self):
        self.assertEqual(unconsumed_definitions([
            op(writes=(1, 2)), op(reads=(2,))]), [])


if __name__ == "__main__":
    unittest.main()
