"""The same SIMD group can now dispatch paired tables in both traversals."""

from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from codex import perf_takehome_under1000 as kernel


class MultiRoundTests(unittest.TestCase):
    def test_three_saved_bits_use_complete_mirror_for_final_table(self):
        root = Path(__file__).resolve().parents[1]
        command = [sys.executable, str(root / "codex/experiment_paired_base.py"),
                   "--arity", "4", "--group-ids", "14", "--round", "15",
                   "--keep-final", "--compact-return", "--recycle-stride", "--relax-jump",
                   "--no-packet-repair", "--horner-index", "--prioritize-copies", "--parallel",
                   "--seeds", "30", "--pattern", "full32", "--teacher-weight", "1024"]
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["cycles"], [994] * 30)

    def test_schedule_preflight_stops_before_assembly(self):
        with patch.object(kernel, "SCHEDULE_PREFLIGHT_ONLY", True):
            builder = kernel.KernelBuilder()
            with self.assertRaisesRegex(ValueError, "schedule preflight only; tables not assembled"):
                builder.build_kernel(10, 2047, 256, 16)
        self.assertEqual(builder.preflight_packet_count, len(builder.instrs))
        self.assertGreater(builder.preflight_packet_count, 0)

    def test_elided_entries_reject_nonadjacent_copy(self):
        root = Path(__file__).resolve().parents[1]
        command = [sys.executable, str(root / "codex/experiment_paired_base.py"),
                   "--arity", "2", "--group-ids", "3,4,5,7,9,11,12,13,15,16,18,19,20,24,29,31",
                   "--also-final-group-ids", "0,1,3,4,5,6,7,9,10,11,12,17,23,28,29",
                   "--round", "4", "--keep-final", "--recycle-stride", "--relax-jump",
                   "--no-packet-repair", "--vector-index", "--horner-index",
                   "--seeds", "1", "--pattern", "full32", "--teacher-weight", "1024"]
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("single-packet direct return requires an adjacent copy packet", result.stderr)
        self.assertNotIn("IndexError", result.stderr)
        # The same full configuration must execute when ready copies receive
        # ALU priority, rather than merely suppressing the layout assertion.
        command += ["--prioritize-copies", "--compact-return", "--omit-level4-diffs"]
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["cycles"], [1060])
        self.assertEqual(report["instructions"], 33129)

    def test_graph_has_separate_round_records(self):
        settings = {
            "PAIRED_DIRECT_BRANCH_LOOKUPS": {3: ((0, 1, 2, 3), (4, 5, 6, 7))},
            "PAIRED_DIRECT_EXTRA_ROUNDS": {3: (15,)},
            "DIRECT_BRANCH_ROUND": 4,
            "DIRECT_BRANCH_ARITY": 4,
            "CHAINED_PAIRED_DIRECT_BASE": True,
            "COMPACT_PAIRED_DIRECT_RETURN": True,
            "RECYCLE_PAIRED_DIRECT_STRIDE": True,
            "SCHEDULE_EXACT_CYCLES": [],
        }
        with ExitStack() as stack:
            for name, value in settings.items():
                stack.enter_context(patch.object(kernel, name, value))
            builder = kernel.KernelBuilder()
            with self.assertRaisesRegex(ValueError, "exact cycle schedule does not match DAG"):
                builder.build_kernel(10, 2047, 256, 16)
            jumps = [op for op in builder.dag_ops if op.tag == "paired_direct_branch_jump"]
            self.assertEqual(Counter((op.group, op.round) for op in jumps), {(3, 4): 2, (3, 15): 2})
            with patch.object(kernel, "HORNER_PAIRED_DIRECT_INDEX", True):
                horner = kernel.KernelBuilder()
                with self.assertRaisesRegex(ValueError, "exact cycle schedule does not match DAG"):
                    horner.build_kernel(10, 2047, 256, 16)
            self.assertEqual(sum(op.engine == "valu" for op in horner.dag_ops),
                             sum(op.engine == "valu" for op in builder.dag_ops)-1)
            self.assertEqual(sum("horner" in op.tag for op in horner.dag_ops), 3)

    def test_overlapping_group_runs_both_rounds(self):
        root = Path(__file__).resolve().parents[1]
        command = [sys.executable, str(root / "codex/experiment_paired_base.py"),
                   "--arity", "4", "--group-ids", "3", "--also-final-group-ids", "3",
                   "--round", "4", "--seeds", "3", "--pattern", "full32",
                   "--teacher-weight", "1024", "--keep-final", "--compact-return",
                   "--recycle-stride", "--relax-jump", "--no-packet-repair", "--vector-index"]
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["cycles"], [990]*3)
        self.assertEqual(report["instructions"], 263459)


if __name__ == "__main__":
    unittest.main()
