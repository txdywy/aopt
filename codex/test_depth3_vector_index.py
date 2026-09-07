"""Execution checks for radix-eight adjacent-lane packing."""
import json
from pathlib import Path
import subprocess
import sys
import unittest


class Depth3VectorIndexTests(unittest.TestCase):
    def test_quad_and_pair_boundary_inputs(self):
        root = Path(__file__).resolve().parents[1]
        for arity in (2, 4):
            with self.subTest(arity=arity):
                command = [sys.executable, str(root / 'codex/experiment_paired_base.py'),
                           '--round', '14', '--group-ids', '2,8,13,15,19,22,24,26,30',
                           '--arity', str(arity), '--keep-final', '--compact-return',
                           '--recycle-stride', '--relax-jump', '--no-packet-repair',
                           '--prioritize-copies', '--parallel', '--restore-hash-count', '88',
                           '--dispatch-lead', '32', '--teacher-weight', '1024',
                           '--last-group-lead', '8', '--policy', '3', '--seeds', '3',
                           '--pattern', 'boundaries', '--omit-fused-node-transform',
                           '--vector-depth3-index']
                result = subprocess.run(command, cwd=root, capture_output=True,
                                        text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(len(report['cycles']), 3)
                self.assertLessEqual(report['scratch'], 1536)
                if arity == 4:
                    self.assertEqual(report['operations']['alu'], 10864)
                    self.assertEqual(report['cycles'], [994]*3)


if __name__ == '__main__':
    unittest.main()
