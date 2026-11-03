"""End-to-end regression for partial tables plus saved-path gathers."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import unittest


class TripleTableTest(unittest.TestCase):
    def test_mixed_round_full32_outputs(self):
        self.check_candidate([], 1011)

    def test_padded_pair_covers_remaining_lanes(self):
        self.check_candidate(['--cover-remainder', '--scalar-level4-cache',
                              '--feedback-rounds', '4', '--feedback-percent', '10'], 987)

    def test_vector_triple_packing(self):
        self.check_candidate(['--cover-remainder', '--scalar-level4-cache', '--vector-index',
                              '--feedback-rounds', '4', '--feedback-percent', '10'], 986)

    def test_early_jump_address_and_return_fusion(self):
        self.check_candidate(['--cover-remainder', '--scalar-level4-cache', '--vector-index',
                              '--isolate-jumps', '--fuse-returns', '--jump-address-lead', '2'], 1014)

    def check_candidate(self, extra, expected):
        root = Path(__file__).resolve().parents[1]
        args = shlex.split(json.loads((root/'codex/table_981_result.json').read_text())['command'])[1:]
        args[args.index('--arity')+1] = '3'
        args.remove('--vector-index')
        args[args.index('--seeds')+1] = '3'
        args += extra
        proc = subprocess.run([sys.executable]+args, cwd=root, capture_output=True,
                              text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(result['cycles'], [expected]*3)
        self.assertEqual(result['arity'], 3)


if __name__ == '__main__':
    unittest.main()
