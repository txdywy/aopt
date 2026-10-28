import random
import unittest

import numpy as np
import z3

from codex.search_early_boundary863 import (
    NAMES, NODE, MASK, early_inputs, wanted, solve, lsb_constraints,
    legal_point, dependency_proofs, make_manifest,
)
from codex.search_live_boundary863 import expression
from codex.test_live_boundary863 import frozen
from codex.cegis_c1_absorb863 import _instr_options
from codex.close_early_boundary863 import low_models
from tests.frozen_problem import myhash_traced


class EarlyBoundaryTests(unittest.TestCase):
    def test_early_values_recover_frozen_hash(self):
        rng = random.Random(8642823)
        for _ in range(256):
            b = rng.getrandbits(32)
            vv = early_inputs(b)
            trace = {}; result = myhash_traced(vv[5], trace, 10, 0)
            self.assertEqual(trace[(10, 0, 'hash_stage', 0)], vv[6])
            self.assertEqual(trace[(10, 0, 'hash_stage', 1)], b)
            self.assertEqual(trace[(10, 0, 'hash_stage', 3)], vv[1])
            self.assertEqual(trace[(10, 0, 'hash_stage', 4)], vv[0])
            self.assertEqual(vv[7], vv[6] >> 19)
            self.assertEqual(vv[8], vv[6] ^ 0xC761C23C)
            trace = {}; myhash_traced(result ^ (NODE ^ 0xB55A4F09), trace, 11, 0)
            self.assertEqual(trace[(11, 0, 'hash_stage', 0)], wanted(vv))

    def test_real_tree_counterexamples_are_reachable(self):
        p = dependency_proofs()
        pair = p['no_mad_fixed_right_shift']
        self.assertNotEqual(pair['first']['target'] >> 31, pair['second']['target'] >> 31)
        self.assertEqual([v >> 31 for v in pair['first']['values']],
                         [v >> 31 for v in pair['second']['values']])
        self.assertEqual(len(p['no_new_downward_bit_transfer']['full32_facts']), 19)
        for b in (0, MASK, 1 << 31, (1 << 19) | 1):
            self.assertTrue(legal_point(b)['frozen_reference_checked'])

    def test_non_T_positive_control_is_synthesized_and_frozen(self):
        row = dict(order=['shr16', '^', 'mad'], wiring=[[5], [6, 9], [10, 11, 12]])
        result = solve(row, 10, control=True)
        self.assertEqual(result['status'], 'proved_equivalent')
        self.assertFalse(result['prefix_mod4_guard_used'])
        cc = result['constants']
        kk = [[z3.BitVecVal(k, 32) for k in pair] for pair in cc]
        guard = z3.SolverFor('QF_BV'); guard.add(z3.Not(z3.And(*lsb_constraints(row, kk))))
        self.assertEqual(guard.check(), z3.unsat)
        rng = random.Random(8642824)
        for _ in range(16):
            bs = np.array([rng.getrandbits(32) for _ in range(8)], dtype=np.uint64)
            with np.errstate(over='ignore'):
                values = early_inputs(bs); expected = wanted(values, True)
            self.assertEqual(frozen(row, cc, values)[-1], expected.tolist())

    def test_nine_word_random_programs_match_frozen_and_symbolic(self):
        rows, _ = make_manifest()
        rng = random.Random(8642825)
        for _ in range(200):
            row = rng.choice(rows)
            cc = [[rng.getrandbits(32), rng.getrandbits(32)] for _ in row['order']]
            bs = [rng.getrandbits(32) for _ in range(8)]
            with np.errstate(over='ignore'):
                values = early_inputs(np.array(bs, dtype=np.uint64))
                numeric = expression(row, cc, values).tolist()
            actual = frozen(row, cc, values)[-1]
            self.assertEqual(actual, numeric)
            self.assertEqual(actual, [expression(row, cc, early_inputs(b)) for b in bs])
            kk = [[z3.BitVecVal(k, 32) for k in pair] for pair in cc]
            self.assertEqual(actual, [z3.simplify(expression(row, kk, early_inputs(z3.BitVecVal(b, 32)))).as_long()
                                     for b in bs])

    def test_packed_truth_tables_agree_with_independent_z3_gate(self):
        rng = random.Random(8642833)
        for order in (['mad', 'shr16', 'mad'], ['mad', 'shr16', '^'],
                      ['&', 'shr16', 'mad']):
            options = [_instr_options(8, order[0]), [(9,)], _instr_options(10, order[2])]
            for case in range(60):
                row = dict(order=order, wiring=[list(rng.choice(o)) for o in options])
                kk = [z3.BitVecs(f'truth_{case}_{i}a truth_{case}_{i}b', 32) for i in range(3)]
                s = z3.SolverFor('QF_BV'); s.add(*lsb_constraints(row, kk))
                self.assertEqual(bool(low_models(row)), s.check() == z3.sat)


if __name__ == '__main__':
    unittest.main()
