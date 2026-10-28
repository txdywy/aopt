import random
import unittest

import numpy as np
import z3

from codex.search_joint_live_boundary863 import (
    NAMES, NODE, inputs, wanted, make_manifest, parity_models, controls, expression,
)
from codex.search_early_boundary863 import early_inputs, legal_point
from codex.test_live_boundary863 import frozen
from codex.build_joint_live_boundary863 import CONTROL, verify


class JointLiveBoundaryTests(unittest.TestCase):
    def test_context_excludes_e_and_matches_frozen_reachable_target(self):
        self.assertEqual(NAMES, ('z', 'u', 'v', 'b', 'prior', 'a', 'a_shr19', 'a_xor_c1'))
        rng = random.Random(8642837)
        for _ in range(32):
            b = rng.getrandbits(32)
            self.assertEqual(inputs(b), early_inputs(b)[1:])
            self.assertEqual(wanted(inputs(b), NODE), legal_point(b)['target'])

    def test_random_four_op_programs_match_frozen_and_symbolic(self):
        rows, stats = make_manifest(4)
        self.assertEqual(len(stats), 111)
        self.assertEqual(len(rows), 444)
        rng = random.Random(8642838)
        for _ in range(200):
            row = rng.choice(rows)
            cc = [[rng.getrandbits(32), rng.getrandbits(32)] for _ in row['order']]
            bs = [rng.getrandbits(32) for _ in range(8)]
            with np.errstate(over='ignore'):
                values = inputs(np.array(bs, dtype=np.uint64))
                numeric = expression(row, cc, values).tolist()
            actual = frozen(row, cc, values)[-1]
            self.assertEqual(actual, numeric)
            self.assertEqual(actual, [expression(row, cc, inputs(b)) for b in bs])
            kk = [[z3.BitVecVal(k, 32) for k in pair] for pair in cc]
            self.assertEqual(actual, [z3.simplify(expression(row, kk, inputs(z3.BitVecVal(b, 32)))).as_long()
                                     for b in bs])

    def test_parity_gate_matches_independent_one_bit_z3(self):
        rows, _ = make_manifest(2)
        rng = random.Random(8642839)
        for case in range(80):
            row = rng.choice(rows)
            ks = [z3.BitVecs(f'joint_truth_{case}_{i}a joint_truth_{case}_{i}b', 1) for i in range(4)]
            s = z3.SolverFor('QF_BV')
            for p in (0, 1, 1 << 19, (1 << 19) | 1):
                outs = []
                for q in (0, 1):
                    values = [z3.BitVecVal(v & 1, 1) for v in inputs(p)]
                    for op, w, cc in zip(row['order'], row['wiring'], ks):
                        pool = values+cc
                        if op == 'shr16':
                            v = z3.BitVecVal(q, 1)
                        else:
                            a, b = pool[w[0]], pool[w[1]]
                            v = (a*b+pool[w[2]] if op == 'mad' else a^b if op == '^' else a&b if op == '&' else a|b)
                        values.append(v)
                    outs.append(values[-1])
                s.add(outs[0] != outs[1])
            self.assertEqual(bool(parity_models(row)), s.check() == z3.sat)

    def test_inverse_tail_normalization_restores_actual_pinned_constants(self):
        results = controls()
        rng = random.Random(8642840)
        for r in results:
            self.assertTrue(r['inverse_affine_tail_normalization'])
            self.assertEqual(r['constants'][0], [9, 0xFD7046C5])
            self.assertEqual(r['constants'][3][0], 4097)
            for _ in range(8):
                bs = np.array([rng.getrandbits(32) for _ in range(8)], dtype=np.uint64)
                with np.errstate(over='ignore'):
                    values = inputs(bs); expected = wanted(values, r['node'])
                self.assertEqual(frozen(r, r['constants'], values)[-1], expected.tolist())

    def test_runtime_root_verify_rejects_fixed_root_or_a_free_e(self):
        verify(CONTROL, 3)
        with self.assertRaisesRegex(ValueError, 'Fixed-root'):
            verify(dict(CONTROL, runtime_node=False))
        fake = dict(live='joint_early', runtime_node=True,
                    order=['shr16', '^', '^', 'mad'],
                    wiring=[[0], [0, 9], [10, 8], [11, 12, 13]],
                    constants=[[0, 0], [0, 0], [0, 0], [4097, 0x7ED55D16]])
        with self.assertRaisesRegex(ValueError, 'equivalence not proved'):
            verify(fake, 3)

    def test_both_physical_arm_orientations_are_checked(self):
        fake = dict(CONTROL,
                    order=CONTROL['order']+['mad', '^', '^'],
                    wiring=CONTROL['wiring']+[[3, 14, 15], [1, 14], [13, 15]],
                    constants=CONTROL['constants']+[[33, 0xE9F8CC1D], [0, 0], [0, 0]])
        with self.assertRaisesRegex(ValueError, 'swapped=True'):
            verify(fake, 3)


if __name__ == '__main__':
    unittest.main()
