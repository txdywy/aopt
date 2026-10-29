"""Positive controls and independent fullword checks for downward pairs."""
from itertools import product
import random
import unittest

import z3

from codex.audit_conservative_tail863 import factor_offset_fits, fixed_product_fits, scalar_initial_fits
from codex.audit_division_first_t863 import source_inverse
from codex.audit_shift_division863 import shift_zero_pair
from codex.audit_two_downward_ops863 import oriented_checks, two_result_class
from codex.cegis_c1_absorb863 import _instr_options
from codex.screen_division_xor_affine863 import quotient
from codex.search_early_boundary863 import MASK, early_inputs, wanted
from tests.frozen_problem import Machine, DebugInfo


class DownwardPairTests(unittest.TestCase):
    def test_bounded_tail_filters_keep_actual_nonunit_and_offset_controls(self):
        vv = [early_inputs(b) for b in shift_zero_pair(4)]
        for q1 in (0, 1):
            qq = [0, q1]
            for initial in range(9):
                for g in (0, 17, 18, MASK):
                    yy = [(g*v[initial]+q) & MASK for v, q in zip(vv, qq)]
                    fits = scalar_initial_fits(vv, yy, 1)
                    self.assertTrue(any(r['initial'] == initial and r['q_at_second'] == q1 and
                                        g % r['scalar_class'][1] == r['scalar_class'][0] for r in fits))
                    yy = [(v[initial]*q+g) & MASK for v, q in zip(vv, qq)]
                    self.assertIn(dict(initial=initial, q_at_second=q1), factor_offset_fits(vv, yy, 1))
            for i in range(9):
                for j in range(i, 9):
                    self.assertIn([i, j], fixed_product_fits(vv[0], (vv[0][i]*vv[0][j]) & MASK))

    def test_actual_zero_small_guards_match_independent_fullword_SMT(self):
        for source in range(9):
            vv = [early_inputs(b) for b in shift_zero_pair(source)]
            yy = [wanted(v) for v in vv]; bound = vv[1][source]
            square = z3.BitVec('zero_square_control_word', 32)
            s = z3.SolverFor('QF_BV'); s.set(timeout=10000); s.add(square*square == yy[0])
            self.assertEqual(s.check(), z3.unsat)
            for initial in range(9):
                q = z3.BitVec('bounded_control_q', 32); g = z3.BitVec('bounded_control_g', 32)
                s = z3.SolverFor('QF_BV'); s.set(timeout=10000)
                s.add(z3.ULE(q, bound), g*vv[0][initial] == yy[0], g*vv[1][initial]+q == yy[1])
                self.assertEqual(s.check(), z3.unsat)
                h = z3.BitVec('bounded_control_h', 32)
                s = z3.SolverFor('QF_BV'); s.set(timeout=10000)
                s.add(z3.ULE(q, bound), h == yy[0], vv[1][initial]*q+h == yy[1])
                self.assertEqual(s.check(), z3.unsat)

    def test_oriented_filters_keep_all_planted_small_ancestor_outputs(self):
        rng = random.Random(8642900)
        vv = [early_inputs(source_inverse(4, p)) for p in (1, 2)]
        for _ in range(200):
            g, a, b = [rng.getrandbits(32) for _ in range(3)]
            for f2 in range(3):
                yy = [(g+a) & MASK, (g*f2+b) & MASK]
                result = oriented_checks(vv, yy, [a, b])
                self.assertEqual(result['status'], 'open')
                if g:
                    self.assertEqual(result['forced_scalar'], g)
                    self.assertTrue(result['scalar_survives'])
            result = oriented_checks(vv, [a, b], [a, b])
            self.assertEqual(result['status'], 'open')  # F1=0 must remain unresolved.

    def test_all_two_result_tail_partitions_match_frozen_zero_semantics(self):
        rng = random.Random(8642901)
        for op in ('mad', '^', '&', '|'):
            for wiring in _instr_options(10, op):
                if not (9 in wiring and 10 in wiring): continue
                kind = two_result_class(op, wiring)
                words = [rng.getrandbits(32) for _ in range(13)]
                expected = None
                if kind in ('oriented_prepared_coefficient', 'oriented_initial_coefficient'):
                    addend = wiring[2]; other = 9 if addend == 10 else 10
                    words[other] = 0; expected = words[addend]
                elif kind == 'oriented_bitwise': words[9] = 0; expected = words[10]
                elif kind == 'AND_zero': words[9] = 0; expected = 0
                elif kind == 'product_plus_factor_zero': words[wiring[2]] = 0; expected = 0
                elif kind == 'square_plus_word_nonsquare':
                    words[wiring[2]] = 0; expected = words[wiring[0]]**2 & MASK
                else:
                    words[9] = 0; expected = words[wiring[2]]
                instruction = ('multiply_add' if op == 'mad' else op, 104, *[8*j for j in wiring])
                m = Machine([0]*32, [{'valu':[instruction]}], DebugInfo({}))
                m.enable_pause = False; m.enable_debug = False
                for j, value in enumerate(words): m.cores[0].scratch[j*8:j*8+8] = [value]*8
                m.run(); self.assertEqual(m.cores[0].scratch[104:112], [expected]*8)

    def test_two_downward_operations_and_tail_match_frozen(self):
        rng = random.Random(8642902)
        vv = [early_inputs(b) for b in [0, 1, MASK, 1 << 31, *(rng.getrandbits(32) for _ in range(4))]]
        first, second = [v[4] for v in vv], [v[0] for v in vv]
        ops = ('//', '%', 'cdiv', '>>')
        tails = {'sum':('multiply_add', 64, 48, 32, 56),
                 'initial':('multiply_add', 64, 48, 8, 56),
                 'product':('multiply_add', 64, 48, 56, 40),
                 'square':('multiply_add', 64, 48, 48, 56),
                 'xor':('^', 64, 48, 56),
                 'one_result':('multiply_add', 64, 8, 32, 56)}
        for op1, op2, chain, variable, edge in product(ops, ops, (False, True), (False, True), (False, True)):
            d1, d2 = (1, MASK) if edge else (65537, 17)
            s1, s2 = (0, 32) if edge else (16, 17)
            k1, k2 = s1 if op1 == '>>' else d1, s2 if op2 == '>>' else d2
            for family, last in tails.items():
                a1 = 8 if op1 == '>>' and variable else 16
                a2 = 48 if op2 == '>>' and variable else 24
                m = Machine([0]*32, [
                    {'valu':[(op1, 48, 0, a1)]},
                    {'valu':[(op2, 56, 48 if chain else 8, a2)]},
                    {'valu':[last]},
                ], DebugInfo({}))
                m.enable_pause = False; m.enable_debug = False
                scratch = m.cores[0].scratch
                for address, values in ((0, first), (8, second), (16, [k1]*8), (24, [k2]*8),
                                        (32, [18]*8), (40, [0xFEDCBA99]*8)):
                    scratch[address:address+8] = values
                m.run()
                q1 = [n >> (s if variable else s1) if op1 == '>>' else quotient(n, d1, op1)
                      for n, s in zip(first, second)]
                q2 = [n >> (p if variable else s2) if op2 == '>>' else quotient(n, d2, op2)
                      for n, p in zip(q1 if chain else second, q1)]
                expected = [((a*18+b) if family == 'sum' else (a*i+b) if family == 'initial'
                             else (a*b+0xFEDCBA99) if family == 'product' else (a*a+b) if family == 'square'
                             else a^b if family == 'xor' else (i*18+b)) & MASK
                            for a, b, i in zip(q1, q2, second)]
                self.assertEqual(scratch[48:56], q1)
                self.assertEqual(scratch[56:64], q2)
                self.assertEqual(scratch[64:72], expected)


if __name__ == '__main__':
    unittest.main()
