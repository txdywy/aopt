"""Exact inverse divisor domains and positive shift/division controls."""
import random
import unittest

from codex.audit_denominator_products863 import intersect, linear_class
from codex.audit_shift_division863 import point_divisors, shift_zero_pair
from codex.screen_division_xor_affine863 import quotient
from codex.search_early_boundary863 import MASK, early_inputs
from tests.frozen_problem import Machine, DebugInfo


def members(domain):
    return set(domain['isolated_divisors']) | {d for lo, hi in domain['intervals'] for d in range(lo, hi+1)}


class ShiftDivisionTests(unittest.TestCase):
    def test_point_inverse_domains_against_complete_small_divisor_domains(self):
        rng = random.Random(8642898)
        for width in (3, 5, 8):
            mask = (1 << width)-1
            cases = [(n, y) for n in range(mask+1) for y in range(2, mask+1)] if width < 8 else [
                (rng.randrange(mask+1), rng.randrange(2, mask+1)) for _ in range(500)]
            for n, y in cases:
                for op in ('//', '%', 'cdiv'):
                    actual = members(point_divisors(n, y, op, width))
                    expected = {d for d in range(1, mask+1) if quotient(n, d, op) == y}
                    self.assertEqual(actual, expected, (width, n, y, op))
        self.assertEqual(point_divisors(13, 13, '%', 8)['intervals'], [[14, 255]])

    def test_two_zero_filters_keep_real_independent_and_dependent_programs(self):
        vv = [early_inputs(b) for b in shift_zero_pair(0)]
        self.assertEqual([v[0] >> 16 for v in vv], [0, 0])
        for op, d in (('//', 17), ('cdiv', 17), ('%', 65537)):
            # Valid independent synthetic kernel:17*(e>>16)+op(b,D).
            yy = [quotient(v[4], d, op) for v in vv]
            self.assertGreater(yy[0], 1)
            domain = point_divisors(vv[0][4], yy[0], op)
            self.assertTrue(d in domain['isolated_divisors'] or any(lo <= d <= hi for lo, hi in domain['intervals']))
            self.assertEqual(quotient(vv[1][4], d, op), yy[1])
            # Valid dependent synthetic kernel:17*b+op(e>>16,D).
            yy = [(17*v[4]+quotient(v[0] >> 16, d, op)) & MASK for v in vv]
            cls = intersect(linear_class(vv[0][4], yy[0]), linear_class(vv[1][4], yy[1]))
            self.assertIsNotNone(cls); self.assertEqual(17 % cls[1], cls[0])

    def test_prepared_shr_independent_and_dependent_quotients_match_frozen(self):
        rng = random.Random(8642899)
        vv = [early_inputs(b) for b in [0, 1, MASK, 1 << 31, *(rng.getrandbits(32) for _ in range(4))]]
        ee, bb = [v[0] for v in vv], [v[4] for v in vv]
        for dependent in (False, True):
            for op in ('//', '%', 'cdiv'):
                for s in (0, 1, 16, 31, 32, MASK):
                    for d in (1, 17, MASK):
                        m = Machine([0]*32, [
                            {'valu':[('>>', 48, 0, 8)]},
                            {'valu':[(op, 56, 48 if dependent else 16, 24)]},
                            {'valu':[('multiply_add', 64, 48, 32, 56)]},
                        ], DebugInfo({}))
                        m.enable_pause = False; m.enable_debug = False
                        scratch = m.cores[0].scratch
                        for address, values in ((0, ee), (8, [s]*8), (16, bb), (24, [d]*8), (32, [17]*8)):
                            scratch[address:address+8] = values
                        m.run(); shifted = [e >> s for e in ee]
                        qq = [quotient(n, d, op) for n in (shifted if dependent else bb)]
                        self.assertEqual(scratch[48:56], shifted)
                        self.assertEqual(scratch[56:64], qq)
                        self.assertEqual(scratch[64:72], [(17*x+q) & MASK for x, q in zip(shifted, qq)])


if __name__ == '__main__':
    unittest.main()
