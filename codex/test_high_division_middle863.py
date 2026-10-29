"""Positive controls for high-only divisor constraints and D,T,D semantics."""
import unittest

from codex.audit_comparison_core863 import fibre
from codex.audit_division_first_t863 import source_inverse
from codex.audit_high_division_middle863 import high_points, screen, suffix_checks
from codex.screen_division_xor_affine863 import quotient
from codex.search_early_boundary863 import MASK, early_inputs
from tests.frozen_problem import Machine, DebugInfo


class HighDivisionMiddleTests(unittest.TestCase):
    def test_high_parity_screen_retains_actual_division_MAD_programs(self):
        points = high_points()
        for op, d in (('//', 65536), ('cdiv', 65536), ('%', 98305)):
            def control(v, op=op, d=d): return (17*quotient(v[0], d, op)+v[4]) & MASK
            result = screen(0, op == 'cdiv', points, control)
            self.assertTrue(result['high_class_variation'])
            self.assertTrue(any(lo <= d <= hi for lo, hi in result['survivor_intervals']))

    def test_high_suffix_relaxations_retain_true_native_MAD_and_masks(self):
        q = [v[0] >> 16 for v in fibre()['initial_values']]
        for g in (0, 1, 17, 18, MASK):
            targets = [(g*x+12345) & 65535 for x in q]
            self.assertEqual(suffix_checks(q, targets)['status'], 'open')
        self.assertTrue(suffix_checks(q, [(x*x+7) & 65535 for x in q])['quadratic_consistent'])
        planted = suffix_checks(q[:4], [(17*x*x+18*x+7) & 65535 for x in q[:4]])
        self.assertEqual(planted['quadratic_null_certificate']['target_residue'], 0)
        for op in ('^', '&', '|'):
            targets = [((x^0xABCD) if op == '^' else (x&0xABCD) if op == '&' else (x|0xABCD)) for x in q]
            result = suffix_checks(q, targets)
            self.assertTrue(result['bitop_fits']); self.assertIsNone(result['unary_bit_collision'])

    def test_D_T_D_controls_match_frozen_and_high_results_equal_numerator(self):
        values = [0, 1, MASK, MASK-1, 1 << 31, 65536, 65537, 0x7FFFFFFF]
        vv = [early_inputs(source_inverse(0, v)) for v in values]
        initials = [v[4] for v in vv]
        for first_op, d1 in (('//', 65536), ('cdiv', 65536), ('%', 98305)):
            for family in ('affine', 'square', 'xor'):
                middle = {'affine':('multiply_add', 64, 56, 32, 8),
                          'square':('multiply_add', 64, 56, 56, 40),
                          'xor':('^', 64, 56, 8)}[family]
                for last_op in ('//', '%', 'cdiv', '>>'):
                    for d2 in ((0, 1, 16, 32, MASK) if last_op == '>>' else (1, 2, 65536, MASK)):
                        m = Machine([0]*32, [
                            {'valu':[(first_op, 56, 0, 16)]},
                            {'valu':[middle]},
                            {'valu':[(last_op, 72, 64, 24)]},
                        ], DebugInfo({}))
                        m.enable_pause = False; m.enable_debug = False
                        s = m.cores[0].scratch
                        for address, words in ((0, values), (8, initials), (16, [d1]*8), (24, [d2]*8),
                                               (32, [17]*8), (40, [0xFEDCBA99]*8)):
                            s[address:address+8] = words
                        m.run(); qq = [quotient(n, d1, first_op) for n in values]
                        mm = [((17*q+i) if family == 'affine' else (q*q+0xFEDCBA99)
                               if family == 'square' else q^i) & MASK for q, i in zip(qq, initials)]
                        out = [n >> d2 if last_op == '>>' else quotient(n, d2, last_op) for n in mm]
                        self.assertEqual(s[64:72], mm); self.assertEqual(s[72:80], out)
                        self.assertTrue(all(y == n for n, y in zip(mm, out) if y > 1 << 31))
                        if last_op == '%': self.assertNotIn(MASK, out)

    def test_prepared_shift_normalizations_match_frozen(self):
        values = [0, 1, MASK, MASK-1, 1 << 31, 65535, 65536, 0x7FFFFFFF]
        for count in (0, 1, 7, 16, 19, 31, 32, 63, MASK):
            factor = 1 << count if count < 32 else 0
            normalized_right = ('//', 40, 0, 16) if count < 32 else ('multiply_add', 40, 0, 24, 24)
            slots = [('>>', 32, 0, 8), normalized_right]
            # Large Python left shifts allocate the entire temporary integer;
            # counts<=63 suffice for the frozen positive control. The formal
            # word identity for every count>=32 is the same zero case.
            if count <= 63: slots += [('<<', 48, 0, 8), ('multiply_add', 56, 0, 16, 24)]
            m = Machine([0]*32, [{'valu':slots}], DebugInfo({}))
            m.enable_pause = False; m.enable_debug = False
            s = m.cores[0].scratch
            for address, words in ((0, values), (8, [count]*8), (16, [factor]*8), (24, [0]*8)):
                s[address:address+8] = words
            m.run(); expected = [n >> count for n in values]
            self.assertEqual(s[32:40], expected); self.assertEqual(s[40:48], expected)
            if count <= 63:
                expected = [(n << count) & MASK for n in values]
                self.assertEqual(s[48:56], expected); self.assertEqual(s[56:64], expected)


if __name__ == '__main__':
    unittest.main()
