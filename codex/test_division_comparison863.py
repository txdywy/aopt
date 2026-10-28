import random
import unittest

import numpy as np

from codex.audit_comparison_core863 import fibre, fitting_forms
from codex.audit_division_first_t863 import first_intervals, refine, source_inverse
from codex.screen_division_boolean863 import small_mask
from codex.screen_division_xor_affine863 import first_divisors, quotient, affine_fit
from codex.search_early_boundary863 import early_inputs, MASK
from tests.frozen_problem import Machine, DebugInfo


class DivisionComparisonTests(unittest.TestCase):
    def test_vector_boolean_screen_matches_independent_scalar_forms(self):
        rng = random.Random(8642859)
        qq = np.array([[rng.randrange(8) for _ in range(256)] for _ in range(8)], dtype=np.uint64)
        y = list(range(8)); got = small_mask(qq, y)
        expected = []
        for column in qq.T:
            q = [int(x) for x in column]
            expected.append(bool(fitting_forms(q, y, 3)) or all((a >> 1) == (b >> 1) for a, b in zip(q, y)))
        self.assertEqual(got.tolist(), expected)

    def test_vector_boolean_screen_retains_planted_controls(self):
        rng = random.Random(8642860)
        for _ in range(40):
            q = [rng.getrandbits(16) for _ in range(8)]
            c = [rng.randrange(2) for _ in range(8)]
            a, b = rng.getrandbits(16), rng.getrandbits(16)
            for y in ([a*x+z for x, z in zip(q, c)], [a*z+x for x, z in zip(q, c)],
                      [x*z+b for x, z in zip(q, c)], [x*z+x for x, z in zip(q, c)],
                      [x*z+z for x, z in zip(q, c)], [x*x+z for x, z in zip(q, c)],
                      [x ^ z for x, z in zip(q, c)]):
                self.assertTrue(small_mask(np.array(q, dtype=np.uint64)[:, None], y)[0])

    def test_parity_interval_refinement_matches_exhaustive_divisors(self):
        rng = random.Random(8642857)
        for _ in range(100):
            n, a = [rng.randrange(65536) for _ in range(2)]
            want = rng.randrange(2)
            actual = refine([[1, 127], [256, 511]], n, a, want)
            got = [d for lo, hi in actual for d in range(lo, hi+1)]
            expected = [d for d in [*range(1, 128), *range(256, 512)] if (n//d ^ a//d) & 1 == want]
            self.assertEqual(got, expected)

    def test_three_bit_interval_filter_keeps_bijective_t_controls(self):
        for divisor in (7, 13, 31):
            nums = [1000+divisor*i for i in range(8)]
            ys = [(17*((n//divisor) ^ 5)+23) & MASK for n in nums]
            actual, _ = first_intervals(nums, ys)
            self.assertTrue(any(lo <= divisor <= hi for lo, hi in actual))
            got = [d for lo, hi in actual for d in range(lo, hi+1)]
            expected = []
            for d in range(1, max(nums)+1):
                qs = [n//d for n in nums]
                if all(((qs[i] ^ qs[j]) & (2*((ys[i] ^ ys[j]) & -(ys[i] ^ ys[j]))-1)) ==
                       ((ys[i] ^ ys[j]) & -(ys[i] ^ ys[j])) for i in range(8) for j in range(i)):
                    expected.append(d)
            self.assertEqual(got, expected)

    def test_live_word_inverses_recover_real_primary_inputs(self):
        rng = random.Random(8642858)
        for source in range(9):
            for _ in range(32):
                value = rng.getrandbits(32)
                if source == 3:
                    value &= ~511
                if source == 7:
                    value &= 8191
                b = source_inverse(source, value)
                self.assertEqual(early_inputs(b)[source], value)

    def test_divisor_intervals_match_exhaustive_floor_rem_ceil(self):
        rng = random.Random(8642852)
        for _ in range(32):
            n0, n1 = [rng.randrange(25000, 65536) for _ in range(2)]
            planted = rng.randrange(1, 4096)
            for op in ('//', '%', 'cdiv'):
                required = (quotient(n0, planted, op) ^ quotient(n1, planted, op)) & 65535
                if required < 2:
                    continue
                actual, _ = first_divisors(n0, n1, required, op, direct_limit=7, divisor_limit=4095)
                expected = [d for d in range(1, 4096)
                            if ((quotient(n0, d, op) ^ quotient(n1, d, op)) & 65535) == required]
                self.assertEqual(actual.tolist(), expected, (n0, n1, required, op))
                self.assertIn(planted, actual)

    def test_full_word_divisor_positive_controls_include_large_remainders(self):
        for op in ('//', 'cdiv'):
            ds, _ = first_divisors(0, 1 << 31, 0x8000, op)
            self.assertIn(65536, ds)
        ds, stats = first_divisors(0, 1 << 31, 0x8000, '%')
        self.assertIn(98304, ds)
        self.assertIn((1 << 31)-32768, ds)
        self.assertGreater(stats['quotient_intervals'], 0)
        ds, _ = first_divisors(0, 1 << 31, 0x9000, '%')
        self.assertIn((1 << 31)-0x9000, ds)

    def test_affine_fit_retains_planted_division_xor_programs(self):
        rng = random.Random(8642853)
        bs = [*range(16), *(rng.getrandbits(32) for _ in range(64))]
        vv = [early_inputs(b) for b in bs]
        for op, divisor, partner in (('//', 65536, 0), ('cdiv', 65536, 0), ('%', 98304, 3)):
            target = [(17*(quotient(v[4], divisor, op) ^ v[partner])+23) & MASK for v in vv]
            ds = np.array([divisor-1, divisor, divisor+1], dtype=np.uint64)
            got, aa, bb, _ = affine_fit(ds, vv, target, 4, partner, op)
            self.assertIn(divisor, got)
            j = got.tolist().index(divisor)
            self.assertEqual(int(aa[j]), pow(17, -1, 1 << 32))
            self.assertEqual(int(bb[j]), (-23*pow(17, -1, 1 << 32)) & MASK)

    def test_division_ceil_comparison_and_xor_mad_match_frozen(self):
        numerators = [0, 1, 0x7FFFFFFF, 0x80000000, MASK, MASK, MASK-1, 65536]
        denominators = [1, MASK, 32769, 65536, MASK, 2, 0x80000000, 65535]
        rhs = [MASK, 0, 13, 0x80000000, 31, MASK, 1, 65535]
        for op in ('//', '%', 'cdiv', '<', '=='):
            m = Machine([0]*32, [
                {'valu':[(op, 24, 0, 8)]},
                {'valu':[('^', 32, 24, 16)]},
                {'valu':[('multiply_add', 56, 32, 40, 48)]},
            ], DebugInfo({}))
            m.enable_pause = False; m.enable_debug = False
            m.cores[0].scratch[0:8] = numerators
            m.cores[0].scratch[8:16] = denominators
            m.cores[0].scratch[16:24] = rhs
            m.cores[0].scratch[40:48] = [17]*8
            m.cores[0].scratch[48:56] = [23]*8
            m.run()
            intermediate = [(int(n < d) if op == '<' else int(n == d) if op == '==' else quotient(n, d, op))
                            for n, d in zip(numerators, denominators)]
            self.assertEqual(m.cores[0].scratch[24:32], intermediate)
            self.assertEqual(m.cores[0].scratch[56:64],
                             [(17*(q ^ r)+23) & MASK for q, r in zip(intermediate, rhs)])

    def test_all_six_boolean_mad_forms_survive_positive_controls(self):
        rng = random.Random(8642854)
        ff = fibre()
        for _ in range(100):
            q = [rng.getrandbits(16) for _ in range(8)]
            c = [rng.randrange(2) for _ in range(8)]
            a, b = rng.getrandbits(16), rng.getrandbits(16)
            cases = {
                'A*q+c':[a*x+y for x, y in zip(q, c)],
                'A*c+q':[a*y+x for x, y in zip(q, c)],
                'q*c+B':[x*y+b for x, y in zip(q, c)],
                'q*c+q':[x*y+x for x, y in zip(q, c)],
                'q*c+c':[x*y+y for x, y in zip(q, c)],
                'q*q+c':[x*x+y for x, y in zip(q, c)],
            }
            for name, y in cases.items():
                self.assertIn(name, fitting_forms(q, y))
        self.assertEqual(len(ff['legal_points']), 8)

    def test_boolean_mad_fit_matches_small_width_brute_force(self):
        rng = random.Random(8642855); width = 4; modulus = 1 << width; mask = modulus-1
        for _ in range(300):
            q = [rng.randrange(modulus) for _ in range(4)]
            y = [rng.randrange(modulus) for _ in range(4)]
            got = fitting_forms(q, y, width)
            expected_aq = any(all(t in ((a*x) & mask, (a*x+1) & mask) for x, t in zip(q, y))
                              for a in range(modulus))
            expected_ac = any(all(t in (x, (x+a) & mask) for x, t in zip(q, y))
                              for a in range(modulus))
            expected_b = any(all(t in (b, (x+b) & mask) for x, t in zip(q, y))
                             for b in range(modulus))
            self.assertEqual('A*q+c' in got, expected_aq)
            self.assertEqual('A*c+q' in got, expected_ac)
            self.assertEqual('q*c+B' in got, expected_b)


if __name__ == '__main__':
    unittest.main()
