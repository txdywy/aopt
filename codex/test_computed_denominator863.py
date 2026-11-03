"""Independent finite-domain and frozen-ISA checks for denominator audits."""
from itertools import combinations_with_replacement
from math import gcd
import random
import unittest

import numpy as np

from codex.branch_computed_denominator863 import (
    branch_scalars, constrain_default, data, default_q, lift_scalar,
    partial_bit_fits, scalar_roots,
)
from codex.audit_denominator_bitops863 import forced_xor
from codex.audit_denominator_products863 import (
    intersect, linear_class, normalize, scalar_initial_case,
    square_case, square_possible, weighted_case,
)
from codex.audit_prepared_numerator863 import (
    denominator_intervals, distinct_tag_slice, low_scalar_filter,
    numerator_candidates, quadratic_interval, remainder_progressions, vector_inverse,
)
from codex.audit_odd_denominator_bypass863 import admissible_pairs, solve_case
from codex.audit_division_first_t863 import source_inverse
from codex.screen_division_xor_affine863 import quotient
from codex.search_early_boundary863 import MASK, early_inputs
from tests.frozen_problem import Machine, DebugInfo


class ExactDenominatorInverseTests(unittest.TestCase):
    def test_integer_quadratic_intervals_against_direct_evaluation(self):
        rng = random.Random(8642882)
        for _ in range(1000):
            a, b, c = rng.randrange(1, 16), rng.randrange(-300, 301), rng.randrange(-100, 101)
            interval = quadratic_interval(a, b, c)
            actual = [] if interval is None else list(range(interval[0], interval[1]+1))
            expected = [x for x in range(-500, 501) if a*x*x-b*x+c <= 0]
            self.assertEqual(actual, expected)

    def test_floor_ceil_inverse_complete_small_domains(self):
        rng = random.Random(8642883)
        for width, samples in ((3, 200), (5, 200), (8, 400), (12, 100)):
            mask = (1 << width)-1
            for _ in range(samples):
                n, c, y = rng.randrange(1, min(128, mask)+1), rng.randrange(mask+1), rng.randrange(mask+1)
                for op in ('//', 'cdiv'):
                    intervals = denominator_intervals(n, c, y, op, width)
                    actual = [p for lo, hi in intervals for p in range(lo, hi+1)]
                    expected = [p for p in range(1, mask+1) if (n*p+quotient(c, p, op)) & mask == y]
                    self.assertEqual(sorted(actual), expected, (width, n, c, y, op))
        # An interval can contain multiple denominators near the vertex;
        # checking only quadratic root estimates would miss these values.
        n, c, y = 1, 240, 31
        self.assertEqual([p for a, b in denominator_intervals(n, c, y, '//', 8) for p in range(a, b+1)],
                         [p for p in range(1, 256) if (p+c//p) & 255 == y])

    def test_word_inverse_retains_actual_denominators_and_ceil_carry(self):
        rng = random.Random(8642884)
        for c in (0, 1, (1 << 31)+1, MASK):
            for p in (1, 2, 3, 65536, MASK, *(rng.randrange(1, 1 << 32) for _ in range(20))):
                n = rng.randrange(1, 129)
                for op in ('//', 'cdiv'):
                    y = (n*p+quotient(c, p, op)) & MASK
                    self.assertTrue(any(lo <= p <= hi for lo, hi in denominator_intervals(n, c, y, op)))
        self.assertEqual(quotient(MASK, MASK, 'cdiv'), 1)

    def test_remainder_progressions_against_all_denominator_words(self):
        rng = random.Random(8642885)
        for width in (3, 6, 10):
            mask = (1 << width)-1
            for _ in range(200):
                n, c, y = rng.randrange(1, mask+1), rng.randrange(mask+1), rng.randrange(mask+1)
                actual = [p for lo, hi, step in remainder_progressions(n, c, y, width)
                          for p in range(lo, hi+1, step)]
                expected = [p for p in range(1, mask+1) if (n*p+c%p) & mask == y]
                self.assertEqual(sorted(actual), expected)

    def test_zero_factor_numerator_candidates_are_complete(self):
        for width in (3, 5, 8):
            mask = (1 << width)-1
            for op in ('//', 'cdiv'):
                expected = [set() for _ in range(mask+1)]
                for c in range(mask+1):
                    for p in range(1, mask+1): expected[quotient(c, p, op)].add(c)
                for y in range(2, mask+1):
                    self.assertEqual(set(numerator_candidates(y, op, width)), expected[y])


class ScalarSliceTests(unittest.TestCase):
    def test_low_scalar_slice_represents_every_word_scalar(self):
        rng = random.Random(8642886); mask = 255
        for _ in range(100):
            xx = [16*rng.randrange(16) for _ in range(16)]
            low_tags = rng.sample(range(16), 16)
            yy = [rng.randrange(16)*16+t for t in low_tags]
            lows, _ = low_scalar_filter(xx, yy, width=8, split=4)
            expected = {g for g in range(256)
                        if len({(y-g*x) & mask for x, y in zip(xx, yy) if (y-g*x) & mask > 128}) <= 1}
            self.assertEqual({g for g in range(256) if g & 15 in lows}, expected)

    def test_high_slice_keeps_true_prepared_division_controls(self):
        xx = [i << 4 for i in range(16)]
        for g in (0, 1, 17, 18, 255):
            for op in ('//', '%', 'cdiv'):
                yy = [(g*x+quotient(255, (x >> 4)+1, op)) & 255 for x in xx]
                indices = distinct_tag_slice(xx, yy, tags=4)
                lows, _ = low_scalar_filter([xx[i] for i in indices], [yy[i] for i in indices], 8, 4)
                self.assertIn(g & 15, lows)
        lows, _ = low_scalar_filter([0], [128], 8, 4)
        self.assertEqual(len(lows), 16)  # Half-word is not a high quotient.

    def test_batch_inverse_matches_unique_scalar_inverse(self):
        rng = np.random.default_rng(8642887)
        xx = rng.integers(0, 1 << 32, 40, dtype=np.uint64)
        for source in (0, 1, 2, 4, 5, 6, 8):
            actual = [int(b) for b in vector_inverse(source, xx)]
            self.assertEqual(actual, [source_inverse(source, int(x)) for x in xx])


class SmallNumeratorBranchTests(unittest.TestCase):
    def test_linear_nonunit_classes_and_intersections_against_small_ring(self):
        rng = random.Random(8642888); width = 7; m = 1 << width
        for _ in range(500):
            a, b, c, d = [rng.randrange(m) for _ in range(4)]
            first, second = linear_class(a, b, width), linear_class(c, d, width)
            both = intersect(first, second)
            actual = set() if both is None else {x for x in range(m) if x % both[1] == both[0]}
            expected = {x for x in range(m) if (a*x-b) % m == 0 and (c*x-d) % m == 0}
            self.assertEqual(actual, expected)
        self.assertEqual(normalize([(0, 1), (1, 2), (3, 4)]), [(0, 1)])
        self.assertEqual(constrain_default(0, 0), (0, 0))
        self.assertIsNone(constrain_default(1, 0))
        for a in (1, 2, 4, 128, 65536, 1 << 31):
            p = 0x12345679; result = constrain_default((a*p) & MASK, a)
            self.assertEqual(result[0], p & ((1 << result[1])-1))

    def test_fullword_small_coefficient_roots_and_real_scalar_branches(self):
        rng = random.Random(8642889)
        for a in range(1, 129):
            for _ in range(3):
                g = rng.getrandbits(32); rhs = a*g
                roots = scalar_roots(a, rhs)
                self.assertEqual(len(roots), gcd(a, 1 << 32))
                self.assertTrue(all((a*x-rhs) & MASK == 0 for x in roots))
                self.assertIn(g, roots)
                self.assertEqual(set(scalar_roots(a, rhs, odd=True)), {x for x in roots if x & 1})
        nn = list(range(1, 33))
        for op in ('//', '%', 'cdiv'):
            for family, scalar in (('plus', 17), ('times', 18)):
                yy = [(3+scalar*quotient(n, 3, op) if family == 'plus'
                       else scalar*3+quotient(n, 3, op)) & MASK for n in nn]
                selected = [(n, y) for n, y in zip(nn, yy) if y > n]
                self.assertTrue(any(n >= 3 for n, _ in selected))
                self.assertIn(scalar, branch_scalars([n for n, _ in selected], [y for _, y in selected], op, family))

    def test_partial_bitop_fits_against_exhaustive_four_bit_keys(self):
        rng = random.Random(8642890)
        for _ in range(100):
            values = [[rng.randrange(16) for _ in range(9)] for _ in range(8)]
            yy = [rng.randrange(16) for _ in values]; widths = [rng.randrange(5) for _ in values]
            possible = []
            for k in range(16):
                possible.append([k]*len(values))
                for j in range(9):
                    possible.extend([[v[j]^k for v in values], [v[j]&k for v in values], [v[j]|k for v in values]])
            for a, b in combinations_with_replacement(range(9), 2):
                possible.extend([[v[a]^v[b] for v in values], [v[a]&v[b] for v in values], [v[a]|v[b] for v in values]])
            expected = any(all((p^y) & ((1 << w)-1) == 0 for p, y, w in zip(row, yy, widths)) for row in possible)
            self.assertEqual(bool(partial_bit_fits(values, yy, widths)), expected)

    def test_scalar_lifting_keeps_nonunit_controls_and_preserves_caps(self):
        nn = list(range(65)); values = [[n]*9 for n in nn]; pp = [n | 0x80000001 for n in nn]
        for op in ('//', '%', 'cdiv'):
            aa = [default_q(n, op) for n in nn]
            for family, g in (('plus', 17), ('times', 18)):
                yy = [((p+g*a) if family == 'plus' else (g*p+a)) & MASK for p, a in zip(pp, aa)]
                result = lift_scalar(values, yy, nn, op, family, cap=512)
                self.assertNotEqual(result['status'], 'excluded')
                if family == 'times': self.assertIn(1, [r['valuation'] for r in result['branches']])
                for row in result['branches']:
                    valuation = row['valuation']
                    rhs = [((y-a) & MASK) >> valuation for y, a in zip(yy, aa)]
                    for step in row['history'][:8]:
                        bit = step['bit']; count = 0
                        for k in range(1, 1 << (bit+1), 2):
                            good = True
                            for j in range(bit+1):
                                buckets = {}
                                for v, y, a, r in zip(values, yy, aa, rhs):
                                    key = tuple((x >> j) & 1 for x in v)
                                    out = (((y-k*a) if family == 'plus' else k*r) >> j) & 1
                                    if key in buckets and buckets[key] != out: good = False; break
                                    buckets[key] = out
                                if not good: break
                            count += good
                        self.assertEqual(step['remaining'], count)
        capped = lift_scalar([[0]*9], [1], [0], '//', 'plus', cap=2)
        self.assertEqual(capped['status'], 'open')
        self.assertTrue(any(not row['complete'] for row in capped['branches']))

    def test_weighted_and_scalar_filters_keep_actual_programs(self):
        _, vv, _, nn, matrix = data(4); vv, nn, matrix = vv[:33], nn[:33], matrix[:33]
        pp = [2*n+1 for n in nn]
        for op in ('//', '%', 'cdiv'):
            qq = [quotient(n, p, op) for n, p in zip(nn, pp)]
            for factor in (0, 3, 4, 7):
                yy = [(v[factor]*p+q) & MASK for v, p, q in zip(vv, pp, qq)]
                self.assertNotEqual(weighted_case(vv, yy, nn, matrix, factor, op)['status'], 'excluded')
                yy = [(17*v[factor]+q) & MASK for v, q in zip(vv, qq)]
                roots = scalar_initial_case(vv, yy, nn, factor)['scalar_classes']
                self.assertTrue(any(17 % modulus == base for base, modulus in roots))
            yy = [(p*p+q) & MASK for p, q in zip(pp, qq)]
            self.assertEqual(square_case(yy, nn, op)['status'], 'open')

    def test_square_criterion_against_complete_word_rings(self):
        for width in range(2, 13):
            m = 1 << width; expected = {x*x % m for x in range(m)}
            self.assertEqual({y for y in range(m) if square_possible(y, width)}, expected)

    def test_xor_forcing_and_odd_collision_filter_keep_real_controls(self):
        nn = list(range(65)); pp = [n | 0x80000001 for n in nn]
        for op in ('//', 'cdiv'):
            yy = [p ^ quotient(n, p, op) for n, p in zip(nn, pp)]
            forced = forced_xor(yy, nn, op)
            self.assertEqual(len(forced), len(nn))
            self.assertTrue(all(r['valid_large_prefix'] and r['forced_prefix'] == pp[r['index']] for r in forced))
        row = dict(op='mad', wiring=[4, 9, 10]); parity = [0, 1]
        pairs = admissible_pairs(row, parity, [[n, n ^ (1 << 31)] for n in range(8)])
        self.assertTrue(pairs)
        def target(v):
            p = (2*v[4]+1) & MASK
            return (v[0]*p+MASK%p) & MASK
        result = solve_case(row, parity, pairs, 0, target=target)
        self.assertEqual(result['status'], 'sat')

    def test_three_op_controls_match_frozen_division_and_tail_semantics(self):
        rng = random.Random(8642891)
        nn = [0, 1, MASK, 1 << 31, *(rng.getrandbits(32) for _ in range(4))]
        factor = [early_inputs(n)[0] for n in nn]
        for op in ('//', '%', 'cdiv'):
            for prepared in (False, True):
                for family in ('plus', 'times', 'weighted', 'xor'):
                    tails = {'plus':('multiply_add', 64, 56, 40, 48),
                             'times':('multiply_add', 64, 48, 40, 56),
                             'weighted':('multiply_add', 64, 8, 48, 56),
                             'xor':('^', 64, 48, 56)}
                    m = Machine([0]*32, [
                        {'valu':[('multiply_add', 48, 0, 16, 24)]},
                        {'valu':[(op, 56, 32 if prepared else 0, 48)]},
                        {'valu':[tails[family]]},
                    ], DebugInfo({}))
                    m.enable_pause = False; m.enable_debug = False
                    s = m.cores[0].scratch
                    for address, words in ((0, nn), (8, factor), (16, [2]*8), (24, [1]*8),
                                           (32, [MASK]*8), (40, [18]*8)): s[address:address+8] = words
                    m.run(); pp = [(2*n+1) & MASK for n in nn]
                    qq = [quotient(MASK if prepared else n, p, op) for n, p in zip(nn, pp)]
                    expected = [((p+18*q) if family == 'plus' else (18*p+q) if family == 'times'
                                 else (i*p+q) if family == 'weighted' else p^q) & MASK
                                for p, q, i in zip(pp, qq, factor)]
                    self.assertEqual(s[56:64], qq)
                    self.assertEqual(s[64:72], expected)


if __name__ == '__main__':
    unittest.main()
