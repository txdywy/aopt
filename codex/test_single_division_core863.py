"""Brute finite domains and genuine controls for the single-division core."""
from itertools import product
import hashlib
import random
import struct
import unittest

from codex.audit_comparison_core863 import fibre
from codex.audit_floor_xor_prefix863 import inverse_profile
from codex.audit_quotient_circular863 import circular_span, unit_span_scan
from codex.audit_runtime_shift_core863 import base_fields, monomials
from codex.audit_small_numerator863 import bit_fits
from codex.close_computed_denominator863 import edge_target
from codex.screen_division_xor_affine863 import quotient
from codex.screen_xor_mad_polynomial import consistent
from codex.search_early_boundary863 import MASK, early_inputs, wanted
from tests.frozen_problem import Machine, DebugInfo


class QuotientCircularTests(unittest.TestCase):
    def test_circular_span_against_all_interval_origins(self):
        for width in (2, 3):
            m = 1 << width
            for members in range(1, 1 << m):
                points = [x for x in range(m) if members >> x & 1]
                expected = min(max((x-origin) % m for x in points) for origin in range(m))
                self.assertEqual(circular_span(points, width), expected)
                for shift in (1, m-1):
                    self.assertEqual(circular_span([x+shift for x in points], width), expected)

    def test_vector_unit_scan_matches_independent_full_unit_enumeration(self):
        rng = random.Random(8642892)
        datasets = [(w, [rng.randrange(1 << w) for _ in range(8)]) for w in (4, 8, 12)]
        datasets.append((16, fibre()['target']))
        for width, targets in datasets:
            m = 1 << width; spans = []
            for a in range(1, m, 2):
                points = sorted({a*y % m for y in targets})
                gaps = [y-x for x, y in zip(points, points[1:])] + [points[0]+m-points[-1]]
                spans.append(m-max(gaps))
            actual = unit_span_scan(targets, width)
            self.assertEqual(actual['minimum_span'], min(spans))
            self.assertEqual(actual['enumerated_odd_units'], m//2)
            raw = b''.join(struct.pack('<I', x) for x in spans)
            self.assertEqual(actual['span_array_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(unit_span_scan([17]*8)['necessary_max_divisor'], MASK)

    def test_rem_circular_bound_retains_actual_large_divisor_programs(self):
        rng = random.Random(8642893); vv = fibre()['initial_values']
        for _ in range(8):
            k, g = rng.getrandbits(32), rng.getrandbits(32) | 1
            pp = [(33*v[4]+k) & MASK for v in vv]
            for d in (1, 3, 65537, 302675, (1 << 31)+1, MASK):
                yy = [(v[0]+g*(p%d)) & MASK for v, p in zip(vv, pp)]
                actual = unit_span_scan(yy, width=10)
                self.assertGreaterEqual(actual['necessary_max_divisor'], d)
                aa = pow(g*d, -1, 1 << 16)
                self.assertLessEqual(circular_span([aa*y for y in yy]), MASK//d)

    def test_floor_ceil_circle_retains_products_and_unbounded_ceil_carry(self):
        rng = random.Random(8642894); vv = fibre()['initial_values']
        for _ in range(32):
            k, g = rng.getrandbits(32), rng.getrandbits(32) | 1
            pp = [v[0] ^ k for v in vv]
            rr = [(v[4]*p) & MASK for v, p in zip(vv, pp)]
            for d in (1, 2, 3, 6, 31, 65536, MASK):
                for op in ('//', 'cdiv'):
                    yy = [(r+g*quotient(p, d, op)) & MASK for r, p in zip(rr, pp)]
                    inverse_g = pow(g, -1, 65536)
                    self.assertLessEqual(circular_span([d*inverse_g*y for y in yy]), d-1)
        self.assertEqual(quotient(MASK, MASK, 'cdiv'), 1)

    def test_large_remainder_has_only_two_low_word_states(self):
        rng = random.Random(8642895); vv = fibre()['initial_values']
        for _ in range(100):
            k = rng.getrandbits(32); d = rng.randrange((1 << 31)+1, MASK+1)
            pp = [v[0] ^ k for v in vv]; qq = [p%d for p in pp]
            self.assertLessEqual(len({q & 65535 for q in qq}), 2)
            for source in range(9):
                for output in ([q ^ v[source] for q, v in zip(qq, vv)],
                               [(q*v[source]+p) & MASK for q, v, p in zip(qq, vv, pp)]):
                    self.assertLessEqual(len({y & 65535 for y in output}), 2)


class FloorXorProfileTests(unittest.TestCase):
    def test_even_floor_xor_permutations_and_small_inverse_profiles(self):
        for width in range(2, 9):
            size = 1 << width
            for d in range(2, size, 2):
                outputs = [p ^ (p//d) for p in range(size)]
                self.assertEqual(set(outputs), set(range(size)))
                actual = inverse_profile(d, width)
                self.assertTrue(all(outputs[p] == y for y, p in enumerate(actual)))
                # Projection to low targets is complete: large input words
                # cannot secretly map below the sampled leading-bit range.
                if width > 2:
                    small = inverse_profile(d, width-1)
                    self.assertEqual(actual[:size//2], small)

    def test_fullword_leading_bit_and_odd_divisor_collision(self):
        rng = random.Random(8642896)
        for _ in range(2000):
            p, d = rng.getrandbits(32), rng.randrange(2, MASK+1)
            self.assertEqual((p ^ (p//d)).bit_length(), p.bit_length())
            odd = rng.getrandbits(32) | 1
            self.assertEqual((odd-1) ^ ((odd-1)//odd), odd ^ (odd//odd))
        for d in (128, 129, 65536, MASK-1, MASK):
            self.assertEqual(inverse_profile(d), list(range(128)))

    def test_small_target_profile_retains_real_first_MAD_and_bitop(self):
        for d in (2, 6, 34, 126, 128, MASK-1):
            pp = inverse_profile(d)
            vv = [early_inputs(b) for b in pp]
            matrix = [monomials(base_fields(b, v)) for b, v in zip(pp, vv)]
            # True synthetic program: P=b*1+0; q=P//D; out=P XOR q.
            # Its target is a full32 permutation when D is even.
            self.assertTrue(consistent(matrix, pp, 32))
            self.assertTrue(bit_fits(vv, pp))
            self.assertEqual([p ^ (p//d) for p in pp], list(range(128)))

    def test_target_zero_and_small_inverse_values_are_real(self):
        for y in (0, 1, 2, 3, 63, 127, 128, 65535, 1 << 31, MASK):
            b = edge_target(y); self.assertEqual(wanted(early_inputs(b)), y)
        at_zero = early_inputs(edge_target(0))
        self.assertTrue(all(at_zero)); self.assertEqual(len(set(at_zero)), 9)


class ConservativePermutationTests(unittest.TestCase):
    def test_complete_small_conservative_permutation_domain(self):
        count = 0
        for values in product(*(range(p+1) for p in range(8))):
            if len(set(values)) == 8:
                self.assertEqual(values, tuple(range(8))); count += 1
        self.assertEqual(count, 1)

    def test_mask_dominance_and_three_op_controls_match_frozen(self):
        rng = random.Random(8642897)
        for _ in range(300):
            initial = rng.sample(range(16), 16)
            for other in ([rng.randrange(16) for _ in range(16)], [0]*16, [15]*16):
                for op in ('&', '|'):
                    values = [x&y if op == '&' else x|y for x, y in zip(initial, other)]
                    if len(set(values)) == 16: self.assertEqual(values, initial)
        vv = fibre()['initial_values']; lanes = [v[4] for v in vv]; initial = [v[0] for v in vv]
        for prefix in ('mad', '^'):
            for op in ('//', '%', 'cdiv'):
                for d in (1, 3, 6, 65536, (1 << 31)+1, MASK):
                    for family in ('product', 'Gq+P', 'GP+q', 'xor'):
                        g = 17 if family == 'Gq+P' else 18; k = 0xFEDCBA99
                        first = ('multiply_add', 48, 0, 16, 24) if prefix == 'mad' else ('^', 48, 0, 24)
                        last = {'product':('multiply_add', 64, 8, 48, 56),
                                'Gq+P':('multiply_add', 64, 40, 56, 48),
                                'GP+q':('multiply_add', 64, 40, 48, 56),
                                'xor':('^', 64, 48, 56)}[family]
                        machine = Machine([0]*32, [{'valu':[first]}, {'valu':[(op, 56, 48, 32)]},
                                                   {'valu':[last]}], DebugInfo({}))
                        machine.enable_pause = False; machine.enable_debug = False
                        s = machine.cores[0].scratch
                        for address, values in ((0, lanes), (8, initial), (16, [33]*8), (24, [k]*8),
                                                (32, [d]*8), (40, [g]*8)): s[address:address+8] = values
                        machine.run()
                        pp = [((33*b+k) & MASK) if prefix == 'mad' else b^k for b in lanes]
                        qq = [quotient(p, d, op) for p in pp]
                        expected = [((i*p+q) if family == 'product' else (g*q+p) if family == 'Gq+P'
                                     else (g*p+q) if family == 'GP+q' else p^q) & MASK
                                    for i, p, q in zip(initial, pp, qq)]
                        self.assertEqual(s[56:64], qq); self.assertEqual(s[64:72], expected)


if __name__ == '__main__':
    unittest.main()
