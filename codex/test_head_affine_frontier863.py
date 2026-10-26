"""Independent checks of bit-separated certificates and coverage refinements."""
from collections import Counter
from itertools import combinations_with_replacement, product
import json
from pathlib import Path
import random
import unittest

import numpy as np

from codex.audit_head_inner_product863 import audit, classify, POLYNOMIAL_ROLES, SCALED_ROLES
from codex.scan_head_inner_product863 import roles
from codex.screen_affine_head_keys863 import decode, encode, key_mask, residue_screen
from codex.screen_affine_head_rectangles863 import axis_separated, rectangles
from codex.search_cross_encoding863 import C1, middle
from codex.search_head_merge863 import evaluate as merged, normalize_output

ROOT = Path(__file__).resolve().parent


class AffineFrontierTests(unittest.TestCase):
    def test_inner_product_complete_audit(self):
        row = audit()
        self.assertEqual((8, 2), (row['closed_orders'], row['open_orders']))
        self.assertEqual(15750, row['affine_wiring_patterns'])
        self.assertEqual({'projected_grammar_unsat': 144, 'unknown': 6}, row['coarse_query_counts'])
        self.assertEqual({('mad', '^', 'mad', 'mad'), ('mad', 'mad', '^', 'mad')},
                         {tuple(x) for x in row['remaining_orders']})
        for second, third, last in product(combinations_with_replacement(range(4), 2),
                roles(), combinations_with_replacement(range(6), 2)):
            category, case = classify(second, third, last)
            if case is not None:
                self.assertIn(1, second)
                self.assertTrue(2 in third or 2 in last)
                self.assertIn(3, last)

    def test_polynomial_and_scaled_unknown_refinements_full32(self):
        rng = random.Random(864381)
        mask = 0xffffffff
        for _ in range(100):
            a, b, key, d, e, x, o = [rng.getrandbits(32) for _ in range(7)]
            t = (a*x+b) & mask
            u = t ^ key
            for role in POLYNOMIAL_ROLES:
                pool = (x, t, u, d, e)
                v = (pool[role[0]]*pool[role[1]]+pool[role[2]]) & mask
                self.assertEqual(v ^ u ^ o, v ^ t ^ (key ^ o))
            for role in SCALED_ROLES:
                pool = (x, t, u, d, e)
                v = (pool[role[0]]*pool[role[1]]+pool[role[2]]) & mask
                mul, offset = (d, e) if role == (2, 3, 4) else (e, d)
                for family, tail in (('key_input', x), ('key_first', t), ('key_middle', u)):
                    coefficients = [a, b, mul, offset, key]
                    self.assertEqual(v ^ tail, merged(x, coefficients, family, 32))
                    normalized, new_o = normalize_output(coefficients, o, family, 32)
                    self.assertEqual(v ^ tail ^ o, merged(x, normalized, family, 32) ^ new_o)

    def test_axis_separation_in_original_full32_integer_arithmetic(self):
        rng = random.Random(864382)
        rects = list(rectangles())
        accepted = 0
        for _ in range(5000):
            a, b, c, d = [rng.getrandbits(32) for _ in range(4)]
            residue = rng.randrange(8)
            base, h, k = rng.choice(rects)
            corners = [8*(base+offset)+residue for offset in (0, h, k, h+k)]
            effective_b = ((a*residue+b) >> 3) & 255
            projected = [((a*x+b) & 2047) >> 3 for x in corners]
            self.assertEqual(projected, [(a*(base+offset)+effective_b) & 255
                for offset in (0, h, k, h+k)])
            if not bool(axis_separated(a & 255, effective_b, base, h, k)):
                continue
            accepted += 1
            # Independently inspect the original 11-bit words. Their changed
            # bit sets are disjoint across BOTH x and t, not t alone.
            ts = [(a*x+b) & 2047 for x in corners]
            self.assertEqual(0, ts[0] ^ ts[1] ^ ts[2] ^ ts[3])
            self.assertEqual(0, ((ts[0] ^ ts[1]) | (corners[0] ^ corners[1])) &
                                ((ts[0] ^ ts[2]) | (corners[0] ^ corners[2])))
            values = []
            for x in corners:
                t = (a*x+b) & 0xffffffff
                q = ((t ^ c) & (x | d)) ^ (t | (x & c))
                r = (q | t) & (x ^ d)
                # A polynomial of degree eight, much broader than two MADs.
                values.append((q*q*r*x + 19*q*t + 23*r*r*r*r*r*r*r*r + x) & 2047)
            self.assertEqual(0, (values[3]-values[1]-values[2]+values[0]) & 2047)
        self.assertGreater(accepted, 400)

    def test_all_recorded_key_masks_match_scalar_full11_evaluation(self):
        report = json.loads((ROOT/'affine_head_keys863_joint.json').read_text())
        total = 0
        for row in report['residues']:
            residue = row['residue']
            for witness in row['witnesses']:
                base, h, k = (witness[key] for key in ('base_y', 'h_y', 'k_y'))
                vals = [middle((8*(base+d)+residue) ^ C1) & 2047 for d in (0, h, k, h+k)]
                bits = sum(int(word, 16) << (64*i)
                           for i, word in enumerate(witness['allowed_upper_key_mask']))
                for key in range(256):
                    p, q, r, s = [value ^ (8*key+7) for value in vals]
                    self.assertEqual(bool(bits >> key & 1), ((s-q-r+p) & 2047) == 0)
                total += 1
        self.assertGreater(total, 100)

    def test_key_bitmap_roundtrip_and_known_witness_all_residues(self):
        array = np.array([[0, 2**64-1], [3, 1 << 63]], dtype=np.uint64)
        np.testing.assert_array_equal(array, decode(encode(array)))
        counts = []
        for residue in range(8):
            row = residue_screen(residue, positive=True)
            self.assertTrue(row['positive_pair_and_zero_key_retained'])
            counts.append(row['remaining_upper_key_assignments'])
        self.assertLess(min(counts), 65536*256)

    def test_joint_lifting_retains_every_bit_and_exact_intersection(self):
        report = json.loads((ROOT/'affine_head_keys863_joint.json').read_text())
        pairs = decode(report['surviving_pairs']).astype(np.int64)
        masks = decode(report['surviving_key_masks'])
        a, b = pairs.T
        expected = np.full(masks.shape, np.iinfo(np.uint64).max, dtype=np.uint64)
        for row in report['residues']:
            table = decode(row['table'])
            r = row['residue']
            expected &= table[((a & 255) << 8) | (((a*r+b) >> 3) & 255)]
        np.testing.assert_array_equal(expected, masks)
        self.assertTrue(np.all(np.any(masks, axis=1)))
        self.assertEqual(report['remaining_upper_key_assignments'], int(np.bitwise_count(masks).sum()))
        # Every high-A/low-B lift of the first residue is checked for membership
        # against the other seven tables, including those ultimately rejected.
        first = decode(report['residues'][0]['table'])
        ids = np.flatnonzero(np.any(first, axis=1))
        candidates = {(int(aa), int(bb)) for aa, bb in pairs}
        counted = 0
        for high_a, low_b in product(range(8), repeat=2):
            aa = (ids >> 8) + 256*high_a
            bb = (ids & 255)*8 + low_b
            allowed = first[ids].copy()
            for row in report['residues'][1:]:
                r = row['residue']
                allowed &= decode(row['table'])[((aa & 255) << 8) | (((aa*r+bb) >> 3) & 255)]
            for x, y in zip(aa[np.any(allowed, axis=1)], bb[np.any(allowed, axis=1)]):
                self.assertIn((int(x), int(y)), candidates)
                counted += 1
        self.assertEqual(len(candidates), counted)


if __name__ == '__main__':
    unittest.main()
