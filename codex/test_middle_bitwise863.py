"""Check new AND/OR search semantics against independent exhaustive oracles."""
import itertools
import json
import random
import subprocess
import sys
import unittest
import z3

from codex.prove_middle_polynomial863 import certificate
from codex.screen_middle_lattice863 import search
from codex.screen_middle_single_mad863 import orders
from codex.search_middle_outer_roles863 import check
from codex.search_live_middle863 import live_words, search as live_search


class MiddleBitwiseTests(unittest.TestCase):
    def test_lattice_enumeration_matches_scalar_oracle(self):
        width, mask = 3, 7
        for bitwise, fn in itertools.product(('&', '|'),
                (lambda x: (x*x+3) & (3*x+5), lambda x: (x*x+3) | (3*x+5))):
            result = search(width, bitwise, fn, chunk_size=3)
            expected_affine, expected_square = [], []
            def condition(p, y):
                return (p & y) == y if bitwise == '&' else (p | y) == y
            for a, b in itertools.product(range(mask+1), repeat=2):
                if all(condition((a*x+b) & mask, fn(x) & mask) for x in range(mask+1)):
                    expected_affine.append([a, b])
            for key in range(mask+1):
                if all(condition((x*x+key) & mask, fn(x) & mask) for x in range(mask+1)):
                    expected_square.append(key)
            self.assertEqual(expected_affine, result['affine_survivors'])
            self.assertEqual(expected_square, result['square_survivors'])

    def test_lattice_positive_programs_at_proof_width(self):
        for bitwise in ('&', '|'):
            fn = (lambda x: (7*x*x+13*x+19) & (3*x+5)) if bitwise == '&' else (
                  lambda x: (7*x*x+13*x+19) | (3*x+5))
            result = search(fn=fn, bitwise=bitwise)
            self.assertIn([3, 5], result['affine_survivors'])
            self.assertFalse(result['excluded'])

    def test_lattice_target_exclusions(self):
        for bitwise in ('&', '|'):
            self.assertTrue(search(bitwise=bitwise)['excluded'])

    def test_outer_role_positive_controls(self):
        for bitwise, source in itertools.product(('&', '|'), ('input', 'constant')):
            def fn(x):
                t, other = 33*x+17, x if source == 'input' else 42
                u = t & other if bitwise == '&' else t | other
                return 9*u+5
            result = check(source, (2, 3, 4), width=8, timeout_ms=5000,
                           fn=fn, bitwise=bitwise)
            self.assertEqual('sat', result['result'], result)
            self.assertEqual('unsat', result['verification'], result)

    def test_first_bitwise_polynomial_positive_controls(self):
        for bitwise in ('&', '|'):
            command = [sys.executable, '-m', 'codex.screen_xor_mad_polynomial',
                       '--width', '8', '--slice', 'middle', '--control',
                       '--bitwise', bitwise]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=20)
            self.assertEqual(0, completed.returncode, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertIn(37, result['survivors'])

    def test_polynomial_certificate_against_arbitrary_degree_controls(self):
        proof = certificate()
        self.assertEqual(1024, proof['mixed_difference'])
        rng = random.Random(863)
        for degree in range(65):
            coeff = [rng.randrange(2048) for _ in range(degree+1)]
            ys = [sum(a*pow(x, n, 2048) for n, a in enumerate(coeff)) & 2047
                  for x in proof['inputs']]
            self.assertEqual(0, (ys[3]-ys[2]-ys[1]+ys[0]) & 2047)

    def test_order_partition(self):
        self.assertEqual(27, len(orders()))
        self.assertTrue(all(row.count('mad') == 1 for row in orders()))
        self.assertEqual(set(orders()), {row for row in itertools.product(('mad','^','&','|'), repeat=3)
                                        if row.count('mad') == 1})

    def test_live_low_words_are_real_fullwidth_preimages(self):
        rng = random.Random(864)
        inverse = pow(4097, -1, 1 << 32)
        for width in (1, 8, 11, 12, 19):
            mask = (1 << width)-1
            for _ in range(32):
                b_low = rng.randrange(mask+1)
                a = (b_low ^ 0xC761C23C) & mask
                prior = ((a-0x7ED55D16)*inverse) & 0xffffffff
                self.assertEqual(a, (4097*prior+0x7ED55D16) & 0xffffffff)
                self.assertEqual(0, a >> 19)
                b = a ^ (a >> 19) ^ 0xC761C23C
                actual = [z3.simplify(v).as_long() for v in live_words(z3.BitVecVal(b_low,width),width)]
                self.assertEqual([b & mask, a & mask, prior & mask, 0], actual)

    def test_live_search_rejects_unsound_narrowing(self):
        for width, seq in ((20, ['mad','mad','^']), (12,['mad','>>','mad'])):
            with self.assertRaises(ValueError):
                live_search(seq,width=width)

    def test_live_positive_witness_and_verifier(self):
        result = live_search(['mad','mad','^'], control=True, pin_control=True)
        self.assertEqual('sat', result['result'])
        self.assertEqual('unsat', result['verification'])


if __name__ == '__main__':
    unittest.main()
