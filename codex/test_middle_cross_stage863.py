"""Independent checks of the cross-stage polynomial exclusions."""
import itertools
import unittest

from codex.screen_quadratic_xor_affine863 import (
    control, quadratic_coefficients, search, target,
)
from codex.screen_square_mad_xor863 import candidates, search as square_search
from codex.search_middle_outer_roles863 import check, roles
from codex.screen_affine_xor_affine863 import search as affine_search


class CrossStageTests(unittest.TestCase):
    def test_quadratic_reconstruction_against_all_small_polynomials(self):
        for width in range(1, 4):
            modulus = 1 << width
            for a, b, c in itertools.product(range(modulus), repeat=3):
                expected = [(a*x*x+b*x+c) % modulus for x in range(modulus)]
                result = quadratic_coefficients(c, a+b+c, 4*a+2*b+c, width)
                self.assertIsNotNone(result)
                aa, bb, cc = result
                self.assertEqual(expected, [(aa*x*x+bb*x+cc) % modulus
                                            for x in range(modulus)])

    def test_vector_search_against_brute_force(self):
        modulus = 8
        for fn in (target, control, lambda x: x*x*x+2*x+1):
            expected = set()
            for a, b, c, d, e in itertools.product(range(modulus), repeat=5):
                if all((((a*x*x+b*x+c) % modulus) ^ ((d*x+e) % modulus))
                       == fn(x) % modulus for x in range(modulus)):
                    expected.add((d, e))
            actual = search(3, fn, chunk_size=7)
            self.assertEqual(expected, {tuple(row[3:]) for row in actual['survivors']})

    def test_quadratic_positive_control_at_proof_width(self):
        result = search(11, control)
        self.assertIn([7, 13, 19, 33, 42], result['survivors'])

    def test_square_relaxation_accepts_all_output_operand_types(self):
        for kind, key, other in candidates(4):
            if (kind, key) not in {('input', 0), ('square-input', 0),
                                   ('constant', 7), ('square-constant', 9)}:
                continue
            # An actual two-MAD program with nonlinear first intermediate.
            first = lambda x: x*x+x if kind == 'square-input' else x*x+9
            fn = lambda x: (first(x)*first(x)+x) ^ other(x)
            result = square_search(4, fn)
            self.assertIn({'kind': kind, 'key': key}, result['survivors'])

    def test_both_middle_exclusions(self):
        self.assertEqual([], search(11)['survivors'])
        self.assertEqual([], square_search(11)['survivors'])

    def test_outer_role_coverage(self):
        expected = {(min(a, b), max(a, b), c)
                    for a, b, c in itertools.product(range(5), repeat=3)
                    if 2 in (a, b, c)}
        self.assertEqual(35, len(expected))
        self.assertEqual(expected, set(roles()))

    def test_outer_search_positive_controls(self):
        for source in ('input', 'constant'):
            fn = lambda x: 9*((33*x+17) ^ (x if source == 'input' else 42))+5
            result = check(source, (2, 3, 4), width=8, timeout_ms=5000, fn=fn)
            self.assertEqual('sat', result['result'])
            self.assertEqual('unsat', result['verification'])

    def test_target_is_permutation_at_low_widths(self):
        for width in range(1, 13):
            mask = (1 << width)-1
            self.assertEqual(mask+1, len({target(x) & mask for x in range(mask+1)}))

    def test_xor_affine_classification(self):
        for width in range(2, 9):
            mask = (1 << width)-1
            for key in range(mask+1):
                slope = ((1 ^ key)-key) & mask
                is_affine = all((slope*x+key) & mask == x ^ key
                                for x in range(mask+1))
                self.assertEqual(key & (mask//2) in (0, mask//2), is_affine)

    def test_affine_search_against_exhaustive_oracle(self):
        width, affine_bits = 4, 2
        modulus, mask = 16, 15
        fn = lambda x: 7*((3*x+5) ^ 6)+9
        expected = set()
        # Directly enumerate original, unnormalized A,B,K,D. C follows from x=0.
        for a, b, key, d in itertools.product(range(1, modulus, 2), range(modulus),
                                             range(modulus), range(1, modulus, 2)):
            c = (fn(0)-d*(b ^ key)) & mask
            if not all((d*(((a*x+b) & mask) ^ key)+c) & mask == fn(x) & mask
                       for x in range(modulus)):
                continue
            if key & 1:
                key, d, c = key ^ mask, (-d) & mask, (c-d) & mask
            if key & (modulus//2):
                key, c = key ^ (modulus//2), (c-d*(modulus//2)) & mask
            expected.add((a, b, key, d, c))
        self.assertTrue(expected)
        result = affine_search(width, affine_bits, fn, chunk_size=7)
        self.assertEqual(expected, {tuple(row) for row in result['survivors']})

    def test_affine_control_and_exclusion_at_proof_width(self):
        fn = lambda x: 9*((33*x+17) ^ 512)+5
        self.assertIn([33, 17, 512, 9, 5], affine_search(fn=fn)['survivors'])
        self.assertEqual([], affine_search()['survivors'])


if __name__ == '__main__':
    unittest.main()
