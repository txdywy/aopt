import itertools
import random
import unittest
from codex.screen_xor_mad_polynomial import consistent


class ModularEliminationTests(unittest.TestCase):
    def test_exhaustive_assignment_oracle(self):
        rng = random.Random(901)
        for width in range(1, 5):
            modulus = 1 << width
            for _ in range(100):
                rows, cols = rng.randrange(1, 5), rng.randrange(1, 4)
                matrix = [[rng.randrange(modulus) for _ in range(cols)]
                          for _ in range(rows)]
                rhs = [rng.randrange(modulus) for _ in range(rows)]
                expected = any(all((sum(v*x for v, x in zip(line, values))-b)
                                   % modulus == 0 for line, b in zip(matrix, rhs))
                               for values in itertools.product(range(modulus), repeat=cols))
                self.assertEqual(expected, consistent(matrix, rhs, width),
                                 (width, matrix, rhs))

    def test_nonunit_pivots(self):
        self.assertFalse(consistent([[2]], [1], 4))
        self.assertTrue(consistent([[2]], [6], 4))
        self.assertFalse(consistent([[2], [4]], [6, 0], 4))
        self.assertTrue(consistent([[2], [4]], [6, 12], 4))


if __name__ == '__main__':
    unittest.main()
