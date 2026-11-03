from itertools import product
import random
import unittest

from codex.search_mixed_affine_restore863 import evaluate, normalize, search
from codex.search_keyed_live_restore863 import evaluate as live_evaluate
from codex.search_restore_scaled863 import solve as scaled_solve


class MixedAffineRestoreTests(unittest.TestCase):
    def test_normalization_exhaustive_two_bit_programs(self):
        for coefficients in product(range(4),repeat=7):
            a,b,c,d,g,h,k = coefficients
            if not g&1 or not (a+c)&1:
                continue
            normalized = normalize(coefficients,2)
            self.assertEqual([evaluate(x,coefficients,2) for x in range(4)],
                             [evaluate(x,normalized,2) for x in range(4)])

    def test_random_normalization_preserves_exposed_functions(self):
        rng = random.Random(8642955)
        for width in (5,11,16,32):
            high = 1 << (width-1)
            for _ in range(100):
                coeff = [rng.getrandbits(width) for _ in range(7)]
                coeff[4] |= 1
                if not (coeff[0]+coeff[2])&1:
                    coeff[0] ^= 1
                normalized = normalize(coeff,width)
                for j in (1,3,6):
                    self.assertEqual(0,normalized[j]&(high|1))
                self.assertEqual(0,normalized[0]&high)
                for x in range(64):
                    self.assertEqual(evaluate(x,coeff,width),evaluate(x,normalized,width))

    def test_automatic_controls_and_real_live_projection(self):
        for coeff in ([21,19,6,23,5,17,42],[6,19,21,23,9,31,13]):
            result = search(5,10,control=coeff)
            self.assertEqual('reduced_sat',result['status'],result)
            for x in range(32):
                self.assertEqual(evaluate(x,coeff,5),evaluate(x,result['coefficients'],5))
        row = dict(order=['mad','mad','^','mad'],wiring=[[0,3,4],[2,4,5],[3,4],[5,6,7]])
        a,b,c,d,g,h,k = coeff
        constants = [[a,b],[c,d],[0,0],[g,h]]
        for x in range(2048):
            self.assertEqual(evaluate(x,coeff,11),live_evaluate(row,constants,x,k,32)&2047)
        control = scaled_solve(11,10,True)
        self.assertEqual('relaxed_reduced_sat',control['status'],control)
        single = dict(order=['mad','mad','^','mad'],wiring=[[2,3,4],[2,4,5],[3,4],[5,6,7]])
        for x in range(2048):
            y = x^k
            plain = (g*(((a*y+b)&2047)^((c*y+d)&2047))+h)&2047
            self.assertEqual(plain,live_evaluate(single,constants,x,k,32)&2047)

    def test_real_family_is_unsat_without_pinning_constants(self):
        result = search(11,10)
        self.assertEqual('unsat',result['status'],result)
        self.assertIsNone(result['control'])
        self.assertFalse(result['new_executable'])


if __name__ == '__main__':
    unittest.main()
