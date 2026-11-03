import random
import unittest

from codex.joint_polynomial_quotient863 import applicable,certificate
from codex.search_early_boundary863 import NODE


class JointPolynomialQuotientTests(unittest.TestCase):
    def test_floor_ceil_mixed_difference_identity_smallword(self):
        rng=random.Random(8642939);mask=255
        mixed=lambda x:(x[3]-x[1]-x[2]+x[0])&mask
        for _ in range(24):
            coeff=[rng.randrange(256) for _ in range(5)]
            points=[1,17,33,49]
            p=[sum(c*x**i for i,c in enumerate(coeff))&mask for x in points]
            self.assertEqual(mixed(p),0)
            for d in range(1,256):
                floor=[x//d for x in p];r=[x%d for x in p]
                self.assertEqual((d*mixed(floor)+mixed(r))&mask,0)
                ceil=[(x+d-1)//d for x in p];r=[q*d-x for x,q in zip(p,ceil)]
                self.assertTrue(all(0<=x<d for x in r))
                self.assertEqual((d*mixed(ceil)-mixed(r))&mask,0)
        proof=certificate()
        self.assertEqual(len(proof['small_divisor_checks']),4)
        self.assertGreater(proof['rectangle']['distance'],2*proof['divisor_at_least5_quotient_maximum'])

    def test_scope_keeps_quotient_factors_and_bitwise_prefixes(self):
        for op in ('//','%','cdiv','shr'):
            row=dict(order=['mad','mad',op,'mad'],wiring=[[0,8,9],[8,9,5],[9],[8,9,10]])
            self.assertIsNotNone(applicable(row,NODE))
            self.assertIsNone(applicable(row,0))
            row['wiring'][-1]=[8,10,9]
            self.assertIsNone(applicable(row,NODE))
            row['wiring'][-1]=[8,9,10];row['order'][0]='^';row['wiring'][0]=[0,8]
            self.assertIsNone(applicable(row,NODE))


if __name__=='__main__':unittest.main()
