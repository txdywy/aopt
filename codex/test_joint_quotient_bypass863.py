import random
import unittest

import numpy as np
import z3

from codex.audit_joint_quotient_bypass863 import (bypass, collision_certificate, row_families,
                                                rectangle_certificate, square_tail_certificate,
                                                mask_rectangle_certificate)
from codex.refine_joint_quotient_range863 import target_quotient_maximum
from codex.search_joint_live_boundary863 import inputs, wanted, NODE, MASK
from codex.test_live_boundary863 import frozen


class JointQuotientBypassTests(unittest.TestCase):
    def test_recorded_families_match_the_actual_frozen_wiring(self):
        rng = random.Random(8642950)
        for row in row_families():
            cc = [[rng.getrandbits(32),rng.getrandbits(32)] for _ in row['order']]
            special = next(i for i,op in enumerate(row['order']) if op in ('//','%','cdiv'))
            cc[special][0] = 65537
            bs = [rng.getrandbits(32) for _ in range(8)]
            vv = inputs(np.array(bs,dtype=np.uint64))
            actual = frozen(row,cc,vv)
            for lane,b in enumerate(bs):
                q = actual[8+special][lane]
                v = inputs(b)
                if row['family']=='numerator_collision':
                    expected = ((cc[2][0]*q+cc[2][1]) & MASK)^v[0]
                elif row['family']=='scaled_z_or_z_square_plus_b':
                    expected = (cc[3][0]*bypass('z_or_z_square_plus_b',v)+q) & MASK
                elif row['family']=='square_tail_zero_v':
                    t = (v[2]*cc[0][0]+v[3]) & MASK
                    expected = ((t*t+q)&MASK)^cc[3][0]
                elif row['family']=='masked_prior':
                    expected = (cc[3][0]*q+(v[4]&cc[2][0])) & MASK
                else:
                    expected = (cc[3][0]*q+bypass(row['family'],v)) & MASK
                self.assertEqual(expected,actual[-1][lane],row)

    def test_collision_is_legal_and_independent_of_the_divisor(self):
        cert = collision_certificate()
        self.assertNotEqual(*cert['residuals'])
        self.assertEqual(*cert['numerators'])
        self.assertTrue(all(p['frozen_reference_checked'] for p in cert['legal_points']))
        for d in (1,2,3,65535,65537,MASK):
            n0,n1 = cert['numerators']
            self.assertEqual(n0//d,n1//d)
            self.assertEqual(n0%d,n1%d)
            self.assertEqual((n0+d-1)//d,(n1+d-1)//d)

    def test_divisor2_has_no_required_bit0_dependency(self):
        b = z3.BitVec('qrange_low_dependency_b',32)
        first,second = inputs(b),inputs(b^65536)
        for x,y in zip(first,second):
            s = z3.SolverFor('QF_BV');s.add((x&65535)!=(y&65535))
            self.assertEqual(z3.unsat,s.check())
        p0,p1 = z3.BitVecs('qrange_prefix0 qrange_prefix1',32)
        for divisor in (2,4):
            for ceil in (False,True):
                def q(p):
                    if not ceil:
                        return z3.UDiv(p,divisor)
                    return z3.Extract(31,0,z3.UDiv(z3.ZeroExt(1,p)+divisor-1,z3.BitVecVal(divisor,33)))
                s = z3.SolverFor('QF_BV');s.add((p0&(2*divisor-1))==(p1&(2*divisor-1)),(q(p0)&1)!=(q(p1)&1))
                self.assertEqual(z3.unsat,s.check())
        s = z3.SolverFor('QF_BV');s.add((wanted(first,NODE)&1)==(wanted(second,NODE)&1))
        self.assertEqual(z3.unsat,s.check())
        self.assertEqual(0x55555555,target_quotient_maximum('//'))
        self.assertEqual(0x55555555,target_quotient_maximum('cdiv'))

    def test_scaled_bypass_rectangle_and_zero_factor_replay(self):
        rect = rectangle_certificate()
        self.assertGreater(rect['target_distance'],2*rect['quotient_bound_divisor_at_least5'])
        square = square_tail_certificate()
        self.assertEqual('unsat',square['status'])
        self.assertEqual(512,len(square['primary_b']))

    def test_mask_rectangle_eliminates_every_mask_and_multiplier(self):
        cert = mask_rectangle_certificate()
        points = cert['primary_b'];vv = [inputs(b) for b in points]
        delta = lambda xs:(xs[3]+xs[0]-xs[1]-xs[2]) & MASK
        rng = random.Random(8642954)
        for _ in range(100):
            a,g,mask = [rng.getrandbits(32) for _ in range(3)]
            self.assertEqual(0,delta([(a*v[3]+v[6])&MASK for v in vv]))
            self.assertEqual(0,delta([v[4]&mask for v in vv]))
            if g&1:
                self.assertEqual(1 << 31,(pow(g,-1,1 << 32)*delta([wanted(v,NODE) for v in vv]))&MASK)
        self.assertTrue(all(p['frozen_reference_checked'] for p in cert['legal_points']))


if __name__ == '__main__':
    unittest.main()
