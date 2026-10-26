"""Independent checks of polynomial relaxation, symmetries and real witnesses."""
from itertools import product
import random
import unittest

import z3

from codex.search_cross_encoding863 import C1,middle
from codex.search_head_polynomial863 import (FAMILIES,canonical_poly,control_coefficients,
    evaluate,interpolate_falling,normalize,poly,search)
from codex.test_free_middle_keys863 import frozen


def falling_coefficients(monomial):
    # x**n=sum_k S(n,k)*x*(x-1)*...*(x-k+1), computed independently.
    out=[0]*len(monomial);row=[1]
    for n,a in enumerate(monomial):
        if n:
            row=[0]+[(row[k-1]if k-1<len(row)else 0)
                     +k*(row[k]if k<len(row)else 0)for k in range(1,n+1)]
        for k,value in enumerate(row):out[k]+=a*value
    return out


def control_rows(family):
    rows=[dict(op='mad',operands=[0,1,2],constants=[6,17])]
    if family=='polynomial_xor':
        rows += [dict(op='mad',operands=[1,1,2],constants=[21,0]),
                 dict(op='mad',operands=[2,3,4],constants=[9,29]),
                 dict(op='^',operands=[1,3,0],constants=[0,0])]
    else:
        rows += [dict(op='mad',operands=[0,1,2]if family=='scaled_quadratic_xor'else[0,2,3],
                      constants=[21,0]if family=='scaled_quadratic_xor'else[8,21]),
                 dict(op='^',operands=[1,2,0],constants=[0,0])]
        rows += [dict(op='mad',operands=[3,4,5],constants=[9,29])
                 if family=='scaled_quadratic_xor'else
                 dict(op='mad',operands=[0,1,3],constants=[0,0])]
    return rows


class HeadPolynomialTests(unittest.TestCase):
    def test_falling_basis_contains_all_degree_four_integer_polynomials(self):
        rng=random.Random(86471)
        for width in(5,8,11,32):
            mask=(1 << width)-1
            for _ in range(80):
                monomial=[rng.getrandbits(width)for _ in range(5)]
                original=falling_coefficients(monomial)
                normalized=canonical_poly(original,width)
                xs=range(1 << width)if width==5 else[0,1,mask,rng.getrandbits(width)]
                for x in xs:
                    target=sum(a*x**j for j,a in enumerate(monomial))&mask
                    self.assertEqual(target,poly(x,original)&mask)
                    self.assertEqual(target,poly(x,normalized)&mask)

    def test_real_four_instruction_controls_match_frozen_full32(self):
        rng=random.Random(86472)
        for family in FAMILIES:
            c=control_coefficients(family)
            rows=control_rows(family)
            for x in[0,1,0xffffffff]+[rng.getrandbits(32)for _ in range(150)]:
                target=frozen(x,rows)
                self.assertEqual(target,evaluate(x,c,family,32))
                self.assertEqual(target,z3.simplify(evaluate(z3.BitVecVal(x,32),c,family,32)).as_long())

    def test_all_output_and_polynomial_symmetries_preserve_function(self):
        rng=random.Random(86473)
        for family,width in product(FAMILIES,(5,8,11,32)):
            mask,high=(1 << width)-1,1 << (width-1)
            template=control_coefficients(family)
            for _ in range(80):
                c={k:[rng.getrandbits(width)for _ in v]if isinstance(v,list)
                   else rng.getrandbits(width)for k,v in template.items()}
                key=rng.getrandbits(width)
                nc,nk=normalize(c,key,family,width)
                self.assertEqual(0,nc['p'][0]&(high|1))
                self.assertEqual(0,(nc['q'][0]if family=='polynomial_xor'else nk)&(high|1))
                xs=range(1 << width)if width==5 else(0,mask,rng.getrandbits(width))
                for x in xs:self.assertEqual(evaluate(x,c,family,width)^key,evaluate(x,nc,family,width)^nk)

    def test_scaled_cuts_follow_from_two_target_observations(self):
        for width in(5,11,32):
            mask,high=(1 << width)-1,1 << (width-1)
            a,b,d,e,f,g,h=z3.BitVecs('cut_a cut_b cut_d cut_e cut_f cut_g cut_h',width)
            expr=lambda x:g*((a*x+b)^(d*x*(x-1)+e*x+f))+h
            solver=z3.SolverFor('QF_BV')
            solver.set(timeout=3000)
            solver.add((expr(0)&1)!=(expr(1)&1),expr(0)!=expr(high),
                       z3.Or(g&1==0,(a+e)&1==0,d&1==1))
            self.assertEqual(z3.unsat,solver.check())
            values=[middle(x^C1)&mask for x in(0,1,high)]
            self.assertNotEqual(values[0]&1,values[1]&1)
            self.assertNotEqual(values[0],values[2])

    def test_automatic_controls_on_stated_complete_domains(self):
        for family in FAMILIES:
            width=5 if family=='quadratic_xor_bypass'else 8
            mask=(1 << width)-1
            report=search(family,width,10,True,interpolate=family=='quadratic_xor_bypass')
            self.assertEqual('equivalent_on_relaxed_low_domain',report['result'],report)
            self.assertEqual(1 << width,report['verified_inputs'])
            self.assertFalse(report['new_executable'])
            for x in range(1 << width):
                self.assertEqual((frozen(x,control_rows(family))&mask)^report['output_key'],
                    evaluate(x,report['coefficients'],family,width))

    def test_interpolation_does_not_invert_even_elements(self):
        rng=random.Random(86474)
        for width in(5,8,11,32):
            mask=(1 << width)-1
            for _ in range(100):
                coefficients=[rng.getrandbits(width)for _ in range(5)]
                values=[poly(x,coefficients)&mask for x in range(5)]
                actual,valid=interpolate_falling(values,width)
                self.assertTrue(valid)
                self.assertEqual(canonical_poly(coefficients,width),actual)
                symbolic,valid=interpolate_falling([z3.BitVecVal(v,width)for v in values],width)
                self.assertTrue(z3.is_true(z3.simplify(valid)))
                self.assertEqual(actual,[z3.simplify(v).as_long()for v in symbolic])
            self.assertFalse(interpolate_falling([0,0,1],width)[1])

    def test_interpolated_full_low11_positive_witness(self):
        report=search('quadratic_xor_bypass',11,5,True,True,True)
        self.assertEqual('equivalent_on_relaxed_low_domain',report['result'],report)
        self.assertEqual(2048,report['verified_inputs'])

    def test_fixed_output_never_applies_free_output_normalization(self):
        rng=random.Random(86475)
        for family in FAMILIES:
            for width in(5,11,32):
                mask=(1 << width)-1
                for _ in range(35):
                    c={k:[rng.getrandbits(width)for _ in v]if isinstance(v,list)
                       else rng.getrandbits(width)for k,v in control_coefficients(family).items()}
                    key=rng.getrandbits(width)
                    nc,nk=normalize(c,key,family,width,True)
                    self.assertEqual(key,nk)
                    for x in(0,mask,rng.getrandbits(width)):
                        self.assertEqual(evaluate(x,c,family,width),evaluate(x,nc,family,width))
            report=search(family,11,5,True,family=='quadratic_xor_bypass',True,
                          input_key=0,fixed_output_key=0xb55afa53)
            self.assertEqual('equivalent_on_relaxed_low_domain',report['result'],report)
            self.assertEqual(0xb55afa53&2047,report['output_key'])


if __name__=='__main__':unittest.main()
