"""Independent finite oracles and positive controls for new exclusions."""
from collections import Counter
import itertools
import random
import unittest

import z3

from codex.prove_first_affine_shift863 import (classify, even_multiplier_certificate,
    h, shapes, variable_count_query)
from codex.screen_live_middle_structure863 import (families, live, operand,
    parameter_count, prefix_certificate, rectangles, search)
from codex.screen_live_middle_outer863 import search as outer_search
from codex.search_live_constant_key863 import check as constant_key_check
from codex.search_live_middle863 import live_words, search as live_search
from codex.search_live_middle_roles863 import roles as middle_roles
from codex.search_live_single_mad863 import bitwise, control_function
from codex.close_live_middle_residual863 import (enumerate_xor, force_mask,
    masked_first_mad, check_tonly, scaled_xor_input_certificate)


def scalar_parameters(index, width, name):
    if name.startswith(('affine', 'mixed')):
        return index >> width, index & ((1 << width)-1)
    return (0, index) if name == 'product_key' else (0, 0)


class LiveMiddleTests(unittest.TestCase):
    def test_partition_covers_every_first_mad(self):
        width, modulus = 3, 8
        covered = set()
        for name, indices in families():
            for rank in range(parameter_count(width, name)):
                d, e = scalar_parameters(rank,width,name)
                covered.add(tuple(operand(x,width,name,indices,d,e) % modulus
                                  for x in range(modulus)))
        for k0, k1 in itertools.product(range(modulus), repeat=2):
            for i,j,k in itertools.product(range(5),repeat=3):
                values = []
                for x in range(modulus):
                    pool = [*live(x,width),k0,k1]
                    values.append((pool[i]*pool[j]+pool[k]) % modulus)
                self.assertIn(tuple(values),covered,(k0,k1,i,j,k))

    def test_screens_match_independent_scalar_oracle(self):
        width, mask = 4, 15
        def fn(x):
            b,a,p = live(x,width)
            return ((b*b+3*a+7) ^ (5*p+11)) & mask
        for bitwise in ('^','&','|'):
            result = search(width,bitwise,fn,chunk_size=7)
            for row in result['families']:
                name, indices = row['family'], row['indices']
                expected = []
                for rank in range(row['tested']):
                    d,e = scalar_parameters(rank,width,name)
                    ps = [operand(x,width,name,indices,d,e)&mask for x in range(mask+1)]
                    if bitwise == '^':
                        qs = [fn(x)^ps[x] for x in range(mask+1)]
                        valid = all((qs[t]-qs[r]-qs[q]+qs[p]) % (mask+1) == 0
                                    for p,q,r,t in rectangles(width,fn))
                    elif bitwise == '&':
                        valid = all((ps[x]&fn(x)) == fn(x) for x in range(mask+1))
                    else:
                        valid = all((ps[x]|fn(x)) == fn(x) for x in range(mask+1))
                    if valid:
                        expected.append(rank)
                self.assertEqual(expected,row['survivor_parameter_ids'],(bitwise,row))

    def test_screen_positive_programs_at_proof_width(self):
        for bitwise in ('^','&','|'):
            def fn(x):
                b,a,p = live(x,11)
                q, t = b*a+p, 33*a+42
                return {'^':q^t, '&':q&t, '|':q|t}[bitwise]
            result = search(bitwise=bitwise,fn=fn)
            row = next(r for r in result['families'] if r['family']=='affine_a')
            self.assertIn((33<<11)+42,row['survivor_parameter_ids'])
            self.assertFalse(result['excluded'])

    def test_rectangle_certificate_and_polynomial_controls(self):
        cert = prefix_certificate()
        self.assertTrue(cert['excluded'])
        self.assertEqual(1024,cert['mixed_difference'])
        self.assertEqual(0,cert['horizontal_mask'] & cert['vertical_mask'])
        rng = random.Random(863)
        for _ in range(200):
            # Arbitrary bitwise prefix, followed by a multivariate polynomial.
            key, coeff = rng.randrange(2048), rng.randrange(2048)
            degrees = [rng.randrange(8) for _ in range(4)]
            ys = []
            for b,a,p in cert['live_values']:
                q = (b^key) | (a & p)
                y = coeff
                for word, degree in zip((b,a,p,q),degrees):
                    y = y*pow(word,degree,2048)%2048
                ys.append(y)
            self.assertEqual(0,(ys[3]-ys[2]-ys[1]+ys[0])%2048)
        self.assertFalse(prefix_certificate(fn=lambda x:live(x,11)[0]*
            (live(x,11)[1]^37)+live(x,11)[2])['excluded'])

    def test_outer_screen_matches_exhaustive_coefficients(self):
        width, modulus = 3, 8
        fn = lambda x: 3*((2*x+1)^live(x,width)[1])+5
        result = outer_search(width,fn,chunk_size=7)
        self.assertFalse(result['excluded'])
        for row in result['rows']:
            name, indices, other = row['family'],row['indices'],row['xor_live']
            expected = []
            for rank in range(row['tested']):
                d,e = scalar_parameters(rank,width,name)
                us = [(operand(x,width,name,indices,d,e)%modulus)^live(x,width)[other]
                      for x in range(modulus)]
                for scale,offset in itertools.product(range(modulus),repeat=2):
                    if all((scale*us[x]+offset-fn(x))%modulus == 0 for x in range(modulus)):
                        expected.append([rank,scale,offset])
            self.assertEqual(expected,row['survivors_id_scale_offset'])

    def test_outer_positive_program_at_proof_width(self):
        result = outer_search(fn=lambda x:9*((34*x+17)^live(x,11)[1])+5)
        row = next(r for r in result['rows']
                   if r['family']=='affine_b' and r['xor_live']==1)
        self.assertIn([(34<<11)+17,9,5],row['survivors_id_scale_offset'])
        self.assertFalse(result['excluded'])

    def test_domain_guards(self):
        for width in (0,13,20):
            with self.assertRaises(ValueError):
                live(0,width)
        with self.assertRaises(ValueError):
            outer_search(fn=lambda x:2*x)

    def test_first_noninvertible_families(self):
        for width in range(2,9):
            modulus,half=1<<width,1<<(width-1)
            for i,j in itertools.combinations_with_replacement(range(3),2):
                for x in range(modulus):
                    p,q=live(x,width),live(x^half,width)
                    self.assertEqual(p[i]*p[j]%modulus,q[i]*q[j]%modulus)
                    for k in range(3):
                        self.assertEqual(0,(p[i]*p[j]+p[k])&1)

    def test_constant_key_search_discovers_controls(self):
        for name in ('affine_a','mixed_ba','mixed_ab'):
            coefficient=34 if name.startswith('mixed') else 33
            def fn(x):
                return 9*(operand(x,8,name,(),coefficient,17)^42)+5
            row=constant_key_check(name,width=8,fn=fn)
            self.assertEqual('sat',row['result'],row)
            self.assertEqual('equivalent_on_reduced_domain',row['verification'],row)
        with self.assertRaises(ValueError):
            constant_key_check('affine_a',fn=lambda x:2*x)

    def test_single_mad_discovered_controls_and_custom_target_guard(self):
        for sequence in (['mad','^','&'],['|','mad','^']):
            row=live_search(sequence,11,5000,target_fn=control_function(sequence,11))
            self.assertEqual('sat',row['result'],row)
            self.assertEqual('unsat',row['verification'],row)
        with self.assertRaises(ValueError):
            live_search(['mad','mad','^'],control=True,target_fn=lambda x:x)

    def test_final_role_partition_and_discovered_controls(self):
        roles=middle_roles()
        self.assertEqual(58,len(set(roles)))
        pool=(0,1,2,4,5,6,7)
        expected={(*pair,c) for pair in itertools.combinations_with_replacement(pool,2)
                  for c in pool if 5 in (*pair,c) and any(i in (0,1,2,4) for i in (*pair,c))}
        self.assertEqual(expected,set(roles))
        for op in ('^','&','|'):
            def fn(x):
                _,a,p,_=live_words(x,11)
                return 9*bitwise(op,33*x+17,a)+p
            row=live_search(['mad',op,'mad'],11,5000,target_fn=fn,
                            last_mad_role=(5,6,2),middle_uses_first=True,
                            middle_other=1 if op=='|' else None)
            self.assertEqual('sat',row['result'],row)
            self.assertEqual('unsat',row['verification'],row)
        for kwargs in ({'last_mad_role':(6,5,2)}, {'last_mad_role':(0,1,8)},
                       {'middle_uses_first':True}):
            with self.assertRaises(ValueError):
                live_search(['mad','mad','^'],**kwargs)
        with self.assertRaises(ValueError):
            live_search(['mad','^','mad'],middle_other=5)


class LiveResidualTests(unittest.TestCase):
    def test_zero_input_reductions_against_brute_force(self):
        width,modulus=3,8
        for kind in ('b*t+u','b*u+t','D*b+u'):
            def fn(x):
                t=3*x+5
                u=(t%modulus)^6
                return x*t+u if kind=='b*t+u' else x*u+t if kind=='b*u+t' else 3*x+u
            report=enumerate_xor(kind,width,fn,chunk_size=7)
            self.assertFalse(report['excluded'])
            for row in report['rows']:
                name,indices=row['family'],row['indices']
                expected=[]
                for rank in range(parameter_count(width,name)):
                    a,b=scalar_parameters(rank,width,name)
                    ts=[operand(x,width,name,indices,a,b)%modulus for x in range(modulus)]
                    for key in range(modulus):
                        for d in range(modulus) if kind=='D*b+u' else (0,):
                            ys=[x*t+(t^key) if kind=='b*t+u' else x*(t^key)+t
                                if kind=='b*u+t' else d*x+(t^key) for x,t in enumerate(ts)]
                            if all((y-fn(x))%modulus==0 for x,y in enumerate(ys)):
                                expected.append([a,b,key,d])
                self.assertEqual(sorted(expected),sorted(row['survivors_A_B_K_D']),(kind,row))

    def test_mask_forcing_against_scalar_oracle(self):
        width,mask=4,15
        fn=lambda x:5*x+((3*x+7)&10)
        for op in ('&','|'):
            report=force_mask(op,width,2,fn)
            expected=[]
            for d in range(mask+1):
                values=[(fn(x)-d*x)&mask for x in range(mask+1)]
                forced=0
                for y in values:
                    forced |= y if op=='&' else mask^y
                if forced&12 != 12:
                    expected.append(d)
            self.assertEqual(expected,report['exceptional_coefficients'])

    def test_masked_first_positive_control_and_exact_oracle(self):
        width,mask=3,7
        fn=lambda x:3*x+((5*live(x,width)[1]+3)&6)
        report=masked_first_mad(3,width,fn,chunk_size=7)
        self.assertFalse(report['excluded'])
        for row in report['rows']:
            expected=[]
            name,indices=row['family'],row['indices']
            for rank in range(row['tested']):
                a,b=scalar_parameters(rank,width,name)
                ts=[operand(x,width,name,indices,a,b)&mask for x in range(mask+1)]
                if any(all((ts[x]&key)==((fn(x)-3*x)&mask) for x in range(mask+1))
                       for key in range(mask+1)):
                    expected.append(rank)
            self.assertEqual(expected,row['survivor_parameter_ids'])

    def test_tonly_discovered_positive_controls(self):
        for kind in ('D*t+u','D*u+t'):
            for name in ('affine_a','mixed_ba','mixed_ab'):
                coefficient=34 if name.startswith('mixed') else 33
                def fn(x):
                    t=operand(x,8,name,(),coefficient,0)
                    u=t^42
                    return 2*t+u if kind=='D*t+u' else 2*u+t
                row=check_tonly(kind,name,8,fn=fn)
                self.assertEqual('sat',row['result'],row)
                self.assertEqual('equivalent_on_reduced_domain',row['verification'],row)

    def test_scaled_xor_low_bit_period_lemma(self):
        self.assertTrue(scaled_xor_input_certificate()['excluded'])
        rng=random.Random(865)
        qualifying=0
        for valuation in (0,1):
            r=3-valuation
            for _ in range(2000):
                c0,c1,key=[rng.randrange(2048) for _ in range(3)]
                role=[rng.randrange(5) for _ in range(3)]
                scale=(2*rng.randrange(1<<(10-valuation))+1)<<valuation
                def t(x):
                    pool=[*live(x,11),c0,c1]
                    return (pool[role[0]]*pool[role[1]]+pool[role[2]])&2047
                if len({t(x)% (1<<r) for x in range(8)})==1:
                    qualifying+=1
                    self.assertEqual(0,(scale*((t(264)^key)-(t(8)^key)))&2047)
        self.assertGreater(qualifying,100)


class FirstAffineShiftTests(unittest.TestCase):
    def test_complete_75_shape_partition(self):
        self.assertEqual(75,len(set(shapes())))
        self.assertEqual(Counter({'quadratic':40,'shift_cardinality':12,
            'bilinear_parity':10,'square_shift_intervals':2,'scaled_shift_intervals':4,
            'zero_input_range':2,'square_sum':1,'scaled_affine_sum_or_top_pair':4}),
            Counter(classify(pair,c) for pair,c in shapes()))

    def test_even_affine_congruence_with_direct_x(self):
        self.assertEqual(31,len(even_multiplier_certificate()['period_witnesses']))
        rng = random.Random(864)
        for width in range(2,9):
            modulus = 1 << width
            for _ in range(100):
                a = 2*rng.randrange(modulus//2)
                b,k0,k1,x = [rng.randrange(modulus) for _ in range(4)]
                q = modulus//(a & -a) if a else 2
                shift = rng.randrange(1,width)
                pair,c = rng.choice(shapes())
                def program(value):
                    t = (a*value+b)%modulus
                    pool = [value,t,t>>shift,k0,k1]
                    return (pool[pair[0]]*pool[pair[1]]+pool[c])%modulus
                self.assertEqual(0,(program((x+q)%modulus)-program(x))%q)

    def test_bilinear_parity_and_square_shift(self):
        for width in range(2,6):
            modulus = 1 << width
            for a in range(1,modulus,2):
                for b in range(modulus):
                    for s in range(1,width):
                        pools = [[x,(a*x+b)%modulus,((a*x+b)%modulus)>>s,0,1]
                                 for x in range(modulus)]
                        for pair,c in shapes():
                            reason = classify(pair,c)
                            if reason not in ('bilinear_parity','square_shift_intervals'):
                                continue
                            ys = [(p[pair[0]]*p[pair[1]]+p[c])%modulus for p in pools]
                            if reason == 'bilinear_parity':
                                self.assertIn(sum(y&1 for y in ys),(modulus//4,3*modulus//4))
                            else:
                                self.assertLess(len(set(ys)),modulus)

    def test_scaled_shift_interval_necessity(self):
        for width in range(2,5):
            modulus = 1 << width
            for a in range(1,modulus,2):
                for b,k,s in itertools.product(range(modulus),range(modulus),range(1,width)):
                    for add_original in (False,True):
                        ys = []
                        for x in range(modulus):
                            t = (a*x+b)%modulus
                            ys.append(((x if add_original else t)+k*(t>>s))%modulus)
                        if len(set(ys)) == modulus:
                            self.assertEqual(0,k%(1<<s))

    def test_sum_and_top_pair_formulas(self):
        for width in range(2,9):
            modulus,half = 1<<width,1<<(width-1)
            self.assertEqual(half,sum(x*x for x in range(modulus))%modulus)
            for s in range(1,width):
                self.assertEqual(half,sum(x>>s for x in range(modulus))%modulus)
                for a,b,k in ((3,5,2),(1,0,4),(modulus-1,modulus-1,6)):
                    for original in (True,False):
                        def program(x):
                            t=(a*x+b)%modulus
                            return (k*(x if original else t)+(t>>s))%modulus
                        diffs = {(program(x^half)-program(x))%modulus for x in range(modulus)}
                        self.assertLessEqual(diffs,{half>>s,(-half>>s)%modulus})
        x=z3.BitVec('paired_x',32)
        def symbolic(value):
            a=4097*value+0x7ed55d16
            return a^z3.LShR(a,19)^0xc761c23c
        solver=z3.SolverFor('QF_BV')
        solver.add(symbolic(x^0x80000000) != symbolic(x)^0x80001000)
        self.assertEqual(z3.unsat,solver.check())

    def test_variable_count_unsat_and_discovered_positive_controls(self):
        self.assertEqual('unsat',variable_count_query()['result'])
        for fn in (lambda x:7*x*x+11*x+19,
                   lambda x:(0xabcdef01^x) if x<32 else 7*x*x+11*x+19):
            result=variable_count_query(fn=fn)
            self.assertEqual('sat',result['result'],result)
            model=result['relaxation_model']
            for x in result['samples']:
                self.assertTrue(((model['a']*x+model['b'])&0xffffffff)<32 or
                    ((model['c2']*x*x+model['c1']*x+model['c0']-fn(x))&0xffffffff)==0)
        # Dropping A!=0 admits a=0, making the relaxation vacuous. This is why
        # the constant-count branch must be handled by a separate proof.
        self.assertEqual('sat',variable_count_query(nonzero=False)['result'])


if __name__ == '__main__':
    unittest.main()
