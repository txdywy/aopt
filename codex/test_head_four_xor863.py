"""Validate complete XOR coverage and the new AND/OR exploration boundaries."""
from itertools import combinations_with_replacement
import json
from pathlib import Path
import random
import unittest

from codex.audit_head_four_xor863 import audit
from codex.scan_head_final_quadratic863 import AFFINE, canonical_second, cases, fixed_roles, first_is_live
from codex.scan_head_single_bitwise863 import cases as bitwise_cases
from codex.scan_head_and_module863 import canonical_keyed
from codex.search_xor_equivalent_middle863 import execute, search
from codex.test_free_middle_keys863 import frozen

ROOT=Path(__file__).resolve().parent


class FourXorClosureTests(unittest.TestCase):
    def test_complete_audit_and_refinement_controls(self):
        r=audit()
        self.assertEqual((10,0),(r['closed_orders'],r['open_orders']))
        self.assertEqual(110,r['direct_refinement_queries'])
        self.assertEqual(104,r['live_refinement_queries'])
        self.assertEqual(6,r['dead_first_refinement_queries'])
        self.assertEqual(480,sum(r['second_role_coverage'].values()))
        prior=json.loads((ROOT/'head_affine_frontier863_audit.json').read_text())
        for case in cases(prior):
            fixed=fixed_roles(*case)
            rows=[dict(op=op,operands=list(fixed[i]),constants=constants)
                for i,(op,constants) in enumerate(zip(('mad','mad','^','mad'),
                    ([5,7],[11,19],[13,23],[17,29])))]
            result=search(('mad','mad','^','mad'),5,5,4500,first_form='affine',
                          fixed_operands=fixed,control_program=rows)
            if not first_is_live(*case):
                self.assertEqual('projected_grammar_unsat',result['result'])
                continue
            self.assertEqual('equivalent_on_reduced_domain',result['result'],result)
            for x in range(32):
                self.assertEqual(frozen(x,rows)&31,execute(x,result['program'],5))

    def test_every_second_role_reduction_at_full32(self):
        rng=random.Random(864387)
        mask=0xffffffff
        count=0
        for pair in combinations_with_replacement(range(4),2):
            for addend in range(4):
                role=(*pair,addend)
                kind,canonical=canonical_second(role)
                for _ in range(20):
                    a,b,c,d=[rng.getrandbits(32) for _ in range(4)]
                    def original(x):
                        pool=(x,(a*x+b)&mask,c,d)
                        return (pool[role[0]]*pool[role[1]]+pool[role[2]])&mask
                    if kind=='affine':
                        self.assertEqual(AFFINE,canonical)
                        constants=[(original(1)-original(0))&mask,original(0)]
                    else:
                        constants=[d if addend==3 else c,0]
                    for x in (0,mask,rng.getrandbits(32)):
                        pool=(x,(a*x+b)&mask,*constants)
                        value=(pool[canonical[0]]*pool[canonical[1]]+pool[canonical[2]])&mask
                        self.assertEqual(original(x),value,(role,canonical))
                    count+=1
        self.assertEqual(800,count)

    def test_affine_second_and_arbitrary_final_mad_is_bivariate_quadratic(self):
        rng=random.Random(864388)
        prior=json.loads((ROOT/'head_affine_frontier863_audit.json').read_text())
        mask=0xffffffff
        for item in prior['remaining_final_templates']:
            if item['source']=='first':continue
            role=item['last_operands']
            for _ in range(100):
                a,b,d,e,k,c,f,x=[rng.getrandbits(32) for _ in range(8)]
                t,p=(a*x+b)&mask,(d*x+e)&mask
                z=p^(x if item['source']=='input' else k)
                pool=(x,t,p,z,c,f)
                actual=(pool[role[0]]*pool[role[1]]+pool[role[2]])&mask
                forms=((0,1,0),(b,a,0),(e,d,0),(0,0,1),(c,0,0),(f,0,0))
                # Independent monomial convolution in formal variables x,z.
                coefficients={}
                basis=((0,0),(1,0),(0,1))
                for ci,ei in zip(forms[role[0]],basis):
                    for cj,ej in zip(forms[role[1]],basis):
                        exponent=tuple(i+j for i,j in zip(ei,ej))
                        coefficients[exponent]=coefficients.get(exponent,0)+ci*cj
                for ci,ei in zip(forms[role[2]],basis):
                    coefficients[ei]=coefficients.get(ei,0)+ci
                expected=sum(v*x**i*z**j for (i,j),v in coefficients.items())&mask
                self.assertEqual(actual,expected)
                self.assertTrue(all(i+j<=2 for i,j in coefficients))

    def test_all_single_bitwise_partitions_accept_known_programs(self):
        self.assertEqual(20,len(bitwise_cases()))
        for order,first in bitwise_cases():
            fixed={}
            rows=[]
            for i,op in enumerate(order):
                if i==0 and op=='mad':
                    role={'affine':(0,1,2),'square_const':(0,0,1),'square_x':(0,0,0)}[first]
                else:
                    role=(i,i+1,i+2) if op=='mad' else (i,i+1,0)
                fixed[i]=role
                rows.append(dict(op=op,operands=list(role),constants=[5+2*i,7+4*i]))
            result=search(order,5,5,4500,first_form=first,fixed_operands=fixed,control_program=rows)
            self.assertEqual('equivalent_on_reduced_domain',result['result'],result)
            for x in range(32):
                self.assertEqual(frozen(x,rows)&31,execute(x,result['program'],5))

    def test_and_key_normalization_and_or_module_identity(self):
        rng=random.Random(864389)
        for trial in range(1000):
            a,b,k=[rng.randrange(2048) for _ in range(3)]
            aa,bb,kk=canonical_keyed(a,b,k)
            sign=(-1 if (a&1023)>512 else 1)*(-1 if k&1024 else 1)
            old=lambda x:((a*x+b)&2047)&k
            new=lambda x:((aa*x+bb)&2047)&kk
            intercept=(old(0)-sign*new(0))&2047
            slope=(old(1)-sign*new(1)-intercept)&2047
            points=range(2048) if trial<12 else [rng.randrange(2048) for _ in range(16)]
            for x in points:
                self.assertEqual(old(x),(sign*new(x)+slope*x+intercept)&2047)
            self.assertLess(kk,1024)
            x,t,k=[rng.getrandbits(32) for _ in range(3)]
            for other in (x,k):
                q=t&other
                self.assertEqual(t|other,(t+other-q)&0xffffffff)
                # The affine substitution is involutive; polynomial degree
                # is preserved in either direction.
                self.assertEqual(q,(t+other-(t|other))&0xffffffff)


if __name__=='__main__':
    unittest.main()
