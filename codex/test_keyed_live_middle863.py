"""Check the legal full32 subdomain, actual ISA semantics and search controls."""
from pathlib import Path
from itertools import islice
import random
import sys
import unittest

import z3

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from frozen_problem import Machine,myhash

from codex.search_keyed_live_middle863 import (C0,C1,MASK,OPS,execute,expression,
    live_values,original_program,reference,search)
from codex.screen_keyed_live_rectangles863 import rectangles,search as rectangle_search
from codex.search_head_merge863 import (FAMILIES,evaluate as head_evaluate,
    symbolic as head_symbolic,normalize_output,search as head_search)
from codex.test_free_middle_keys863 import frozen as frozen_single
from codex.search_xor_equivalent_middle863 import search as key_search
from codex.scan_keyed_live_frontier863 import canonical_role,partitions
from codex.search_keyed_affine_bypass863 import (evaluate as bypass_evaluate,
    normalize as normalize_bypass,search as bypass_search)


def head_rows(coefficients,family):
    a,b,d,e=coefficients[:4]
    rows=[dict(op='mad',operands=[0,1,2],constants=[a,b])]
    if family in ('parallel','triple_xor'):
        rows += [dict(op='mad',operands=[0,2,3],constants=[d,e]),
                 dict(op='^',operands=[1,2,0],constants=[0,0])]
        rows += [dict(op='mad',operands=[3,4,5],constants=coefficients[4:])
                 if family=='parallel'else dict(op='^',operands=[0,3,0],constants=[0,0])]
    else:
        source,tail=family.split('_')
        rows += [dict(op='^',operands=[1,0 if source=='input'else 2,0],
                      constants=[coefficients[4],0]),
                 dict(op='mad',operands=[2,3,4],constants=[d,e]),
                 dict(op='^',operands=[3,dict(input=0,first=1,middle=2)[tail],0],constants=[0,0])]
    return rows


def frozen(a,key,rows):
    machine=Machine([],[],None)
    core=machine.cores[0]
    values=[a,((a-C0)*pow(4097,-1,1 << 32))&MASK,a^(a>>19)^key]
    for row in rows:
        pool=values+row['constants']
        x,y,z=[pool[j]for j in row['operands']]
        core.scratch[:24]=[x]*8+[y]*8+[z]*8
        machine.scratch_write={}
        if row['op']=='mad':machine.valu(core,'multiply_add',24,0,8,16)
        else:machine.alu(core,row['op'],24,0,8)
        values.append(machine.scratch_write[24])
    return values[-1]


class KeyedLiveTests(unittest.TestCase):
    def test_full32_reference_and_four_instruction_program(self):
        rng=random.Random(86452)
        for a in [0,1,0x7ffff,0x80000,0x80001,MASK]+[rng.getrandbits(32)for _ in range(200)]:
            prior=((a-C0)*pow(4097,-1,1 << 32))&MASK
            self.assertEqual(a,(4097*prior+C0)&MASK)
            h=myhash(prior)^0xb55a4f09
            expected=h^(h>>16)
            self.assertEqual(expected,reference(a,32)&MASK)
            self.assertEqual(expected,frozen(a,C1,original_program()))

    def test_low_domain_is_actual_full32_subdomain(self):
        rng=random.Random(86453)
        for width in (2,3,4,5,8,11,13,16):
            mask=(1 << width)-1
            xs=range(1 << width)if width<=5 else [0,mask]+[rng.randrange(1 << width)for _ in range(60)]
            for a in xs:
                key=rng.getrandbits(32)
                expected=[v&mask for v in live_values(a,key,32)]
                self.assertEqual(expected,live_values(a,key&mask,width))
                symbolic=live_values(z3.BitVecVal(a,width),z3.BitVecVal(key&mask,width),width)
                self.assertEqual(expected,[z3.simplify(v).as_long()for v in symbolic])
                self.assertEqual(reference(a,32)&mask,reference(a,width)&mask)
                self.assertEqual(reference(a,32)&mask,
                    z3.simplify(reference(z3.BitVecVal(a,width),width)).as_long())

    def test_random_candidate_programs_against_frozen_isa(self):
        rng=random.Random(86454)
        for _ in range(100):
            rows=[dict(op=rng.choice(OPS),operands=[rng.randrange(i+5)for _ in range(3)],
                       constants=[rng.getrandbits(32)for _ in range(2)])for i in range(3)]
            key=rng.getrandbits(32)
            for a in (0,MASK,rng.getrandbits(32)):
                expected=frozen(a,key,rows)
                self.assertEqual(expected,execute(a,key,rows,32))
                self.assertEqual(expected,z3.simplify(expression(z3.BitVecVal(a,32),key,rows,32)).as_long())

    def test_automatic_control_is_exhaustive_but_reduced_only(self):
        result=search(('mad','mad','^'),5,10,9000,control=True)
        self.assertEqual('equivalent_on_restricted_low_domain',result['result'],result)
        self.assertEqual(32,result['verified_inputs'])
        self.assertFalse(result['new_executable'])
        for a in range(32):
            self.assertEqual((reference(a,5,True)^result['output_key'])&31,
                execute(a,result['input_key'],result['program'],5))

    def test_full32_positive_search_control(self):
        result=search(('mad','mad','^','mad'),32,10,9000,pin_original=True)
        self.assertEqual('proved_equivalent_full32',result['result'],result)
        self.assertFalse(result['new_executable'])

    def test_rectangles_against_independent_output_key_enumeration(self):
        for width in (4,6):
            mask=(1 << width)-1
            rects=list(rectangles(width))
            for bypass in (None,'a','prior'):
                report=rectangle_search(width,bypass)
                survivors=[]
                for key in range(1 << width):
                    good=True
                    for row in rects:
                        yy=[]
                        for a in row['a_values']:
                            prior=((a-C0)*pow(4097,-1,1 << 32))&MASK
                            tail=0 if bypass is None else a if bypass=='a'else prior
                            yy.append((reference(a,32)^key^tail)&mask)
                        if (yy[3]-yy[1]-yy[2]+yy[0]) % row['divisor']:
                            good=False;break
                    if good:survivors.append(key)
                self.assertEqual(len(survivors),report['remaining_output_keys'])
                self.assertEqual(survivors[:16],report['survivor_examples'])
                self.assertTrue(rectangle_search(width,bypass,True)['zero_control_retained'])

    def test_rectangles_hold_for_arbitrary_keys_and_bitwise_prefixes(self):
        rng=random.Random(86455)
        for row in islice(rectangles(11),100):
            for _ in range(12):
                key,c,d=[rng.randrange(2048)for _ in range(3)]
                yy=[]
                for a,p in zip(row['a_values'],row['prior_values']):
                    q=((a^key)&p)|(a^c)
                    s=(q^p)&(a|d)
                    yy.append((q*q*p+c*s*a+d*p*p+q*s+17)&2047)
                self.assertEqual(0,(yy[3]-yy[1]-yy[2]+yy[0]) % row['divisor'])

    def test_head_families_against_full32_frozen_isa(self):
        rng=random.Random(86456)
        for family in FAMILIES:
            for _ in range(50):
                coefficients=[rng.getrandbits(32)for _ in range(6 if family=='parallel'else 4 if family=='triple_xor'else 5)]
                rows=head_rows(coefficients,family)
                for x in (0,MASK,rng.getrandbits(32)):
                    expected=frozen_single(x,rows)
                    self.assertEqual(expected,head_evaluate(x,coefficients,family,32))
                    self.assertEqual(expected,z3.simplify(head_symbolic(z3.BitVecVal(x,32),coefficients,family)).as_long())

    def test_head_output_key_normalizations_preserve_function(self):
        rng=random.Random(86457)
        for family in FAMILIES:
            for width in (3,5,11,32):
                mask,high=(1 << width)-1,1 << (width-1)
                for _ in range(40):
                    coeff=[rng.getrandbits(width)for _ in range(6 if family=='parallel'else 4 if family=='triple_xor'else 5)]
                    key=rng.getrandbits(width)
                    cv,ov=normalize_output(coeff,key,family,width)
                    self.assertEqual(0,ov&(high|1))
                    xs=range(1 << width)if width<=5 else(0,mask,rng.getrandbits(width))
                    for x in xs:
                        self.assertEqual(head_evaluate(x,coeff,family,width)^key,
                            head_evaluate(x,cv,family,width)^ov)

    def test_all_eight_automatic_head_controls(self):
        for family in FAMILIES:
            report=head_search(8,10,control=True,family=family)
            self.assertEqual('equivalent_on_reduced_domain',report['result'],report)
            self.assertEqual(256,report['verified_inputs'])
            self.assertFalse(report['new_executable'])

    def test_fixed_endpoint_does_not_apply_free_key_symmetry(self):
        report=key_search(('mad','^','mad'),5,10,9000,
            fixed_input_key=17,fixed_output_key=0,first_form='affine',
            middle_source='key',last_operands=(2,3,4))
        self.assertEqual('equivalent_on_reduced_domain',report['result'],report)
        self.assertEqual(17,report['input_key'])
        self.assertEqual(0,report['output_key'])
        with self.assertRaises(ValueError):
            key_search(('mad','mad','^','mad'),5,pin_original=True,fixed_input_key=1)

    def test_final_mad_partition_covers_all_constant_aliases(self):
        roles={role for _,role in partitions()}
        self.assertEqual(50,len(roles))
        for a in range(7):
            for b in range(a,7):
                for c in range(7):
                    if 4 in (a,b,c):
                        self.assertIn(canonical_role((a,b,c)),roles)

    def test_affine_bypass_normalizations_match_frozen_full32(self):
        rng=random.Random(86458)
        for width in (3,5,11,32):
            mask,high=(1 << width)-1,1 << (width-1)
            for _ in range(100):
                cv=[rng.getrandbits(width)for _ in range(6)]
                ov=rng.getrandbits(width)
                nc,no=normalize_bypass(cv,ov,width)
                for v in (*nc[4:],no):self.assertEqual(0,v&(high|1))
                xs=range(1 << width)if width<=5 else(0,mask,rng.getrandbits(width))
                for x in xs:
                    self.assertEqual(bypass_evaluate(x,cv,width)^ov,
                                     bypass_evaluate(x,nc,width)^no)
                    if width==32:
                        a,b,d,e,ik,key=cv
                        rows=[dict(op='^',operands=[0,1,0],constants=[ik,0]),
                              dict(op='mad',operands=[1,2,3],constants=[a,b]),
                              dict(op='^',operands=[2,3,0],constants=[key,0]),
                              dict(op='mad',operands=[0,4,5],constants=[d,e]),
                              dict(op='mad',operands=[3,5,4],constants=[1,0])]
                        self.assertEqual(frozen_single(x,rows),bypass_evaluate(x,cv,32))

    def test_affine_bypass_positive_controls_for_both_parities(self):
        for parity in (0,1):
            report=bypass_search(8,10,parity,True)
            self.assertEqual('equivalent_on_relaxed_low_domain',report['result'],report)
            self.assertEqual(256,report['verified_inputs'])
            self.assertFalse(report['new_executable'])

    def test_exact_bypass_constraints_and_controls_survive_normalization(self):
        rng=random.Random(86459)
        for source in ('a','prior'):
            for _ in range(60):
                coeff=[rng.randrange(2048)for _ in range(6)]
                coeff[3]=(0 if source=='a'else -coeff[2]*C0)&2047
                norm,key=normalize_bypass(coeff,rng.randrange(2048),11)
                self.assertEqual(norm[3],(0 if source=='a'else -norm[2]*C0)&2047)
            for parity in (0,1):
                report=bypass_search(8,10,parity,True,source)
                self.assertEqual('equivalent_on_relaxed_low_domain',report['result'],report)
                self.assertEqual(256,report['verified_inputs'])


if __name__=='__main__':unittest.main()
