"""Independent checks for key normalization, masked ISA and rectangle screens."""
import random
import unittest

import numpy as np

from codex.search_absorb_restore863 import normalize_free_input, search
from codex.search_cross_encoding863 import middle, program, PRE_C5
from codex.search_masked_restore863 import evaluate, search as masked_search
from codex.screen_free_keys_polynomial863 import search as screen
from codex.test_free_middle_keys863 import frozen
from codex.search_affine_xor_keys863 import normalize_bypass, evaluate as affine_eval, search as affine_search
from codex.search_xor_equivalent_middle863 import search as partition_search, execute
from codex.audit_free_keys_lattice863 import audit as lattice_audit
from codex.audit_free_key_frontier863 import audit as complete_audit


class RestoreFrontierTests(unittest.TestCase):
    def test_final_affine_domination_reduction_exhaustively(self):
        # All endpoints/affine operands at four bits, including the positive
        # case of an odd affine operand equal to the target.
        size,mask = 16,15
        xs = np.arange(size,dtype=np.int32)
        targets = np.array([[middle(int(x)^ik)^ok for x in xs]
                            for ik in range(size) for ok in range(size)],dtype=np.int64)&mask
        matches = 0
        for a in range(size):
            for b in range(size):
                operand = (a*xs+b)&mask
                for op in ('&','|'):
                    good = np.all((targets&operand) == targets,axis=1) if op == '&' else \
                           np.all((targets|operand) == targets,axis=1)
                    if a&1:
                        expected = np.all(targets == operand,axis=1)
                        matches += int(np.sum(expected))
                    else:
                        expected = np.full(len(targets),a == 0 and b == (mask if op == '&' else 0))
                    self.assertTrue(np.array_equal(good,expected),(a,b,op))
        self.assertGreater(matches,0)

    def test_lattice_certificate_checks_low_domain_properties(self):
        result = lattice_audit()
        self.assertEqual(6,result['excluded_orders'])
        self.assertEqual(0,result['unresolved_orders'])
        self.assertFalse(result['new_executable'])

    def test_sole_affine_input_key_normalization(self):
        rng = random.Random(86436)
        for width in (3,5,11,32):
            mask,high = (1 << width)-1,1 << (width-1)
            for _ in range(80):
                a,b,k,d,ik = [rng.getrandbits(width)for _ in range(5)]
                na,nb,ni = a,b,ik
                if ni&1:
                    ni ^= mask
                    na,nb = -na,nb-na
                if ni&high:
                    ni ^= high
                    nb += na*high
                na,nb = na&mask,nb&mask
                self.assertEqual(ni&(high|1),0)
                for x in (range(1 << width)if width <= 5 else [0,mask,rng.getrandbits(width)]):
                    old_t,new_t = (a*(x^ik)+b)&mask,(na*(x^ni)+nb)&mask
                    self.assertEqual(old_t,new_t)
                    for bitwise in ('&','|'):
                        old_q,new_q = ((old_t&k,new_t&k)if bitwise == '&'else(old_t|k,new_t|k))
                        self.assertEqual((d*old_q+old_t)&mask,(d*new_q+new_t)&mask)

    def test_complete_unshifted_grammar_evidence(self):
        result = complete_audit()
        self.assertEqual(64,result['excluded_orders'])
        self.assertEqual({'projected_grammar_unsat':19,'unknown':8},result['one_mad_raw_counts'])
        self.assertEqual(4,result['minimum_instructions_over_endpoint_keys'])
        for op in ('^','&','|'):
            rows = [dict(op='^',operands=[0,1,0],constants=[7,0]),
                    dict(op='mad',operands=[0,1,2],constants=[9,0]),
                    dict(op=op,operands=[1,2,0],constants=[0,0])]
            control = partition_search(('^','mad',op),5,10,9000,control_program=rows)
            self.assertEqual('equivalent_on_reduced_domain',control['result'])
            self.assertEqual(32,control['verified_inputs'])
            for x in range(32):
                self.assertEqual(execute(x,rows,5),execute(x,control['program'],5))

    def test_bypass_normalization_preserves_exposed_function(self):
        rng = random.Random(86435)
        for width in (3,5,11,32):
            mask,high = (1 << width)-1,1 << (width-1)
            for _ in range(80):
                cv = [rng.getrandbits(width) for _ in range(4)]
                cv[0] &= mask-1
                cv[2] |= 1
                ik,ok = rng.getrandbits(width),rng.getrandbits(width)
                nc,ni,no = normalize_bypass(cv,ik,ok,width)
                for value in (ni,no,nc[1]):
                    self.assertEqual(value & (high|1),0)
                xs = range(1 << width) if width <= 5 else [0,1,mask,rng.getrandbits(width)]
                for x in xs:
                    self.assertEqual(affine_eval(x ^ ik,cv,'bypass',mask)^ok,
                                     affine_eval(x ^ ni,nc,'bypass',mask)^no)
        result = affine_search('bypass',8,10,9000,True)
        self.assertEqual(result['result'],'equivalent_on_reduced_domain',result)

    def test_fixed_partitions_recover_known_programs(self):
        first_forms = {'affine':(0,1,2),'square_const':(0,0,1),'square_x':(0,0,0)}
        second_forms = {'x_square':(0,0),'x_first':(0,1),'first_square':(1,1),
                        'scale_input':(0,2),'scale_first':(1,2)}
        tail_forms = {'scaled':(2,3,4),'x_product':(0,2,4),'first_product':(1,2,4),
                      'square':(2,2,4),'addend':(0,3,2)}
        controls = []
        for first,indices in first_forms.items():
            initial = dict(op='mad',operands=list(indices),constants=[7,11])
            for second,(a,b) in second_forms.items():
                rows = [initial,dict(op='mad',operands=[a,b,3],constants=[5,9]),
                        dict(op='^',operands=[1,2,0],constants=[0,0])]
                controls.append((rows,dict(first_form=first,second_form=second)))
        for source,indices in {'input':(0,1,0),'key':(1,2,0)}.items():
            for tail,last in tail_forms.items():
                rows = [dict(op='mad',operands=[0,1,2],constants=[7,11]),
                        dict(op='^',operands=list(indices),constants=[6,0]),
                        dict(op='mad',operands=list(last),constants=[5,9])]
                controls.append((rows,dict(first_form='affine',middle_source=source,tail_form=tail)))
        for bitwise in ('^','&','|'):
            for rows,kwargs in controls:
                program = [dict(row,op=bitwise if row['op'] == '^' else row['op']) for row in rows]
                result = partition_search(tuple(r['op'] for r in program),5,10,9000,
                                          control_program=program,**kwargs)
                self.assertEqual(result['result'],'equivalent_on_reduced_domain',(bitwise,kwargs,result))
                for x in range(32):
                    self.assertEqual(execute(x,program,5),execute(x,result['program'],5))
                self.assertFalse(result['new_executable'])

    def test_free_input_normalization_preserves_all_small_inputs(self):
        rng = random.Random(86431)
        for width in (3,5,11,32):
            mask,high = (1 << width)-1,1 << (width-1)
            for _ in range(80):
                cv = [rng.getrandbits(width) for _ in range(6)]
                cv[0] |= 1
                cv[2] &= mask-1
                cv[4] |= 1
                ik = rng.getrandbits(width)
                nc,ni = normalize_free_input(cv,ik,width)
                self.assertEqual(ni & (high|1),0)
                self.assertEqual(nc[0]&1,1)
                self.assertEqual(nc[2]&1,0)
                self.assertEqual(nc[1]&1,0)
                xs = range(1 << width) if width <= 5 else [0,1,mask,rng.getrandbits(width)]
                for x in xs:
                    self.assertEqual(program(x ^ ik,cv,width),program(x ^ ni,nc,width))

    def test_inverse_tail_key_matches_full32_fold(self):
        rng = random.Random(86432)
        for _ in range(300):
            value = rng.getrandbits(32)
            altered = value ^ PRE_C5
            self.assertEqual(altered ^ (altered >> 16),
                             value ^ (value >> 16) ^ 0xb55a4f09)

    def test_four_op_unknown_input_key_positive_control(self):
        result = search(8,10,fixed_multiplier=9,control=True)
        self.assertEqual(result['result'],'equivalent_on_reduced_domain',result)
        self.assertEqual(result['verified_inputs'],256)
        for x in range(256):
            expected = middle(x ^ result['input_key'] ^ 42)&255
            self.assertEqual(expected,program(x,result['coefficients'],8))
        self.assertFalse(result['new_executable'])

    def test_masked_programs_match_frozen_full32_isa(self):
        rng = random.Random(86433)
        for family in ('parallel','nested'):
            for _ in range(100):
                cv = [rng.getrandbits(32) for _ in range(5)]
                a,b,d,e,k = cv
                rows = [dict(op='mad',operands=[0,1,2],constants=[a,b])]
                if family == 'parallel':
                    rows += [dict(op='mad',operands=[0,2,3],constants=[d,e]),
                             dict(op='&',operands=[2,3,0],constants=[k,0]),
                             dict(op='^',operands=[1,3,0],constants=[0,0])]
                else:
                    rows += [dict(op='&',operands=[1,2,0],constants=[k,0]),
                             dict(op='^',operands=[0,2,0],constants=[0,0]),
                             dict(op='mad',operands=[3,4,5],constants=[d,e])]
                for x in (0,0xffffffff,rng.getrandbits(32)):
                    self.assertEqual(frozen(x,rows),evaluate(x,cv,family,0xffffffff))

    def test_masked_positive_controls_are_exhaustive_reduced_only(self):
        for family in ('parallel','nested'):
            result = masked_search(family,8,10,control=True)
            self.assertEqual('equivalent_on_reduced_domain',result['result'],result)
            self.assertEqual(256,result['verified_inputs'])
            self.assertFalse(result['new_executable'])
            control_cv = [297,109,66,22,252] if family == 'parallel' else [66,22,9,109,252]
            for x in range(256):
                self.assertEqual(evaluate(x ^ result['input_key'] ^ 42,control_cv,family,255),
                                 evaluate(x,result['coefficients'],family,255))

    def test_bypass_rectangles_match_independent_small_enumeration(self):
        for xi,cf in ((True,False),(False,True),(True,True)):
            result = screen(4,xor_input=xi,carry_free=cf)
            survivors = []
            for ik in range(16):
                for ok in range(16):
                    good = True
                    for r in result['rectangles']:
                        ys = []
                        for delta in (0,r['h'],r['k'],r['h']+r['k']):
                            x = (r['base']+delta)&15
                            ys.append((middle(x ^ ik)^ok^(x if xi else 0))&15)
                        if (ys[3]-ys[1]-ys[2]+ys[0])%r['divisor']:
                            good = False
                            break
                    if good:
                        survivors.append((ik,ok))
            self.assertEqual(len(survivors),result['remaining_pairs'])
            self.assertTrue(screen(6,affine_control=True,xor_input=xi,
                                   carry_free=cf)['zero_key_control_retained'])

    def test_carry_free_rectangles_hold_for_polynomials_in_bitwise_prefixes(self):
        rng = random.Random(86434)
        for width in (11,32):
            mask = (1 << width)-1
            for _ in range(300):
                i,j = rng.sample(range(width),2)
                h,k = 1 << i,1 << j
                base = rng.getrandbits(width)&~(h|k)
                key = rng.getrandbits(width)
                second_key,third_key = rng.getrandbits(width),rng.getrandbits(width)
                for op in ('xor','and','or'):
                    def value(x):
                        t = x ^ key if op == 'xor' else x & key if op == 'and' else x | key
                        s = (x|second_key)^third_key
                        return (3*x*x*t*t*t+7*x*t+5*t*t+11*t*s*s+1)&mask
                    ys = [value(base+d) for d in (0,h,k,h+k)]
                    self.assertEqual((ys[3]-ys[1]-ys[2]+ys[0])%min(h*k,1 << width),0)

    def test_all_key_pairs_fail_each_new_necessary_screen(self):
        for xi,cf in ((True,False),(False,True),(True,True)):
            result = screen(xor_input=xi,carry_free=cf)
            self.assertEqual(4194304,result['initial_key_pairs'])
            self.assertEqual(0,result['remaining_pairs'])
            self.assertFalse(result['new_executable'])


if __name__ == '__main__':
    unittest.main()
