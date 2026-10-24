"""Finite oracles and independent controls for the new SHR certificates."""
import itertools
import json
from pathlib import Path
import random
import unittest

import z3

from codex.prove_middle_shift863 import (MASK, binary, classify_even, classify_odd,
    coverage, final_mask_certificate, finite_witnesses, middle, prefix_orders, prefix_query, shapes,
    target, top_pair_query, variable_count_query, violating_rectangles)
from codex.screen_shift_middle863 import full_shift
from codex.search_middle_shift_orders863 import command


def evaluate_role(width, a, b, shift, k0, k1, role):
    mask = (1 << width)-1
    values = []
    for x in range(mask+1):
        t = (a*x+b) & mask
        pool = [x,t,t >> shift,k0,k1]
        i,j,k = role
        values.append((pool[i]*pool[j]+pool[k]) & mask)
    return values


def oracle_prefix(order, width=2):
    """Enumerate real programs without SMT selectors or symmetry cuts."""
    modulus,mask = 1 << width,(1 << width)-1
    x = tuple(range(modulus))
    constants = [tuple([v]*modulus) for v in range(modulus)]
    firsts = set()
    for operands in itertools.product([x,*constants],repeat=3 if order[0]=='mad' else 2):
        firsts.add(tuple(binary(order[0],*(a[i] for a in operands)) & mask
                         for i in range(modulus)))
    represented = set([x,*constants,*firsts])
    for first in firsts:
        for operands in itertools.product([x,first,*constants],
                                           repeat=3 if order[1]=='mad' else 2):
            represented.add(tuple(binary(order[1],*(a[i] for a in operands)) & mask
                                  for i in range(modulus)))
    return represented


class MiddleShiftTests(unittest.TestCase):
    def test_operand_partition_and_fixed_witnesses(self):
        cert = finite_witnesses()
        self.assertEqual(75,len(set(shapes())))
        self.assertEqual(75,sum(cert['odd_shapes'].values()))
        self.assertEqual(75,sum(cert['even_shapes'].values()))
        self.assertEqual(30,len(cert['nonconstant_power_of_two_differences']))
        self.assertEqual(45,len(cert['large_shift_quotient_witnesses']))
        for row in cert['nonconstant_power_of_two_differences']:
            actual = [(target((x+row['step'])&MASK)-target(x))&MASK for x in row['inputs']]
            self.assertEqual(actual,row['differences'])
            self.assertNotEqual(*actual)
        # The r=1,s=2 argument works modulo2**31, a stricter witness check.
        actual = cert['nonconstant_power_of_two_differences'][0]['differences']
        self.assertNotEqual(actual[0]%(1 << 31),actual[1]%(1 << 31))
        for row in cert['large_shift_quotient_witnesses']:
            self.assertEqual(0,row['difference']%row['period'])
            self.assertGreater(row['quotient'],row['largest_possible_q_plus_one'])

    def test_target_top_pair_mod4_and_explicit_inverse(self):
        self.assertEqual('unsat',top_pair_query()['result'])
        x = z3.BitVec('target_mod4_x',32)
        solver = z3.SolverFor('QF_BV')
        solver.add((middle(x)-x)&3 != 2)
        self.assertEqual(z3.unsat,solver.check())
        rng = random.Random(865)
        # u XOR (512*u+K) has a unit bit diagonal. Four fixed-point steps
        # recover32 bits (nine more bits per step), giving a full inverse.
        key = (0xaccf6200-512*0xe9f8cc1d)&MASK
        for value in [0,1,MASK,1 << 31]+[rng.getrandbits(32) for _ in range(1000)]:
            z = (value-0xfd7046c5)*pow(9,-1,1 << 32)&MASK
            u = 0
            for _ in range(4):
                u = z^((512*u+key)&MASK)
            inverse = (u-0xe9f8cc1d)*pow(33,-1,1 << 32)&MASK
            self.assertEqual(value,target(inverse))

    def test_odd_multiplier_lemmas_against_complete_small_domains(self):
        width,modulus,half = 4,16,8
        for a,b,shift,(k0,k1) in itertools.product(range(1,16,2),(0,1,3,7,15),
                range(1,4),((0,0),(1,2),(3,7),(4,8),(15,14))):
            for role in shapes():
                ys = evaluate_role(width,a,b,shift,k0,k1,role)
                category = classify_odd(role)
                compatible = all(ys[x^half] == ys[x]^half for x in range(modulus))
                if category == 'q_only_cardinality':
                    self.assertLess(len(set(ys)),modulus)
                elif category in ('q_times_odd_diagonal','q_squared_plus_odd_diagonal',
                                  'polynomial_plus_q'):
                    self.assertFalse(compatible,(a,b,shift,k0,k1,role))
                elif category == 'scaled_q_plus_odd_diagonal' and compatible:
                    other = role[1] if role[0] == 2 else role[0]
                    scale = (k0,k1)[other-3]
                    self.assertEqual(0,scale%(1 << (shift+1)))
                    deltas = {(ys[(x+(1 << shift))%modulus]-ys[x])%modulus
                              for x in range(modulus)}
                    self.assertEqual(1,len(deltas))

    def test_even_multiplier_lemmas_against_complete_small_domains(self):
        width,modulus,half,quarter = 4,16,8,4
        for a,b,shift,(k0,k1) in itertools.product(range(2,16,2),(0,1,3,7,15),
                range(1,4),((0,0),(1,2),(3,7),(4,8),(15,14))):
            r = (a & -a).bit_length()-1
            period = 1 << (width-r)
            for role in shapes():
                ys = evaluate_role(width,a,b,shift,k0,k1,role)
                category = classify_even(role)
                compatible = all(ys[x^half] == ys[x]^half for x in range(modulus))
                if category == 'periodic_without_x':
                    self.assertTrue(all(ys[(x+period)%modulus] == ys[x] for x in range(modulus)))
                elif category == 'q_times_x' and compatible:
                    self.assertGreater(r,shift)
                elif category == 'x_squared_plus_q':
                    self.assertFalse(compatible)
                elif category in ('q_times_t_plus_x','q_squared_plus_x','scaled_q_plus_x',
                                  'x_times_t_plus_q','scaled_x_plus_q'):
                    deltas = {(ys[(x+period)%modulus]-ys[x])%modulus for x in range(modulus)}
                    self.assertEqual(1,len(deltas))
                if r != 1:
                    continue
                quarter_compatible = all((ys[(x+quarter)%modulus]-ys[x])%quarter == 0
                                         for x in range(modulus))
                if category in ('q_times_t_plus_x','q_squared_plus_x'):
                    self.assertFalse(all((y-x)%4 == 2 for x,y in enumerate(ys)))
                elif category == 'scaled_q_plus_x' and quarter_compatible:
                    other = role[1] if role[0] == 2 else role[0]
                    scale = (k0,k1)[other-3]
                    self.assertEqual(0,scale%(1 << (shift-1)))
                elif category in ('x_times_t_plus_q','scaled_x_plus_q') and shift >= 2:
                    self.assertFalse(quarter_compatible)

    def test_affine_shift_projection_identity_and_surviving_controls(self):
        for a,b,s,x in itertools.product(range(16),range(16),range(1,4),range(16)):
            if a%(1 << s) == 0:
                q = ((a*x+b)%16) >> s
                self.assertEqual(q,((a >> s)*x+(b >> s))%(1 << (4-s)))
        # These are actual compatible permutations in the allowed grammar.
        # They survive the structural lemmas; F-specific witnesses exclude F.
        for s in (1,2,3):
            ys = evaluate_role(8,3,5,s,1 << (s+1),0,(2,3,0))
            self.assertEqual(256,len(set(ys)))
            for bit in range(8):
                step = 1 << bit
                self.assertTrue(all((ys[(x+step)%256]-ys[x])%step == 0 for x in range(256)))

    def test_rectangle_query_and_polynomial_oracle(self):
        rows = violating_rectangles()
        self.assertEqual(211,len(rows))
        rng = random.Random(866)
        for row in rows:
            modulus = 1 << row['width']
            self.assertEqual(modulus,row['steps'][0]*row['steps'][1])
            self.assertNotEqual(0,row['mixed_difference'])
            coefficients = [rng.getrandbits(32) for _ in range(12)]
            ys = [sum(c*pow(x,i,modulus) for i,c in enumerate(coefficients))%modulus
                  for x in row['inputs']]
            self.assertEqual(0,(ys[3]-ys[2]-ys[1]+ys[0])%modulus)
        self.assertEqual('unsat',variable_count_query()['result'])
        self.assertEqual('sat',variable_count_query(nonzero=False)['result'])
        fn = lambda x: (x+full_shift(x,((1 << 31)*x+9)&MASK))&MASK
        control = variable_count_query(fn=fn,fixed_affine=(1 << 31,9))
        self.assertEqual('sat',control['result'])
        self.assertGreater(len(control['rectangles']),0)

    def test_prefix_solver_against_independent_two_bit_enumeration(self):
        rng = random.Random(867)
        universe = set(itertools.product(range(4),repeat=4))
        for order in prefix_orders():
            represented = oracle_prefix(order)
            positive = rng.sample(sorted(represented),min(8,len(represented)))
            negative = rng.sample(sorted(universe-represented),min(8,len(universe-represented)))
            for values in positive+negative:
                row = prefix_query(order,width=2,timeout_ms=1000,fn=lambda x:values[x])
                self.assertEqual('sat' if values in represented else 'unsat',row['result'],(order,values,row))

    def test_prefix_unpinned_controls_and_high_output_cover(self):
        for order,fn in ((('mad','^'),lambda x:(3*x+5)^x),
                         (('&','mad'),lambda x:3*(x&0x36)+5),
                         (('mad','mad'),lambda x:(3*x+5)*7+x)):
            row = prefix_query(order,fn=fn)
            self.assertEqual('sat',row['result'],row)
            self.assertEqual(2048,row['verified_inputs'])
        for low in range(2048):
            x = low if target(low) & (1 << 31) else low^(1 << 31)
            self.assertEqual(low,x&2047)
            self.assertTrue(target(x)&(1 << 31))
            self.assertEqual(target(low)&2047,target(x)&2047)

    def test_coverage_keeps_raw_unknown_and_mad_lowering_is_exact(self):
        path = Path(__file__).with_name('middle_shift_closure863_results.json')
        r = json.loads(path.read_text())
        refinement = json.loads(path.with_name('middle_shift_refinement863_results.json').read_text())
        c = coverage(True,r['square_first'],r['final_shift_prefixes'],refinement)
        self.assertEqual((36,12),(c['closed'],c['unresolved']))
        self.assertEqual(48,len(c['orders']))
        self.assertTrue(any(v['result']=='unknown' for v in r['final_shift_prefixes']))
        for a,b in itertools.product(range(16),repeat=2):
            self.assertEqual((a+b)%16,(a*1+b)%16)
            self.assertEqual((a-b)%16,(b*15+a)%16)
            self.assertEqual((a*b)%16,(a*b+0)%16)
        args = command(('mad','>>','^'),8000,64,32,True)
        self.assertIn('--high-bit-pairs',args)
        self.assertEqual('64',args[args.index('--consecutive-samples')+1])
        for kwargs in ({'width':1},{'width':32},{'timeout_ms':0}):
            with self.assertRaises(ValueError):
                prefix_query(('mad','^'),**kwargs)
        with self.assertRaises(ValueError):
            prefix_query(('mad','>>'))
        with self.assertRaises(ValueError):
            variable_count_query(timeout_ms=0)

    def test_final_bitwise_necessary_conditions(self):
        cert = final_mask_certificate()
        self.assertEqual(15,len(cert['roles']))
        self.assertEqual(31,len(cert['or_literal_witnesses']))
        for row in cert['or_literal_witnesses']:
            self.assertNotEqual(0,row['forbidden_or_bits'])
            self.assertNotEqual(row['target'],row['target'] | row['shifted'])
        x = cert['and_high_output_input']
        for s in range(1,33):
            self.assertNotEqual(target(x),target(x)&full_shift(x,s))
        xs = cert['variable_count_rectangle']
        ys = [target(x)&2047 for x in xs]
        self.assertEqual(1024,(ys[3]-ys[2]-ys[1]+ys[0])&2047)
        for x in xs:
            for k in (0,1,MASK):
                self.assertEqual(0,full_shift(k,x))
        # Independent small-word fact: AND/OR with a nonidentity constant
        # cannot be onto, even with an arbitrary input permutation.
        for w in range(2,7):
            m = (1 << w)-1
            for k in range(m+1):
                self.assertEqual(k==m,len({x&k for x in range(m+1)})==m+1)
                self.assertEqual(k==0,len({x|k for x in range(m+1)})==m+1)


if __name__ == '__main__':
    unittest.main()
