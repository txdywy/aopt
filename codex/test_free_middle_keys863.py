"""Independent ISA and exhaustive reduced-domain checks for free XOR keys."""
from pathlib import Path
import random
import sys
import unittest

import z3

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from frozen_problem import Machine, myhash

from codex.search_cross_encoding863 import middle
from codex.search_full_live_shift863 import MASK, prior_for_b
from codex.search_xor_equivalent_middle863 import OPS, execute, expression, search
from codex.screen_free_keys_polynomial863 import search as polynomial_screen
from codex.search_affine_xor_keys863 import evaluate, normalize_nested, search as fixed_search


def frozen(x, rows):
    machine = Machine([],[],None)
    core = machine.cores[0]
    values = [x]
    for row in rows:
        pool = values+row['constants']
        a,b,c = [pool[j] for j in row['operands']]
        core.scratch[:24] = [a]*8+[b]*8+[c]*8
        machine.scratch_write = {}
        if row['op'] == 'mad':
            machine.valu(core,'multiply_add',24,0,8,16)
        else:
            machine.alu(core,row['op'],24,0,8)
        values.append(machine.scratch_write[24])
    return values[-1]


class FreeMiddleKeyTests(unittest.TestCase):
    def test_target_and_highest_key_normalization(self):
        rng = random.Random(883)
        for _ in range(200):
            b,ik,ok = [rng.getrandbits(32) for _ in range(3)]
            g = myhash(prior_for_b(b)) ^ 0xb55a4f09
            self.assertEqual((g ^ (g >> 16))&MASK, middle(b)&MASK)
            self.assertEqual((middle(b ^ ik)^ok)&MASK,
                (middle(b ^ (ik & 0x7fffffff)) ^ (ok ^ (ik & 0x80000000)))&MASK)

    def test_arbitrary_programs_match_frozen_isa_and_fixed_z3(self):
        rng = random.Random(884)
        for _ in range(120):
            rows = []
            for i in range(rng.randrange(1,5)):
                rows.append(dict(op=rng.choice(OPS),
                    operands=[rng.randrange(i+3) for _ in range(3)],
                    constants=[rng.getrandbits(32) for _ in range(2)]))
            for x in (0,MASK,rng.getrandbits(32)):
                actual = frozen(x,rows)
                self.assertEqual(actual,execute(x,rows,32))
                self.assertEqual(actual,z3.simplify(expression(z3.BitVecVal(x,32),rows,32)).as_long())

    def test_nonzero_key_automatic_control_is_only_a_reduced_result(self):
        report = search(('mad','^','mad'),5,10,9000)
        self.assertEqual('equivalent_on_reduced_domain',report['result'],report)
        self.assertEqual(32,report['verified_inputs'])
        for x in range(32):
            self.assertEqual((middle(x ^ report['input_key']) ^ report['output_key'])&31,
                             execute(x,report['program'],5))
        self.assertFalse(report['new_executable'])

    def test_original_full32_positive_control(self):
        report = search(('mad','mad','^','mad'),32,10,9000,pin_original=True)
        self.assertEqual('proved_equivalent_full32',report['result'],report)
        self.assertEqual(0,report['input_key'])
        self.assertEqual(0,report['output_key'])
        for x in (0,1,MASK,0xdeadbeef):
            self.assertEqual(middle(x)&MASK,frozen(x,report['program']))

    def test_vectorized_key_screen_matches_independent_small_exhaustion(self):
        report = polynomial_screen(4)
        survivors = []
        for ik in range(16):
            for ok in range(16):
                good = True
                for row in report['rectangles']:
                    b,h,k,m = [row[j] for j in ('base','h','k','divisor')]
                    ys = [(middle(((b+d)&15)^ik)^ok)&15 for d in (0,h,k,h+k)]
                    if (ys[3]-ys[1]-ys[2]+ys[0]) % m:
                        good = False
                        break
                if good:
                    survivors.append((ik,ok))
        self.assertEqual(len(survivors),report['remaining_pairs'])
        control = polynomial_screen(6,affine_control=True)
        self.assertTrue(control['zero_key_control_retained'])

    def test_all_low11_key_pairs_fail_the_polynomial_necessary_condition(self):
        report = polynomial_screen()
        self.assertEqual(4194304,report['initial_key_pairs'])
        self.assertEqual(0,report['remaining_pairs'])
        self.assertEqual('all_key_pairs_excluded',report['result'])
        self.assertFalse(report['new_executable'])

    def test_nested_normalizations_preserve_the_exposed_function(self):
        rng = random.Random(886)
        for width in (3,5,11,32):
            mask,high = (1 << width)-1,1 << (width-1)
            for _ in range(80):
                original = [rng.getrandbits(width) for _ in range(5)]
                original[0] |= 1
                original[2] |= 1
                ik,ok = rng.getrandbits(width),rng.getrandbits(width)
                normalized,iv,ov = normalize_nested(original,ik,ok,width)
                self.assertEqual(0,iv & (high|1))
                self.assertEqual(0,ov & (high|1))
                self.assertEqual(0,normalized[4] & (high|1))
                self.assertEqual(0,normalized[1] & high)
                xs = range(1 << width) if width <= 5 else [0,1,mask,rng.getrandbits(width)]
                for x in xs:
                    self.assertEqual(evaluate(x ^ ik,original,'nested',mask)^ok,
                        evaluate(x ^ iv,normalized,'nested',mask)^ov)

    def test_fixed_structure_positive_controls(self):
        for family in ('parallel','nested'):
            report = fixed_search(family,8,10,9000,True)
            self.assertEqual('equivalent_on_reduced_domain',report['result'],report)
            for x in range(256):
                y = x ^ report['input_key']
                expected = ((21*y+19)^(128*y+34) if family == 'parallel' else
                            5*((21*y+19)^42)+23)
                self.assertEqual((expected ^ report['output_key'])&255,
                    evaluate(x,report['coefficients'],family,255))

    def test_search_guards(self):
        for kwargs in ({'width':13},{'seconds':0},{'check_ms':0},{'pin_original':True}):
            with self.assertRaises(ValueError):
                search(('mad','^','mad'),**kwargs)
        with self.assertRaises(ValueError):
            search(('mad','>>','mad'))


if __name__ == '__main__':
    unittest.main()
