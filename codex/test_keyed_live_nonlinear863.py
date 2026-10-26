"""Independent coverage and positive controls for the nonlinear live search."""
from itertools import combinations_with_replacement,product
import random
import unittest

from codex.search_keyed_live_middle863 import execute,search
from codex.scan_keyed_live_nonlinear863 import canonical_first,control_program,first_forms
from codex.test_keyed_live_middle863 import frozen


class NonlinearLiveTests(unittest.TestCase):
    def test_complete_first_role_classification(self):
        forms=set(first_forms())
        self.assertEqual(28,len(forms))
        self.assertEqual(4,sum(kind=='mixed'for kind,_ in forms))
        self.assertEqual(24,sum(kind=='product'for kind,_ in forms))
        for a,b in combinations_with_replacement(range(5),2):
            for c in range(5):
                kind,role=canonical_first((a,b,c))
                if kind!='affine':self.assertIn((kind,role),forms)

    def test_custom_targets_match_full32_frozen_isa(self):
        rng=random.Random(86461)
        for (_,first),op,source in product(first_forms(),('^','&','|'),('a','prior','keyed','key')):
            rows=control_program(first,op,source)
            for a in (0,0xffffffff,rng.getrandbits(32)):
                self.assertEqual(frozen(a,42,rows),execute(a,42,rows,32))

    def test_all_28_fixed_role_controls_full32(self):
        for _,first in first_forms():
            rows=control_program(first)
            report=search(('mad','^','mad'),32,5,4500,first_operands=first,
                middle_source='key',control_program=rows,pin_control=True)
            self.assertEqual('proved_equivalent_full32',report['result'],report)
            self.assertTrue(report['control'])
            self.assertEqual(list(first),report['program'][0]['operands'])
            self.assertFalse(report['new_executable'])

    def test_automatic_mixed_and_product_controls(self):
        for first in ((0,3,2),(0,2,1)):
            rows=control_program(first)
            report=search(('mad','^','mad'),5,10,9000,first_operands=first,
                middle_source='key',control_program=rows)
            self.assertEqual('equivalent_on_restricted_low_domain',report['result'],report)
            self.assertEqual(32,report['verified_inputs'])
            for a in range(32):
                self.assertEqual(execute(a,42,rows,5)^report['output_key'],
                    execute(a,report['input_key'],report['program'],5))

    def test_rejects_incompatible_restrictions(self):
        for opts in (dict(first_operands=(3,0,2)),dict(first_operands=(0,5,2)),
                     dict(first_operands=(0,3,2),first_source='a'),dict(pin_control=True)):
            with self.assertRaises(ValueError):search(('mad','^','mad'),5,**opts)


if __name__=='__main__':unittest.main()
