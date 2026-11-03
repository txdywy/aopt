"""Independent arithmetic and witness checks for the narrower AND search."""
import random
import unittest

import z3

from codex.audit_keyed_live_outer863 import audit, wiring_class
from codex.audit_head_polynomial863 import audit as head_audit
from codex.search_head_and_basis863 import convert, evaluate, search
from codex.search_head_polynomial863 import evaluate as xor_evaluate


class AndBasisTests(unittest.TestCase):
    def test_exact_change_of_basis_including_full_uint32(self):
        rng = random.Random(86481)
        for width in (5, 8, 11, 32):
            for _ in range(100):
                coefficients = {key: [rng.getrandbits(width) for _ in range(size)]
                                for key, size in (('p', 2), ('q', 3), ('r', 5))}
                converted = convert(coefficients, width)
                xs = range(32) if width == 5 else (0, 1, (1 << width)-1, rng.getrandbits(width))
                for x in xs:
                    self.assertEqual(xor_evaluate(x, coefficients, 'quadratic_xor_bypass', width),
                                     evaluate(x, converted, width))
            # Verify narrowing itself symbolically, not just on sampled coefficients.
            p, q = z3.BitVecs('narrow_p narrow_q', width)
            solver = z3.SolverFor('QF_BV')
            solver.add(p ^ q != p+q-2*z3.ZeroExt(1,
                z3.Extract(width-2, 0, p) & z3.Extract(width-2, 0, q)))
            self.assertEqual(z3.unsat, solver.check())

    def test_automatic_and_pinned_positive_controls(self):
        auto = search(8, 10, control=True)
        self.assertEqual('equivalent_on_relaxed_low_domain', auto['result'], auto)
        self.assertEqual(256, auto['verified_inputs'])
        for fixed in (None, 0xb55afa53):
            pinned = search(11, 5, fixed_output_key=fixed, control=True, pin_control=True)
            self.assertEqual('equivalent_on_relaxed_low_domain', pinned['result'], pinned)
            self.assertEqual(2048, pinned['verified_inputs'])

    def test_outer_audit_and_all_final_pair_categories(self):
        result = audit()
        self.assertEqual(0, result['unresolved_within_order'])
        self.assertEqual(264600, result['wiring_patterns'])
        self.assertEqual(86, result['refinement_queries'])
        self.assertEqual(40, result['one_input_second_roles'])
        for last, category in (((3, 4), 'two_mad_outputs'), ((4, 4), 'constant'),
                ((2, 4), 'polynomial_xor_a'), ((1, 4), 'polynomial_xor_prior'),
                ((4, 5), 'polynomial_xor_constant')):
            self.assertEqual(category, wiring_class(last))

    def test_head_audit_retains_the_four_open_orders(self):
        result = head_audit()
        self.assertEqual(6, result['closed_orders'])
        self.assertEqual(4, result['open_orders'])
        self.assertEqual({('mad', '^', '^', 'mad'), ('mad', '^', 'mad', '^'),
            ('mad', '^', 'mad', 'mad'), ('mad', 'mad', '^', 'mad')},
            {tuple(order) for order in result['remaining_orders']})
        self.assertFalse(result['new_executable'])


if __name__ == '__main__':
    unittest.main()
