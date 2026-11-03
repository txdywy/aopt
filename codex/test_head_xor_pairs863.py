"""Coverage, exact XOR-template reductions and automatic positive controls."""
from itertools import combinations_with_replacement, product
import random
import unittest

from codex.audit_head_xor_pairs863 import audit, classify
from codex.scan_head_xor_pairs863 import PAIRS, control_program, fixed_roles, last_roles
from codex.search_xor_equivalent_middle863 import execute, search
from codex.test_free_middle_keys863 import frozen


class HeadXorPairTests(unittest.TestCase):
    def test_thirteen_final_roles_cover_all_both_output_connections(self):
        actual = set(last_roles())
        self.assertEqual(13, len(actual))
        for a, b in combinations_with_replacement(range(6), 2):
            for c in range(6):
                if {2, 3}.issubset((a, b, c)):
                    self.assertIn((a, min(b, 4), min(c, 4)), actual)
        report = audit()
        self.assertEqual(18900, report['affine_wiring_patterns'])
        self.assertEqual(52, report['direct_queries'])
        self.assertEqual((7, 3), (report['closed_orders'], report['open_orders']))

    def test_every_hard_reduction_preserves_full32_xor_constants_and_mad(self):
        rng = random.Random(86491)
        mask = 0xffffffff
        checked = 0
        for second, third, pair, c in product(combinations_with_replacement(range(4), 2),
                combinations_with_replacement(range(5), 2),
                combinations_with_replacement(range(6), 2), range(6)):
            last = (*pair, c)
            family, role = classify(second, third, last)
            if role is None:
                continue
            a, b, k, l, j, n, e, f = [rng.getrandbits(32) for _ in range(8)]
            u_mask = (1, 2, 4, 8)[second[0]] ^ (1, 2, 4, 8)[second[1]]
            v_mask = (1, 2, u_mask, 16, 32)[third[0]] ^ (1, 2, u_mask, 16, 32)[third[1]]

            def key(bits):
                value = 0
                for bit, constant in ((4, k), (8, l), (16, j), (32, n)):
                    if bits & bit:
                        value ^= constant
                return value

            u_key, v_key = key(u_mask), key(v_mask)
            first_key, second_key = {
                'xor_then_key': (0, v_key),
                'key_then_xor': (u_key, 0),
                'parallel_input_key': (0, u_key if u_mask & 3 == 2 else v_key),
                'parallel_keys': (u_key, v_key),
            }[family]
            rows = control_program(family, role)
            for row, constants in zip(rows, ([a, b], [first_key, 0], [second_key, 0],
                                             [f if 5 in last else e, 0])):
                row['constants'] = constants
            for x in (0, mask, rng.getrandbits(32)):
                t = (a*x+b) & mask
                pool = (x, t, k, l)
                u = pool[second[0]] ^ pool[second[1]]
                pool = (x, t, u, j, n)
                v = pool[third[0]] ^ pool[third[1]]
                pool = (x, t, u, v, e, f)
                expected = (pool[last[0]]*pool[last[1]]+pool[last[2]]) & mask
                self.assertEqual(expected, execute(x, rows, 32), (family, second, third, last))
            checked += 1
        self.assertEqual(256, checked)

    def test_all_fifty_two_automatic_controls_and_frozen_semantics(self):
        rng = random.Random(86492)
        for family, role in product(PAIRS, last_roles()):
            rows = control_program(family, role)
            report = search(('mad', '^', '^', 'mad'), 5, 5, 4500,
                first_form='affine', fixed_operands=fixed_roles(family, role), control_program=rows)
            self.assertEqual('equivalent_on_reduced_domain', report['result'], report)
            self.assertEqual(32, report['verified_inputs'])
            for x in range(32):
                self.assertEqual(frozen(x, rows) & 31, execute(x, report['program'], 5))
            for x in (0, 0xffffffff, rng.getrandbits(32)):
                self.assertEqual(frozen(x, rows), execute(x, rows, 32))

    def test_inner_product_fixed_roles_admit_known_programs(self):
        for source, third, tail in product(('input', 'key'),
                ((0, 2, 1), (1, 2, 3), (2, 2, 4), (2, 3, 4)), (0, 1, 2)):
            fixed = {0: (0, 1, 2), 1: (0, 1, 0) if source == 'input' else (1, 2, 0),
                     2: third, 3: (tail, 3, 0)}
            rows = [dict(op=op, operands=list(fixed[i]), constants=constants)
                for i, (op, constants) in enumerate(zip(('mad', '^', 'mad', '^'),
                    ([5, 7], [13, 19], [9, 23], [0, 0])))]
            report = search(('mad', '^', 'mad', '^'), 5, 5, 4500,
                first_form='affine', fixed_operands=fixed, control_program=rows)
            self.assertEqual('equivalent_on_reduced_domain', report['result'], report)
            for x in range(32):
                self.assertEqual(frozen(x, rows) & 31, execute(x, report['program'], 5))


if __name__ == '__main__':
    unittest.main()
