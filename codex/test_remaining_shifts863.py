"""Independent arithmetic and real-program controls for the remaining SHR cases."""
import itertools
import json
from pathlib import Path
import random
import unittest

import z3

from codex.close_remaining_shifts863 import (MASK, bitwise_certificates, closure,
    final_shift_certificates, final_xor_certificates, literal_pair_scan,
    literal_rectangle, pair_samples, rectangle, shift, target,
    variable_pair_certificates, xor_key_query)
from codex.screen_xor_mad_polynomial import consistent


def mixed(values, width):
    return (values[0]-values[1]-values[2]+values[3]) % (1 << width)


def bitop(op, x, y):
    return {'^': x ^ y, '&': x & y, '|': x | y}[op]


class RemainingShiftTests(unittest.TestCase):
    def check_polynomials(self, row, words):
        """Independent basis for every possible final multiply-add."""
        basis = [[1]*4, *words]
        for left, right in itertools.combinations_with_replacement(words, 2):
            basis.append([x*y for x, y in zip(left, right)])
        for values in basis:
            self.assertEqual(0, mixed(values, row['width']), (row, values))
        self.assertNotEqual(0, mixed([target(x) for x in row['inputs']], row['width']))

    def test_real_full32_shifts_and_invalid_domains(self):
        for x, count, op in itertools.product((0, 1, 2048, 0x87654321, MASK),
                (0, 1, 9, 19, 31, 32, 33, MASK), ('<<', '>>')):
            word = z3.BitVecVal(x, 32)
            expected = word << count if op == '<<' else z3.LShR(word, count)
            self.assertEqual(z3.simplify(expected).as_long(), shift(x, count, op))
        for args in ((-1, 0), (MASK+1, 0), (0, -1), (0, MASK+1), (0, 1, '^')):
            with self.assertRaises(ValueError):
                shift(*args)
        with self.assertRaises(ValueError):
            rectangle(3, 8, 8, 11)
        with self.assertRaises(ValueError):
            literal_rectangle(33)
        with self.assertRaises(ValueError):
            xor_key_query(rectangle(3, 8, 0, 11), timeout_ms=0)

    def test_literal_bitwise_prefixes_with_all_operand_roles(self):
        # Actual full32 prefixes, including constants and bypassed results.
        for count in range(33):
            row = literal_rectangle(count)
            xs = row['inputs']
            for key in (0, MASK, 0x12345678, 0x9e3779b9):
                constants = [[key]*4, [key ^ MASK]*4]
                for op in ('^', '&', '|'):
                    for left, right in itertools.combinations_with_replacement([xs, *constants], 2):
                        ts = [bitop(op, x, y) for x, y in zip(left, right)]
                        for source in (xs, ts, *constants):
                            qs = [shift(x, count) for x in source]
                            self.check_polynomials(row, [xs, ts, qs])
                    for source in (xs, *constants):
                        qs = [shift(x, count) for x in source]
                        for left, right in itertools.combinations_with_replacement([xs, qs, *constants], 2):
                            ts = [bitop(op, x, y) for x, y in zip(left, right)]
                            self.check_polynomials(row, [xs, qs, ts])

    def test_variable_bitwise_count_domains(self):
        cert = bitwise_certificates()
        rng = random.Random(868)
        keys = list(range(1024))+[rng.getrandbits(32) for _ in range(100)]
        first, second = cert['xor_count_disjoint_blocks']
        for key in keys:
            self.assertTrue(any(all((x ^ key) >= 32 for x in row['inputs'])
                                for row in (first, second)))
            self.assertTrue(all((x | key) >= 32 for x in cert['or_count_zero_domain']['inputs']))
        for row in cert['and_count_high_bits']:
            key = 1 << row['forbidden_mask_bit']
            for other_bits in (0, MASK, 0x19283746):
                self.assertTrue(all(x & (key | other_bits) >= 32 for x in row['inputs']))
                ts = [x & (key | other_bits) for x in row['inputs']]
                self.check_polynomials(row, [row['inputs'], ts, [0]*4])
        row = cert['and_count_low_mask']
        for key in range(32):
            ts = [x & key for x in row['inputs']]
            self.assertEqual([0]*4, ts)
            for source in (row['inputs'], ts, [MASK]*4):
                self.check_polynomials(row, [row['inputs'], ts,
                    [shift(x, t) for x, t in zip(source, ts)]])

    def test_final_xor_exclusions_and_actual_positive_programs(self):
        cert = final_xor_certificates()
        self.assertTrue(cert['excluded'])
        self.assertEqual(33, len(cert['literals']))
        key = 0x9e3779b9
        for row in cert['literals']:
            s = row['shift']
            rect = row['polynomial_xor_q_rectangle']
            fn = lambda x: ((x*shift(x, s)+17)&MASK) ^ key
            self.assertEqual('sat', xor_key_query(rect, fn=fn, fixed_key=key)['result'])
        # A complete key enumeration, independent of SMT, for the variable case.
        row = cert['variable_count_rectangle']
        values = [target(x) for x in row['inputs']]
        self.assertFalse(any(mixed([y ^ k for y in values], 11) == 0 for k in range(2048)))

    def test_every_literal_pair_and_surviving_real_mads(self):
        result = literal_pair_scan()
        self.assertTrue(result['excluded'])
        self.assertEqual(6534, result['positive_controls'])
        self.assertEqual(6534, sum(r['tested'] for r in result['rows']))
        # A separate SMT oracle uses concrete full32 shifts then low-bit
        # congruences. It does not invoke the modular elimination routine.
        xs = pair_samples()
        for op1, op2, s, k, source in (
                ('>>', '>>', 1, 9, 'x'), ('<<', '>>', 4, 9, 'q'),
                ('>>', '<<', 19, 5, 'q'), ('<<', '>>', 31, 31, 'q')):
            cs = z3.BitVecs(' '.join('pair_oracle_'+str(i) for i in range(10)), 11)
            solver = z3.SolverFor('QF_BV')
            solver.set(timeout=5000)
            for x in xs:
                q = shift(x, s, op1)
                t = shift(x if source == 'x' else q, k, op2)
                terms = [1, x, q, t, x*x, x*q, x*t, q*q, q*t, t*t]
                solver.add(sum(c*v for c, v in zip(cs, terms)) == target(x))
            self.assertEqual(z3.unsat, solver.check(), (op1, op2, s, k, source))

    def test_arbitrary_mad_operands_are_in_the_pair_relaxation(self):
        rng = random.Random(869)
        xs = [rng.getrandbits(32) for _ in range(20)]
        for _ in range(30):
            op1, op2 = rng.choice(('<<', '>>')), rng.choice(('<<', '>>'))
            s, k = rng.randrange(33), rng.randrange(33)
            qs = [shift(x, s, op1) for x in xs]
            ts = [shift(q, k, op2) for q in qs]
            matrix = [[1, x, q, t, x*x, x*q, x*t, q*q, q*t, t*t]
                      for x, q, t in zip(xs, qs, ts)]
            constants = [rng.getrandbits(32) for _ in range(2)]
            roles = [rng.randrange(5) for _ in range(3)]
            rhs = []
            for x, q, t in zip(xs, qs, ts):
                values = [x, q, t, *constants]
                a, b, c = [values[i] for i in roles]
                rhs.append((a*b+c)&MASK)
            self.assertTrue(consistent(matrix, rhs, 32))

    def test_variable_pair_count_case_splits(self):
        cert = variable_pair_certificates()
        for first_op, field, second_ops in (
                ('>>', 'second_count_first_shr', ('<<', '>>')),
                ('<<', 'second_count_first_shl', ('>>',))):
            for witness in cert[field]:
                row = witness['rectangle']
                xs = row['inputs']
                qs = [shift(x, witness['first_count'], first_op) for x in xs]
                for source in (xs, qs, [MASK]*4):
                    for op in second_ops:
                        ts = [shift(x, q, op) for x, q in zip(source, qs)]
                        self.check_polynomials(row, [xs, qs, ts])
        for s, row in enumerate(cert['literal_shr_on_high_domain']):
            xs = row['inputs']
            self.assertTrue(all(x >= 32 for x in xs))
            qs = [shift(x, s) for x in xs]
            for op in ('<<', '>>'):
                self.assertEqual([0]*4, [shift(MASK, x, op) for x in xs])
                self.check_polynomials(row, [xs, qs, [0]*4])

    def test_final_shift_conditions_in_complete_small_domains(self):
        cert = final_shift_certificates()
        self.assertTrue(cert['excluded'])
        for width in range(1, 13):
            mask = (1 << width)-1
            for t in range(mask+1):
                right = 0 if t >= width else t >> t
                left = 0 if t >= width else (t << t)&mask
                self.assertEqual(0, right)
                self.assertEqual(0, left&1)
        for row in cert['positive_first_count_odd_witnesses']:
            x, s = row['odd_input'], row['first_count']
            self.assertTrue(target(x)&1)
            self.assertNotEqual(0, x >> s)
        self.assertEqual([0x6c6565ca, 0x6c62f0f3, 0x6be04f45], list(map(target, (0, 1, 3))))

    def test_coverage_preserves_historical_unknowns_and_scope(self):
        report = json.loads(Path(__file__).with_name('middle_shift_complete863_results.json').read_text())
        result = closure(report)
        self.assertEqual((48, 0), (result['closed'], result['unresolved']))
        self.assertEqual(48, len({tuple(r['sequence']) for r in result['orders']}))
        self.assertEqual(12, sum(not r['previously_excluded'] for r in result['orders']))
        self.assertEqual(864, result['best_verified_cycles'])
        self.assertFalse(result['new_executable'])
        old = json.loads(Path(__file__).with_name('middle_shift_closure863_results.json').read_text())
        self.assertTrue(any(r['result'] == 'unknown' for r in old['final_shift_prefixes']))


if __name__ == '__main__':
    unittest.main()
