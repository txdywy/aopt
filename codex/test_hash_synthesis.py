"""Optional research tests; run with the isolated z3-solver interpreter."""

import unittest
import json
from pathlib import Path
import subprocess
import sys
import random
import z3

from codex.synthesize_hash_cegis import OPS, apply, middle_reference
from codex.search_encoded_middle_dag863 import KEYS, prefix_certificate, sequences


def middle(x):
    return 9 * ((33*x + 0xE9F8CC1D) ^ (16896*x + 0xACCF6200)) + 0xFD7046C5


class SynthesisSemanticsTests(unittest.TestCase):
    def test_encoded_middle_full32_semantics_and_projection(self):
        rng = random.Random(870)
        for ik, ok in KEYS.values():
            for x in (0, 1, 0xffffffff, 1 << 31, *[rng.getrandbits(32) for _ in range(30)]):
                expected = (middle(x ^ ik) ^ ok) & 0xffffffff
                self.assertEqual(expected, middle_reference(x, ik, ok))
                symbolic = middle_reference(z3.BitVecVal(x, 32), ik, ok)
                self.assertEqual(expected, z3.simplify(symbolic).as_long())
                for width in (1, 8, 11, 16, 31):
                    projected = middle_reference(z3.BitVecVal(x, width), ik, ok)
                    self.assertEqual(expected & ((1 << width)-1), z3.simplify(projected).as_long())
        for keys in ((-1, 0), (0, 1 << 32)):
            with self.assertRaises(ValueError):
                middle_reference(0, *keys)

    def test_encoded_middle_cli_controls_and_exhaustive_verification(self):
        for mode, (ik, ok) in KEYS.items():
            cmd = [sys.executable, '-m', 'codex.synthesize_hash_cegis', '--slice', 'middle',
                '--width', '8', '--instructions', '3', '--op-sequence=^,mad,^',
                '--middle-input-xor', hex(ik), '--middle-output-xor', hex(ok),
                '--canonical-operands', '--consecutive-samples', '16',
                '--seconds', '10', '--check-ms', '9000']
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            self.assertEqual(0, proc.returncode, proc.stderr)
            report = json.loads(proc.stdout.splitlines()[-1])
            self.assertEqual('proved_equivalent_modulo', report['result'], (mode, report))
            self.assertEqual((ik, ok), (report['middle_input_xor'], report['middle_output_xor']))
            for x in range(256):
                values = [x]
                for row in report['program']:
                    pool = values + row['constants']
                    a, b, c = [pool[i] for i in row['operands']]
                    value = a*b+c if row['op'] == 'mad' else a ^ b
                    values.append(value & 255)
                self.assertEqual((middle(x ^ ik) ^ ok)&255, values[-1])

    def test_encoded_middle_cli_rejects_mismatched_target(self):
        for args in (['--middle-input-xor', '-1'], ['--middle-output-xor', '0x100000000'],
                     ['--slice', 'first', '--middle-input-xor', '1']):
            proc = subprocess.run([sys.executable, '-m', 'codex.synthesize_hash_cegis', *args],
                                  capture_output=True, text=True, timeout=5)
            self.assertEqual(2, proc.returncode)

    def test_encoded_bitwise_prefix_certificate_with_high_degree_controls(self):
        self.assertEqual(10, len(sequences()))
        rng = random.Random(871)
        count = 0
        for mode in KEYS:
            for order in sequences():
                cert = prefix_certificate(mode, order)
                if cert is None:
                    continue
                count += 1
                row = cert['rectangle']
                self.assertNotEqual(0, row['mixed_difference'])
                for _ in range(30):
                    k1, k2, c1, c2 = [rng.getrandbits(32) for _ in range(4)]
                    values = []
                    for x in row['inputs']:
                        q, t = x ^ k1, x ^ k2
                        values.append((c1*pow(x, 5, 2048)*pow(q, 3, 2048)
                            +c2*pow(t, 4, 2048)*pow(q, 2, 2048)) & 2047)
                    self.assertEqual(0, (values[0]-values[1]-values[2]+values[3]) & 2047)
        self.assertEqual(6, count)

    def test_first_mad_forms_cover_all_operand_choices(self):
        import itertools
        x, k0, k1 = z3.BitVecs('form_x form_k0 form_k1', 32)
        values = (x, k0, k1)
        for a, b, c in itertools.product(range(3), repeat=3):
            if a == b == 0:
                expected = x*x + values[c]
                # This is precisely square-input or square-constant.
            else:
                linear = (values[b] if a == 0 else 0) + (values[a] if b == 0 else 0)
                constant = values[a]*values[b] if a != 0 and b != 0 else 0
                linear = linear + (1 if c == 0 else 0)
                constant = constant + (values[c] if c != 0 else 0)
                expected = linear*x + constant
            solver = z3.SolverFor('QF_BV')
            solver.add(values[a]*values[b]+values[c] != expected)
            self.assertEqual(solver.check(), z3.unsat, (a,b,c))

    def test_first_slice_high_bit_propagation(self):
        x = z3.BitVec('first_high_bit_x', 32)
        def first(value):
            a = 4097*value + 0x7ED55D16
            return a ^ z3.LShR(a, 19) ^ 0xC761C23C
        solver = z3.SolverFor('QF_BV')
        solver.set(timeout=5000)
        solver.add(first(x ^ 0x80000000) != (first(x) ^ 0x80001000))
        self.assertEqual(solver.check(), z3.unsat)

    def test_canonical_operand_symmetries(self):
        a, b, c, other_c = z3.BitVecs('canonical_a canonical_b canonical_c canonical_other_c', 32)
        for index, op in enumerate(OPS):
            selector = z3.BitVecVal(index, 4)
            solver = z3.SolverFor('QF_BV')
            if op in ('+', '*', '^', '&', '|', 'mad'):
                solver.add(apply(selector, a, b, c) != apply(selector, b, a, c))
                self.assertEqual(solver.check(), z3.unsat, op)
            if op != 'mad':
                solver = z3.SolverFor('QF_BV')
                solver.add(apply(selector, a, b, c) != apply(selector, a, b, other_c))
                self.assertEqual(solver.check(), z3.unsat, op)

    def test_first_slice_fixed_grammar_and_positive_control(self):
        script = Path(__file__).with_name('synthesize_hash_cegis.py')
        command = [sys.executable, str(script), '--slice', 'first',
                   '--instructions', '3', '--op-sequence', 'mad,>>,^',
                   '--seconds', '10', '--check-ms', '9000', '--canonical-operands',
                   '--high-bit-pairs']
        for control, expected in ((False, 'grammar_unsatisfiable'),
                                  (True, 'proved_equivalent')):
            with self.subTest(control=control):
                result = subprocess.run(command + (['--control'] if control else []),
                                        capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout.splitlines()[-1])
                self.assertEqual(report['result'], expected)
                self.assertEqual(report['width'], 32)

    def test_low_byte_affine_is_exact(self):
        x = z3.BitVec("low_byte", 8)
        solver = z3.SolverFor("QF_BV")
        solver.add(middle(x) != 41*x + 202)
        self.assertEqual(solver.check(), z3.unsat)

    def test_low_byte_affine_is_not_full_width(self):
        # A reduced-width equivalence must never be promoted to uint32.
        actual = z3.simplify(middle(z3.BitVecVal(0, 32))).as_long()
        self.assertNotEqual(actual, 202)
        self.assertEqual(actual & 255, 202)

    def test_nonshift_grammar_commutes_with_low_bit_projection(self):
        x, y, z = z3.BitVecs("projection_x projection_y projection_z", 32)
        low = lambda value: z3.Extract(10, 0, value)
        for index, op in enumerate(OPS):
            if op in ("<<", ">>"):
                continue
            with self.subTest(op=op):
                solver = z3.SolverFor("QF_BV")
                selector = z3.BitVecVal(index, 4)
                solver.add(low(apply(selector, x, y, z)) !=
                           apply(selector, low(x), low(y), low(z)))
                self.assertEqual(solver.check(), z3.unsat)

    def test_right_shift_does_not_commute(self):
        # Discarding high bits before a right shift loses observable data.
        x = z3.BitVecVal(1 << 11, 32)
        self.assertEqual(z3.simplify(z3.Extract(10, 0, z3.LShR(x, 1))).as_long(), 1024)
        self.assertEqual(z3.simplify(z3.LShR(z3.Extract(10, 0, x), 1)).as_long(), 0)


if __name__ == "__main__":
    unittest.main()
