"""Optional research tests; run with the isolated z3-solver interpreter."""

import unittest
import json
from pathlib import Path
import subprocess
import sys
import z3

from codex.synthesize_hash_cegis import OPS, apply


def middle(x):
    return 9 * ((33*x + 0xE9F8CC1D) ^ (16896*x + 0xACCF6200)) + 0xFD7046C5


class SynthesisSemanticsTests(unittest.TestCase):
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
