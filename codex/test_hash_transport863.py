"""Check research formulas against the untouched frozen ISA and SMT controls."""
import random
import sys
from pathlib import Path
import unittest

import z3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from frozen_problem import Machine
from codex.search_hash_transport863 import (MASK, OPS, FIRST_FORMS,
    first_operation, operation, search, target)
from codex.prove_middle_quadratic863 import certificate
from codex.prove_first_form_coverage863 import prove


class TransportTests(unittest.TestCase):
    def test_opcode_formulas_match_frozen_machine(self):
        machine = Machine([], [], None)
        core = machine.cores[0]
        rng = random.Random(863)
        cases = [(0,1,0), (MASK,MASK,MASK), (MASK,2,1), (MASK,31,19),
                 (MASK,32,MASK), (MASK,63,0), (0x80000000,7,MASK)]
        cases += [(rng.getrandbits(32),rng.getrandbits(32) or 1,rng.getrandbits(32))
                  for _ in range(24)]
        for code in OPS:
            for a,b,c in cases:
                if code in ('<<','>>'): b %= 64
                core.scratch[:24] = [a]*8 + [b]*8 + [c]*8
                machine.scratch_write = {}
                if code == 'mad': machine.valu(core,'multiply_add',24,0,8,16)
                else: machine.alu(core,code,24,0,8)
                result, valid = operation(code, *(z3.BitVecVal(v,32) for v in (a,b,c)))
                self.assertTrue(z3.is_true(z3.simplify(valid)))
                self.assertEqual(z3.simplify(result).as_long(),machine.scratch_write[24],
                                 (code,a,b,c))

    def test_zero_divisors_are_invalid(self):
        for code in ('//','%','cdiv'):
            _, valid = operation(code,z3.BitVecVal(MASK,32),z3.BitVecVal(0,32),z3.BitVecVal(0,32))
            self.assertTrue(z3.is_false(z3.simplify(valid)))

    def test_first_forms_match_frozen_machine(self):
        machine = Machine([],[],None)
        core = machine.cores[0]
        rng = random.Random(864)
        for form in FIRST_FORMS:
            for original_x in [0,1,31,32,63,MASK] + [rng.getrandbits(32) for _ in range(12)]:
                x,d,e = original_x,rng.getrandbits(32) or 1,rng.getrandbits(32)
                # Avoid allocating enormous Python ints for left shifts.
                if form in ('constant-shl','self-shl'): x %= 64
                if form == 'shr': d = d % 31 + 1
                if form == 'shr16': d = 16
                if form == 'affine': code,args = 'mad',(x,d,e)
                elif form == 'square': code,args = 'mad',(x,x,d)
                elif form == 'square-input': code,args = 'mad',(x,x,x)
                elif form.startswith('constant-'):
                    code,args = ('<<' if form.endswith('shl') else '>>'),(d,x,0)
                elif form.startswith('self-'):
                    code,args = ('<<' if form.endswith('shl') else '>>'),(x,x,0)
                elif form in ('less-than','greater-than','equal'):
                    code = '==' if form == 'equal' else '<'
                    args = (d,x,0) if form == 'greater-than' else (x,d,0)
                else: code,args = ('>>' if form in ('shr','shr16') else form),(x,d,0)
                core.scratch[:24] = [args[0]]*8+[args[1]]*8+[args[2]]*8
                machine.scratch_write = {}
                if code == 'mad': machine.valu(core,'multiply_add',24,0,8,16)
                else: machine.alu(core,code,24,0,8)
                value,valid = first_operation(form,*(z3.BitVecVal(v,32) for v in (x,d,e)))
                self.assertTrue(z3.is_true(z3.simplify(valid)))
                self.assertEqual(z3.simplify(value).as_long(),machine.scratch_write[24],
                                 (form,x,d,e))

    def test_linear_transport_identity_all_uint32(self):
        x,n = z3.BitVecs('state node',32)
        linear = lambda v: v ^ z3.LShR(v,16)
        solver = z3.SolverFor('QF_BV')
        solver.add(z3.Or(linear(linear(x)) != x,
                        linear(x ^ linear(n)) != (linear(x) ^ n)))
        self.assertEqual(solver.check(),z3.unsat)

    def test_positive_controls(self):
        for code,control in (('^','xor'),('mad','mad')):
            result = search(code,10,control)
            self.assertEqual(result['status'],'proved_equivalent',result)

    def test_middle_change_of_variable_all_uint32(self):
        b = z3.BitVec('middle_input',32)
        q = 33*b + 0xe9f8cc1d
        original = 9*(q ^ (16896*b + 0xaccf6200)) + 0xfd7046c5
        # Odd 33 has a multiplicative inverse modulo 2**32.
        self.assertEqual((33*pow(33,-1,1 << 32)) & MASK,1)
        solver = z3.SolverFor('QF_BV')
        solver.add(target(q,objective='middle-tail') != original)
        self.assertEqual(solver.check(),z3.unsat)

    def test_quadratic_certificate_with_independent_solver(self):
        proof = certificate()
        self.assertEqual(proof['status'],'polynomial_template_excluded')
        width = proof['width']
        a,b,c = z3.BitVecs('quadratic_a quadratic_b quadratic_c',width)
        solver = z3.SolverFor('QF_BV')
        points = {0,1,2,*(row['input'] for row in proof['rejected'])}
        for point in points:
            # Derive each expected low-bit value from the full-width SMT
            # objective, independently of the certificate's integer target.
            full = target(z3.BitVecVal(point,32),objective='middle-tail')
            solver.add(a*point*point+b*point+c == z3.Extract(width-1,0,full))
        self.assertEqual(solver.check(),z3.unsat)
        for row in proof['rejected']:
            full = target(z3.BitVecVal(row['input'],32),objective='middle-tail')
            self.assertEqual(z3.simplify(z3.Extract(width-1,0,full)).as_long(),
                             row['expected'])
            self.assertNotEqual(row['expected'],row['candidate'])

    def test_mad_after_any_fixed_right_shift_is_excluded(self):
        result = search('mad',10,first='shr')
        self.assertEqual(result['status'],'unsat',result)

    def test_first_form_symbolic_coverage(self):
        result = prove()
        self.assertEqual(result['first_instruction_shapes'],60)
        self.assertEqual(result['normalized'],54)
        self.assertEqual(result['invalid_at_zero'],6)
        forms = {form for row in result['rows'] for form in row.get('families',[])}
        self.assertEqual(forms,set(FIRST_FORMS)-{'shr16'})


if __name__ == '__main__':
    unittest.main()
