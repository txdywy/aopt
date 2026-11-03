"""Independent oracles for encoding-boundary and shift-polynomial research."""
import itertools
import random
import unittest

import z3

from codex.search_cross_encoding863 import C1, C5, PRE_C5, middle, normalize, program, search
from codex.screen_shift_middle863 import (MASK, full_shift, literal_screen,
    samples_for, target, variable_count_certificate)
from codex.screen_xor_mad_polynomial import consistent, monomials
from codex.search_middle_shift_orders863 import command, sequences


class EncodingResearchTests(unittest.TestCase):
    def test_every_two_bit_program_and_normalization(self):
        # Enumerate ALL six coefficients without the search's symmetry cuts.
        for coefficients in itertools.product(range(4), repeat=6):
            outputs = [program(x,coefficients,2) for x in range(4)]
            bijective = len(set(outputs)) == 4
            a,b,d,e,g,h = coefficients
            self.assertEqual(bijective, bool(g & 1) and (a & 1) != (d & 1))
            if bijective:
                normalized = normalize(coefficients,2)
                self.assertEqual(outputs,[program(x,normalized,2) for x in range(4)])
                self.assertEqual([normalized[i]&1 for i in (0,1,2,4)],[1,0,0,1])

    def test_solver_matches_all_two_bit_permutations(self):
        represented = {tuple(program(x,c,2) for x in range(4))
                       for c in itertools.product(range(4),repeat=6)}
        for values in itertools.permutations(range(4)):
            row = search(2,timeout_ms=1000,target_values=values)
            self.assertEqual(row['result'], 'sat' if values in represented else 'unsat')

    def test_full32_output_key_transport(self):
        x = z3.BitVec('cross_key_x',32)
        transform = lambda y: y ^ z3.LShR(y,16)
        solver = z3.SolverFor('QF_BV')
        solver.add(transform(x ^ PRE_C5) != (transform(x) ^ C5))
        self.assertEqual(solver.check(), z3.unsat)
        solver = z3.SolverFor('QF_BV')
        solver.add(transform(transform(x)) != x)
        self.assertEqual(solver.check(), z3.unsat)

    def test_targets_are_bijective_and_projections_match(self):
        rng = random.Random(864)
        for ik,ok in ((C1,0),(0,PRE_C5),(C1,PRE_C5)):
            values = [(middle(x^ik)^ok)&4095 for x in range(4096)]
            self.assertEqual(len(set(values)),4096)
            for _ in range(100):
                x = rng.getrandbits(32)
                symbolic = z3.simplify(middle(z3.BitVecVal(x,32)^ik)^ok).as_long()
                self.assertEqual(values[x&4095],symbolic&4095)

    def test_unpinned_positive_search(self):
        for coefficients in ((21,19,128,34,5,23),(64,91,11,27,3,47)):
            result = search(11,timeout_ms=10000,control_coefficients=coefficients)
            self.assertEqual(result['result'],'sat',result)
            self.assertEqual(result['verification'],'exhaustive_low_domain')
            self.assertEqual(result['verified_inputs'],2048)

    def test_cross_boundary_targets_excluded(self):
        for ik,ok in ((C1,0),(0,PRE_C5),(C1,PRE_C5)):
            result = search(11,ik,ok,timeout_ms=10000)
            self.assertEqual(result['result'],'unsat',result)

    def test_reject_nonpermutation_and_bad_domains(self):
        with self.assertRaises(ValueError):
            search(2,target_values=[0]*4)
        with self.assertRaises(ValueError):
            search(32)
        with self.assertRaises(ValueError):
            search(2,timeout_ms=0)
        with self.assertRaises(ValueError):
            normalize((1,0,1,0,1,0),2)


class ShiftResearchTests(unittest.TestCase):
    def test_full_shift_semantics_and_narrowing_counterexample(self):
        for x in (0,1,2048,0x12345678,MASK):
            for amount in (0,1,9,16,31,32,33,MASK):
                expected = z3.simplify(z3.LShR(z3.BitVecVal(x,32),amount)).as_long()
                self.assertEqual(full_shift(x,amount),expected)
        self.assertNotEqual(full_shift(2048,1)&2047,full_shift(2048&2047,1))
        with self.assertRaises(ValueError):
            full_shift(1,-1)

    def test_all_literal_counts_with_positive_programs(self):
        for amount in range(33):
            result = literal_screen(amount)
            self.assertTrue(result['inconsistent_full32'],amount)
            self.assertTrue(result['two_mad_positive_control'],amount)

    def test_independent_smt_polynomial_oracle(self):
        # Each q is evaluated on its original full32 input first. Only the
        # subsequent polynomial congruences are projected to eleven bits.
        # This is not an eleven-bit emulation of SHR.
        for amount in (1,9,19,31):
            coefficients = z3.BitVecs(' '.join('oracle_c'+str(i) for i in range(15)),11)
            solver = z3.SolverFor('QF_BV')
            solver.set(timeout=5000)
            for x in samples_for(amount):
                terms = monomials(x,full_shift(x,amount),MASK)
                solver.add(sum(c*k for c,k in zip(coefficients,terms)) == target(x))
            self.assertEqual(solver.check(),z3.unsat,amount)

    def test_variable_count_polynomial_certificate(self):
        row = variable_count_certificate()
        self.assertEqual(row['mixed_difference'],1024)
        for numerator in (0,37,MASK):
            self.assertTrue(all(full_shift(numerator,x)==0 for x in row['samples']))
        rng = random.Random(865)
        for _ in range(50):
            coefficients = [rng.getrandbits(32) for _ in range(30)]
            values = [sum(c*pow(x,i,2048) for i,c in enumerate(coefficients))&2047
                      for x in row['samples']]
            self.assertEqual((values[3]-values[2]-values[1]+values[0])&2047,0)

    def test_arbitrary_two_mad_continuations_fit_relaxation(self):
        rng = random.Random(866)
        samples = [rng.getrandbits(32) for _ in range(35)]
        for amount in (1,9,19,31,32):
            q = [full_shift(x,amount) for x in samples]
            matrix = [monomials(x,y,MASK) for x,y in zip(samples,q)]
            for _ in range(10):
                constants = [rng.getrandbits(32) for _ in range(4)]
                first = [rng.randrange(4) for _ in range(3)]
                second = [rng.randrange(5) for _ in range(3)]
                output = []
                for x,y in zip(samples,q):
                    p = [x,y,*constants[:2]]
                    t = (p[first[0]]*p[first[1]]+p[first[2]])&MASK
                    p = [x,y,t,*constants[2:]]
                    output.append((p[second[0]]*p[second[1]]+p[second[2]])&MASK)
                self.assertTrue(consistent(matrix,output,32))

    def test_target_and_scan_command(self):
        for x in (0,1,256,264,1<<31,MASK):
            self.assertEqual(target(x),z3.simplify(middle(z3.BitVecVal(x,32))).as_long())
        orders = sequences()
        self.assertEqual(len(orders),48)
        for order in orders:
            self.assertIn('mad',order)
            self.assertIn('>>',order)
            args = command(order)
            self.assertEqual(args[args.index('--width')+1],'32')
            self.assertIn('--op-sequence='+','.join(order),args)


if __name__ == '__main__':
    unittest.main()
