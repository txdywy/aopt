"""Cross-check accelerated exact elimination against the Python implementation."""
import json
from pathlib import Path
import random
import subprocess
import unittest

import numpy as np

from codex.screen_affine_head_keys863 import (cube_separated, cubes, decode,
    mixed_key_mask, offsets, residue_screen)
from codex.screen_head_bivariate863 import compile_accelerator, run
from codex.screen_xor_mad_polynomial import consistent
from codex.scan_head_keyed_module863 import canonical_triple
from codex.scan_head_final_product863 import XOR_ROLES, last_roles
from codex.search_xor_equivalent_middle863 import execute, search
from codex.test_free_middle_keys863 import frozen
from codex.audit_head_affine_frontier863 import audit as frontier_audit
from codex.search_cross_encoding863 import C1

ROOT = Path(__file__).resolve().parent


class BivariateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.executable = compile_accelerator(Path('/tmp/aopt-head-bivariate863-test'))

    def test_output_key_elimination_against_python_all_512_keys(self):
        rng = random.Random(864383)
        for case in range(24):
            rows, cols = rng.randrange(1, 8), rng.randrange(1, 5)
            multiplier = 1 << (case % 6)
            matrix = [[rng.randrange(2048)*multiplier & 2047 for _ in range(cols)] for _ in range(rows)]
            rhs = [rng.randrange(2048) for _ in range(rows)]
            # Include consistent and rank-deficient controls, not just random
            # matrices whose RHS usually fails a divisibility condition.
            if case % 3 == 0:
                coefficients = [rng.randrange(2048) for _ in range(cols)]
                key = 2*rng.randrange(512)
                rhs = [(sum(a*b for a,b in zip(row, coefficients)) & 2047) ^ key for row in matrix]
            data = f'{rows} {cols}\n' + ''.join(' '.join(map(str, row+[b]))+'\n'
                for row,b in zip(matrix,rhs))
            done = subprocess.run([str(self.executable), 'matrix'], input=data,
                                  capture_output=True, text=True, check=True)
            actual = set(map(int, done.stdout.split()))
            expected = {key for key in range(0, 1024, 2)
                        if consistent(matrix, [value ^ key for value in rhs], 11)}
            self.assertEqual(expected, actual, case)

    def test_positive_polynomial_and_top_bit_normalizations(self):
        result = run(ROOT/'affine_head_cubes863_joint.json', self.executable, 4, True)
        self.assertTrue(result['positive_witness_retained'])
        rng = random.Random(864384)
        for _ in range(1000):
            a, b, x, output_key = [rng.randrange(2048) for _ in range(4)]
            q = ((a*x+b) & 2047) ^ x
            aa, bb = a & 1023, b & 1023
            qq = ((aa*x+bb) & 2047) ^ x
            self.assertEqual(q, (qq+(a-aa)*x+(b-bb)) & 2047)
            value = (17*x*q+23*q*q+31*x+7) & 2047
            normalized = output_key
            new_value = value
            if normalized & 1:
                normalized ^= 2047
                new_value = -new_value-1
            if normalized & 1024:
                normalized ^= 1024
                new_value += 1024
            self.assertEqual(value ^ output_key, (new_value & 2047) ^ normalized)

    def test_higher_dimensional_cubes_and_positive_controls(self):
        rng = random.Random(864385)
        shapes = list(cubes(8))
        seen = set()
        accepted = 0
        for _ in range(3000):
            base, axes = rng.choice(shapes)
            dimension = len(axes)
            seen.add(dimension)
            # Slopes 0 and 1 provide controls even in high dimensions; others
            # exercise arbitrary carry patterns and rejected separation.
            a = rng.choice((0, 1, 2, 3, rng.randrange(256)))
            b = rng.randrange(256)
            xx = [8*(base+d)+3 for d in offsets(axes)]
            tt = [(a*x+8*b-3*a) & 2047 for x in xx]
            separated = bool(cube_separated(np.array([a]), np.array([b]), base, axes)[0])
            if not separated:
                continue
            accepted += 1
            kk = rng.randrange(2048)
            values = []
            for x,t in zip(xx,tt):
                q = (t ^ kk) & (x | 73)
                values.append((q*q*x + 17*t*q + 23*x*x*x*x + 19) & 2047)
            signed = sum((-1)**(dimension-subset.bit_count())*value
                         for subset,value in enumerate(values))
            self.assertEqual(0, signed & 2047)
            # Check the entire upper-key bitmap against independent scalars.
            upper = [v >> 3 for v in values]
            packed = mixed_key_mask(upper)
            bits = sum(int(v) << (64*i) for i,v in enumerate(packed))
            for key in (0, 1, 127, 128, 255):
                delta = sum((-1)**(dimension-subset.bit_count())*(value ^ key)
                            for subset,value in enumerate(upper))
                self.assertEqual((delta & 255) == 0, bool(bits >> key & 1))
        self.assertGreater(accepted, 500)
        self.assertTrue({2, 3, 4, 5, 6}.issubset(seen))
        for residue in (0, 3, 7):
            self.assertTrue(residue_screen(residue, True, 8)['positive_pair_and_zero_key_retained'])

    def test_cube_refinement_preserves_rectangle_table_intersection(self):
        old = json.loads((ROOT/'affine_head_keys863_joint.json').read_text())
        new = json.loads((ROOT/'affine_head_cubes863_joint.json').read_text())
        self.assertEqual(8, new['max_dimension'])
        for before, after in zip(old['residues'], new['residues']):
            x, y = decode(before['table']), decode(after['table'])
            self.assertTrue(np.all((x & y) == y))
        # This attempt found no additional eliminations, recorded explicitly.
        np.testing.assert_array_equal(decode(old['surviving_pairs']), decode(new['surviving_pairs']))
        np.testing.assert_array_equal(decode(old['surviving_key_masks']), decode(new['surviving_key_masks']))

    def test_keyed_prefix_normalization_is_invertible_affine_substitution(self):
        rng = random.Random(864386)
        for trial in range(1000):
            a, b, key = [rng.randrange(2048) for _ in range(3)]
            aa, bb, kk = canonical_triple(a, b, key)
            sign = (-1 if (a & 1023) > 512 else 1)*(-1 if key & 512 else 1)
            old = lambda x: ((a*x+b) & 2047) ^ key
            new = lambda x: ((aa*x+bb) & 2047) ^ kk
            intercept = (old(0)-sign*new(0)) & 2047
            slope = (old(1)-sign*new(1)-intercept) & 2047
            self.assertIn(slope, (0, 1024))
            points = range(2048) if trial < 12 else [rng.randrange(2048) for _ in range(16)]
            for x in points:
                self.assertEqual(old(x), (sign*new(x)+slope*x+intercept) & 2047)
            self.assertLessEqual(aa, 512)
            self.assertLess(kk, 512)
            if aa % 2 == 0:
                self.assertEqual((bb & 1, kk & 1), (0, 0))

    def test_remaining_scan_fixed_selectors_accept_real_programs(self):
        self.assertEqual(51, len(last_roles()))
        for source, role in XOR_ROLES.items():
            for last in ((0, 3, 1), (1, 3, 4), (3, 3, 2), (1, 2, 3), (3, 4, 5)):
                fixed = {0: (0, 1, 2), 2: role, 3: last}
                rows = [dict(op=op, operands=list(operands), constants=constants)
                    for op, operands, constants in (
                        ('mad', (0, 1, 2), [5, 7]), ('mad', (0, 1, 2), [11, 19]),
                        ('^', role, [13, 23]), ('mad', last, [17, 29]))]
                row = search(('mad', 'mad', '^', 'mad'), 5, 5, 4500,
                    first_form='affine', fixed_operands=fixed, control_program=rows)
                self.assertEqual('equivalent_on_reduced_domain', row['result'], row)
                for x in range(32):
                    self.assertEqual(frozen(x, rows) & 31, execute(x, row['program'], 5))

    def test_accelerator_target_against_frozen_isa_entire_low11_domain(self):
        source = Path('/tmp/aopt-head-target-check863.cpp')
        executable = Path('/tmp/aopt-head-target-check863')
        source.write_text('#define main unused_entry\n#include "'+str(ROOT/'screen_head_bivariate863.cpp')+
            '"\n#undef main\nint main(){for(int x=0;x<2048;++x)std::cout<<middle(x)<<"\\n";return 0;}\n')
        subprocess.run(['clang++','-std=c++17','-O2','-o',str(executable),str(source)],
                       check=True,capture_output=True,text=True)
        output = subprocess.run([str(executable)],check=True,capture_output=True,text=True)
        actual = list(map(int,output.stdout.split()))
        self.assertEqual(2048,len(actual))
        original = [dict(op=op,operands=list(role),constants=list(constants)) for op,role,constants in (
            ('mad',(0,1,2),(33,0xe9f8cc1d)),('mad',(0,2,3),(16896,0xaccf6200)),
            ('^',(1,2,0),(0,0)),('mad',(3,4,5),(9,0xfd7046c5)))]
        self.assertEqual(actual,[frozen(x ^ C1,original) & 2047 for x in range(2048)])

    def test_complete_frontier_audit_has_one_open_order(self):
        report = frontier_audit()
        self.assertEqual((9,1),(report['closed_orders'],report['open_orders']))
        self.assertEqual(1986560,report['keyed_module_triples'])
        self.assertTrue(report['input_xor_branch_closed'] and report['keyed_branch_closed'])
        self.assertEqual(12,len(report['remaining_final_templates']))
        self.assertEqual(75600,report['final_product_wiring_patterns'])
        self.assertEqual({'direct_unsat':115,'scaled_polynomial_xor_proof':15,
            'unresolved_literal_roles':23},report['final_product_classifications'])


if __name__ == '__main__':
    unittest.main()
