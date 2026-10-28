from pathlib import Path
import json
import random
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from codex.run_middle_residual_enum863 import encode
from codex.screen_middle_division_residual863 import finite_anchors
from codex.audit_middle_mad_quotient863 import HALF_RECTANGLE, mixed
from codex.audit_odd_prefix_collisions863 import evaluate, key_grid
from codex.audit_small_numerator863 import bit_fits, forced_denominators
from codex.cegis_c1_absorb863 import _instr_options
from codex.screen_division_xor_affine863 import quotient
from codex.search_early_boundary863 import MASK, early_inputs
from tests.frozen_problem import Machine, DebugInfo


class MiddleResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('clang++') or shutil.which('c++')
        if compiler is None:
            raise unittest.SkipTest('C++ compiler unavailable for research verifier')
        cls.tmp = tempfile.TemporaryDirectory(prefix='aopt-residual-tests-')
        cls.binary = Path(cls.tmp.name)/'scanner'
        subprocess.run([compiler, '-O3', '-std=c++20', '-pthread',
                        'codex/enumerate_middle_residual863.cpp', '-o', str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'tmp'):
            cls.tmp.cleanup()

    def run_cases(self, cases):
        result = subprocess.run([str(self.binary), '4'], input=encode(cases),
                                text=True, capture_output=True, check=True)
        return {r['id']:r for r in map(json.loads, result.stdout.splitlines())}

    def test_simd_scanner_matches_complete_numpy_domains_and_wrap(self):
        rng = random.Random(8642865); cases = []
        for width in (3, 8, 12, 17):
            mask = (1 << width)-1
            for _ in range(3):
                rs = list(dict.fromkeys([0, 1, mask, 1 << (width-1),
                                         *(rng.randrange(mask+1) for _ in range(40))]))
                cases.append(dict(id=len(cases), width=width, residuals=rs))
        results = self.run_cases(cases)
        for case in cases:
            width = case['width']; mask = (1 << width)-1; half = 1 << (width-1)
            aa = np.arange(1, mask+1, 2, dtype=np.uint64)
            counts = np.zeros(len(aa), dtype=np.uint32)
            for r in case['residuals']:
                counts += ((aa*r) & mask) > half
            expected = [[int(a), int(h)] for a, h in zip(aa, counts) if h <= 1]
            actual = results[case['id']]
            self.assertTrue(actual['complete'])
            self.assertEqual(actual['evaluated_odd_parameters'], 1 << (width-1))
            self.assertEqual(actual['models'], expected)

    def test_scanner_retains_real_bounded_and_one_high_controls(self):
        width = 12; mask = (1 << width)-1; inv = pow(17, -1, 1 << width)
        cases = []
        for family in ('bounded', 'one_high'):
            qs = [(n % 64 if family == 'bounded' else mask//(n | 1)) for n in range(64)]
            rs = list(dict.fromkeys((17*q) & mask for q in qs))
            cases.append(dict(id=len(cases), width=width, residuals=rs))
        actual = self.run_cases(cases)
        for case in cases:
            models = dict(actual[case['id']]['models'])
            self.assertIn(inv, models)
            self.assertEqual(models[inv], case['id'])

    def test_duplicate_residuals_rejected_and_half_is_not_high(self):
        good = dict(id=0, width=8, residuals=[0, 128])
        actual = self.run_cases([good])[0]
        self.assertEqual(len(actual['models']), 128)
        self.assertTrue(all(high == 0 for _, high in actual['models']))
        bad = dict(id=0, width=8, residuals=[1, 1])
        result = subprocess.run([str(self.binary)], input=encode([bad]), text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)

    def test_zero_dividend_and_even_multiplier_anchors_are_real(self):
        result = finite_anchors()
        self.assertEqual(len(result['zero_dividend_addend_cases']), 9)
        self.assertTrue(all(not r['possible_initial_addends'] for r in result['zero_dividend_addend_cases']))
        self.assertTrue(all(r['zero_witness']['frozen_reference_checked'] for r in result['zero_dividend_addend_cases']))
        self.assertEqual(len(result['initial_multiplier_cases']), 9)


class MiddleCertificateTests(unittest.TestCase):
    def test_small_numerator_forcing_retains_actual_odd_denominators(self):
        nn = list(range(65)); h = 12345
        for op in ('cdiv', '%'):
            pp = [2*n+1 for n in nn]
            yy = [(p*quotient(n, p, op)+h) & MASK for n, p in zip(nn, pp)]
            forced = forced_denominators(nn, yy, h, op)
            self.assertTrue(forced)
            for row in forced:
                self.assertEqual(row['forced_prefix'], pp[row['numerator']])
                self.assertTrue(row['parity_compatible'] and row['exceeds_numerator'])

    def test_one_bitop_interpolation_keeps_real_mask_and_runtime_forms(self):
        rng = random.Random(8642869)
        values = [early_inputs(rng.getrandbits(32)) for _ in range(32)]
        for _ in range(40):
            a, b = rng.randrange(9), rng.randrange(9); key = rng.getrandbits(32)
            for op in ('^', '&', '|'):
                y = [v[a]^key if op == '^' else v[a]&key if op == '&' else v[a]|key for v in values]
                self.assertTrue(bit_fits(values, y))
                y = [v[a]^v[b] if op == '^' else v[a]&v[b] if op == '&' else v[a]|v[b] for v in values]
                self.assertTrue(bit_fits(values, y))
        self.assertTrue(bit_fits(values, [17]*len(values)))

    def test_newton_and_bit_grids_match_all_keys_on_small_ring(self):
        rng = random.Random(8642867); mask = 7
        rows = [('mad', w) for w in ([0, 9, 10], [0, 9, 1], [0, 1, 9], [9, 9, 0],
                                    [0, 9, 9], [0, 0, 1], [9, 10, 0], [0, 1, 2])]
        rows += [(op, w) for op in ('^', '&', '|') for w in ([0, 9], [0, 1], [9, 10])]
        collisions = 0
        for op, wiring in rows:
            for parity in ((0, 0), (0, 1), (1, 0), (1, 1)):
                grid = key_grid(op, parity); seen = {}
                for _ in range(128):
                    v = [rng.randrange(8) for _ in range(9)]
                    signature = tuple(evaluate(op, wiring, v, keys) & mask for keys in grid)
                    if signature in seen:
                        old = seen[signature]; collisions += 1
                        for k0 in range(parity[0], 8, 2):
                            for k1 in range(parity[1], 8, 2):
                                self.assertEqual(evaluate(op, wiring, v, [k0, k1]) & mask,
                                                 evaluate(op, wiring, old, [k0, k1]) & mask)
                    seen[signature] = v
        self.assertGreater(collisions, 100)

    def test_three_op_polynomial_plus_quotient_identities_match_frozen(self):
        rng = random.Random(8642868); vv = [early_inputs(b) for b in HALF_RECTANGLE]
        columns = [[v[j] for v in vv]*2 for j in range(9)]
        options = _instr_options(8, 'mad')
        divisors = [1, 2, 3, 4, 5, 65535, 65536, 65537, (1 << 30)+1, MASK]
        for case in range(60):
            wiring = rng.choice(options); keys = [rng.getrandbits(32) for _ in range(2)]
            d = divisors[case % len(divisors)]; op = ('//', '%', 'cdiv')[case % 3]
            a, b = [rng.randrange(11) for _ in range(2)]
            addr = lambda j: j*8 if j < 9 else 96 if j == 9 else 72
            machine = Machine([0]*32, [
                {'valu':[('multiply_add', 96, *[j*8 for j in wiring])]},
                {'valu':[(op, 104, 96, 88)]},
                {'valu':[('multiply_add', 112, addr(a), addr(b), 104)]},
            ], DebugInfo({}))
            machine.enable_pause = False; machine.enable_debug = False
            scratch = machine.cores[0].scratch
            for j, col in enumerate(columns): scratch[j*8:j*8+8] = col
            scratch[72:80] = [keys[0]]*8; scratch[80:88] = [keys[1]]*8; scratch[88:96] = [d]*8
            machine.run()
            pp = scratch[96:104]; qq = scratch[104:112]; yy = scratch[112:120]
            self.assertEqual(mixed(pp[:4]), 0)
            self.assertEqual(len({p & 65535 for p in pp}), 1)
            self.assertEqual(qq, [quotient(p, d, op) for p in pp])
            factors = columns+[pp, [keys[0]]*8]
            rr = [(x*y) & MASK for x, y in zip(factors[a], factors[b])]
            self.assertEqual(mixed(rr[:4]), 0)
            self.assertEqual(yy, [(r+q) & MASK for r, q in zip(rr, qq)])
            if op != '%':
                adjust = [(p-d*q) if op == '//' else (d*q-p) for p, q in zip(pp, qq)]
                self.assertTrue(all(0 <= r < d for r in adjust))
                expected = -mixed(adjust[:4]) if op == '//' else mixed(adjust[:4])
                self.assertEqual((d*mixed(yy[:4])) & MASK, expected & MASK)

    def test_quadratic_reflection_closes_the_discovery_miss(self):
        from codex.audit_division_first_t863 import source_inverse
        c = 0xFD7046C5; z = (-(18*c+1)*pow(81, -1, 1 << 32)) & MASK
        values = [early_inputs(source_inverse(1, v)) for v in (0, z)]
        self.assertEqual((values[0][0]**2+values[0][1]) & MASK,
                         (values[1][0]**2+values[1][1]) & MASK)


if __name__ == '__main__':
    unittest.main()
