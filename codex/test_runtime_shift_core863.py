from itertools import product
import random
import unittest

import numpy as np

from codex.cegis_c1_absorb863 import _instr_options
from codex.search_early_boundary863 import early_inputs, MASK
from codex.close_early_boundary863 import rectangle
from codex.test_live_boundary863 import frozen
from codex.screen_joint_affine_slice863 import data
from tests.frozen_problem import Machine, DebugInfo


class RuntimeShiftCoreTests(unittest.TestCase):
    def test_left_right_shift_edges_and_forced_zero_counts_match_frozen(self):
        words = [0, 1, 3, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFE, MASK]
        for op in ('<<', '>>'):
            for count in (0, 1, 15, 31, 32, 63):
                m = Machine([0]*32, [{'valu':[(op, 16, 0, 8)]}], DebugInfo({}))
                m.enable_pause = False; m.enable_debug = False
                m.cores[0].scratch[0:8] = words; m.cores[0].scratch[8:16] = [count]*8
                m.run(); out = m.cores[0].scratch[16:24]
                expected = [((v << count) if op == '<<' else v >> count) & MASK for v in words]
                self.assertEqual(out, expected)
                if op == '<<' and any(v & 1 for v in out):
                    self.assertEqual(count, 0)
                if op == '>>' and any(v >= 1 << 31 for v in out):
                    self.assertEqual(count, 0)

    def test_variable_right_shift_after_real_first_operations_matches_frozen(self):
        rng = random.Random(8642845)
        for case in range(64):
            op = ('^', '&', '|', 'mad')[case % 4]
            bs = [rng.getrandbits(32) for _ in range(8)]
            values = [early_inputs(b) for b in bs]
            i, j, k = [rng.randrange(9) for _ in range(3)]
            c = (0, 1, 31, MASK, 1 << 16, rng.getrandbits(32))[case % 6]
            source = [v[i] for v in values]; addend = [v[j] for v in values]
            datum = [v[k] for v in values]
            amounts = [((v*c+a) & MASK) if op == 'mad' else v^c if op == '^' else v&c if op == '&' else v|c
                       for v, a in zip(source, addend)]
            first = ('multiply_add', 16, 0, 8, 24) if op == 'mad' else (op, 16, 0, 8)
            self_data = case % 5 == 0
            program = [{'valu':[first]}, {'valu':[('>>', 40, 16 if self_data else 32, 16)]}]
            m = Machine([0]*32, program, DebugInfo({})); m.enable_pause = False; m.enable_debug = False
            m.cores[0].scratch[0:8] = source; m.cores[0].scratch[8:16] = [c]*8
            m.cores[0].scratch[24:32] = addend; m.cores[0].scratch[32:40] = datum
            m.run()
            expected = [d >> a for d, a in zip(amounts if self_data else datum, amounts)]
            self.assertEqual(m.cores[0].scratch[40:48], expected)
            if self_data:
                self.assertEqual(expected, [0]*8)

    def test_rectangle_first_mad_and_small_word_wrap_argument(self):
        vv = [early_inputs(b) for b in rectangle()['primary_b']]
        rng = random.Random(8642846)
        options = _instr_options(8, 'mad')
        for _ in range(100):
            row = dict(order=['mad'], wiring=[list(rng.choice(options))])
            cc = [[rng.getrandbits(32), rng.getrandbits(32)]]
            initial = [[vv[i % 4][j] for i in range(8)] for j in range(9)]
            actual = frozen(row, cc, initial)[-1][:4]
            self.assertEqual(len({v & 65535 for v in actual}), 1)
            self.assertEqual((actual[3]-actual[1]-actual[2]+actual[0]) & MASK, 0)
        nonvacuous = 0
        for low in range(16):
            for a, b, c in product(range(16), repeat=3):
                ts = [low+16*a, low+16*b, low+16*c]
                ts.append((ts[1]+ts[2]-ts[0]) & 255)
                if sum(t < 4 for t in ts) >= 3:
                    nonvacuous += 1
                    self.assertEqual(ts, [low]*4)
                    self.assertLess(low, 4)
        self.assertGreater(nonvacuous, 0)

    def test_slicing_keeps_full32_overflow_and_relaxes_actual_affine_tail(self):
        rng = random.Random(8642847); width = 11; mask = (1 << width)-1
        hs = np.arange(1 << width, dtype=np.uint64)
        for _ in range(16):
            low = rng.randrange(65536); _, vv = data(hs, low)
            source = rng.randrange(4); addend = rng.choice((None, 0, 1, 2, 3))
            a, b, g, h = [rng.getrandbits(32) for _ in range(4)]; g |= 1
            with np.errstate(over='ignore'):
                y = vv[addend] if addend is not None else b
                t = (a*vv[source]+y) & MASK
                target = (g*(t ^ (t >> 16))+h) & MASK
            self.assertTrue(np.all((t & 65535) == (t[0] & 65535)))
            alpha = a & mask
            beta = (int(t[0] >> 16)-alpha*int(vv[source][0] >> 16)-
                    (int(vv[addend][0] >> 16) if addend is not None else 0)) & mask
            key = int(t[0]) & mask; r = pow(g, -1, 1 << width); s = -r*h & mask
            with np.errstate(over='ignore'):
                left = ((alpha*(vv[source] >> 16)+(vv[addend] >> 16 if addend is not None else 0)+beta) & mask) ^ key
                right = (r*target+s) & mask
            self.assertTrue(np.all(left == right))


if __name__ == '__main__':
    unittest.main()
