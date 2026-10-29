"""Checks for folding C5 into the middle of the hash (cegis_middle_c5_863)."""
import gzip
import json
import random
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from frozen_problem import HASH_STAGES
from codex import cegis_middle_c5_863 as m
from codex.cegis_c1_absorb863 import wirings_general

MASK = m.MASK
RESULTS = m.RESULTS


def stage(a, k):
    op1, val1, op2, op3, val3 = HASH_STAGES[k]
    fns = {'+': lambda x, y: x + y, '^': lambda x, y: x ^ y,
           '<<': lambda x, y: x << y, '>>': lambda x, y: x >> y}
    return fns[op2](fns[op1](a, val1) & MASK, fns[op3](a, val3) & MASK) & MASK


def L16(e):
    return e ^ e >> 16


class MiddleC5Tests(unittest.TestCase):
    def test_target_is_hash_middle(self):
        """F is stages 2-4, and L16(F(y) ^ D) is the full stage-5 output with C5."""
        rng = random.Random(1)
        for _ in range(2000):
            y = rng.getrandbits(32)
            e = stage(stage(stage(y, 2), 3), 4)
            self.assertEqual(m.middle(y, MASK), e)
            self.assertEqual(L16(e ^ m.D), stage(e, 5))

    def test_output_keys(self):
        """Each output key leaves an affine-compatible XOR after stage 5, and all four
        agree with D below bit 15."""
        rng = random.Random(2)
        for o, c in zip(m.OKEYS, (0, 1 << 31, MASK, (1 << 31) - 1)):
            self.assertEqual(o & 0x7FFF, m.D & 0x7FFF)
            for _ in range(200):
                e = rng.getrandbits(32)
                self.assertEqual(L16(e ^ o), stage(e, 5) ^ c)

    def test_input_key_through_c1(self):
        """Stage 1 ends with the C1 XOR, so any input key K costs nothing."""
        rng = random.Random(3)
        for _ in range(500):
            a, key = rng.getrandbits(32), rng.getrandbits(32)
            self.assertEqual(a ^ (HASH_STAGES[1][1] ^ key) ^ a >> 19, stage(a, 1) ^ key)

    def test_polynomial_and_bitlocal_witnesses(self):
        """No integer polynomial (four MADs) and no bitwise-only program fits any key."""
        rng = random.Random(4)
        for key in list(range(2048)) + [rng.getrandbits(32) for _ in range(64)]:
            x, h, k, d = m.mixed_difference_witness(key)
            self.assertEqual(h * k, 2048)
            self.assertNotEqual(d, 0)
            self.assertNotEqual(m.bitlocal_witness(key), 0)

    def test_controls(self):
        """The natural MAD,MAD,XOR,MAD wiring is refuted with D and found without it."""
        order, wiring = ('mad', 'mad', '^', 'mad'), ((0, 1, 2), (0, 2, 3), (1, 2), (3, 4, 5))
        self.assertEqual(m.solve(order, wiring, 12, 120)['result'], 'unsat')
        saved = m.D
        try:
            m.D = 0
            r = m.solve(order, wiring, 10, 120)
        finally:
            m.D = saved
        self.assertEqual(r['result'], 'sat')

    def test_results_cover_every_wiring(self):
        """Every canonical mixed wiring appears once in the stored run, none SAT."""
        want = {(','.join(o), json.dumps([list(w) for w in wr]))
                for k in (1, 2, 3) for o in m.orders(k) for wr in wirings_general(o)}
        got, results, lines = set(), set(), 0
        with gzip.open(RESULTS, 'rt') as f:
            for line in f:
                r = json.loads(line)
                got.add((r['order'], json.dumps(r['wiring'])))
                results.add(r['result'])
                lines += 1
        self.assertEqual(len(want), 396249)
        self.assertEqual((got, lines), (want, len(want)))
        self.assertEqual(results, {'unsat', 'timeout'})

    def test_audit_closes_everything(self):
        """CEGIS, the carry-free certificate and the exact lifter leave nothing open."""
        a = m.audit()
        self.assertEqual(a['open'], [])
        self.assertEqual(sum(a['closed'].values()), 396249)

    def test_carry_free_certificate(self):
        """Certificates exist for every K and O mod 2^11, and polynomials in z and
        bit-local values of z really vanish on the rectangles used."""
        cert = m.carry_free_certificate(11)
        self.assertEqual(len(cert), 2048)
        rects = sorted({r for v in cert.values() for r in v})
        rng, mask = random.Random(7), 2047
        for _ in range(3000):
            a, b, c, d, e, f = (rng.getrandbits(32) for _ in range(6))
            P = lambda z: ((a * z + (z ^ b)) * (z & c) + d * (z | e) * ((z ^ f) & c)) & mask
            x, i = rng.choice(rects)
            h, k = 1 << i, 1 << (11 - i)
            self.assertEqual((P(x | h | k) - P(x | h) - P(x | k) + P(x)) & mask, 0)

    def test_carry_free_class(self):
        self.assertEqual(m.carry_free_class(('mad', 'mad', 'mad', '^'),
                                            ((0, 1, 2), (0, 2, 1), (2, 3, 4), (3, 4))), 'class+key')
        self.assertEqual(m.carry_free_class(('^', 'mad', '&', 'mad'),
                                            ((0, 1), (1, 2, 0), (0, 1), (2, 3, 4))), 'class')
        self.assertIsNone(m.carry_free_class(('mad', 'mad', '^', 'mad'),
                                             ((0, 1, 2), (0, 2, 3), (1, 2), (3, 4, 5))))

    def test_lift_controls(self):
        """The C and numpy lifters agree on the natural wiring (refuted at level 11);
        without D the C lifter finds a full width-12 solution."""
        order, wiring = ('mad', 'mad', '^', 'mad'), ((0, 1, 2), (0, 2, 3), (1, 2), (3, 4, 5))
        c, py = m.lift_c(order, wiring), m.lift(order, wiring, 12)
        self.assertEqual((c['result'], c['level']), ('unsat', 11))
        self.assertEqual((py['result'], py['level']), ('unsat', 11))
        self.assertEqual(m.lift_c(order, wiring, d=0)['result'], 'survivors')

    def test_symmetry_pins(self):
        """Every pinned wiring: random constants have pinned twins computing the same
        function up to an input XOR, which the free key K absorbs."""
        from codex.cegis_c1_absorb863 import run_numpy
        import numpy as np
        rng, mask = random.Random(6), (1 << 12) - 1
        xs = np.arange(1 << 12, dtype=np.uint64)
        seen = 0
        for k in (1, 2, 3):
            for o in m.orders(k):
                for w in wirings_general(o):
                    pins = m.symmetry_pins(o, w)
                    if not pins:
                        continue
                    seen += 1
                    consts = [(rng.getrandbits(12), rng.getrandbits(12)) for _ in o]
                    new, shift = m.merge_constants(o, w, consts, mask)
                    self.assertTrue(all(new[i][s] == v & mask for (i, s), v in pins.items()))
                    self.assertTrue(np.array_equal(run_numpy(o, w, consts, xs, mask),
                                                   run_numpy(o, w, new, xs ^ np.uint64(shift), mask)))
        self.assertEqual(seen, 78343)


if __name__ == '__main__':
    unittest.main()
