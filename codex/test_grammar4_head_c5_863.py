"""Checks for the d9/d10 head with the C5 key (grammar4_head_c5_863)."""
import random
import unittest

from codex import grammar3_head863 as g
from codex import grammar4_head_c5_863 as hc

C0 = g.C0


def y19(u, key=hc.C5):
    return ((4097 * (u ^ key) + C0) & g.MASK) >> 19 & 1


class HeadC5Tests(unittest.TestCase):
    def test_counts(self):
        ts = hc.templates()
        a = [hc.lemma_a(p) for p in ts]
        b = [hc.lemma_b(p) and not x for p, x in zip(ts, a)]
        self.assertEqual((len(ts), sum(a), sum(b)), (12254, 10160, 774))

    def test_y19_dependence(self):
        """u19 always flips y19, u20..u31 never touch it, each of u1..u18 sometimes does."""
        rng = random.Random(11)
        us = [rng.getrandbits(32) for _ in range(20000)]
        for j in range(1, 32):
            flips = sum(y19(u) != y19(u ^ 1 << j) for u in us)
            if j == 19:
                self.assertEqual(flips, len(us))
            elif j >= 20:
                self.assertEqual(flips, 0)
            else:
                self.assertGreater(flips, 0)

    def test_target_bit0(self):
        """Target bit 0 is u0 ^ y19 ^ const for every odd m."""
        rng = random.Random(12)
        const = None
        for _ in range(4000):
            u = rng.getrandbits(32)
            bit = g.target(u ^ hc.C5) & 1 ^ u & 1 ^ y19(u)
            const = bit if const is None else const
            self.assertEqual(bit, const)

    def test_target_move_order(self):
        """Flipping input bit j >= 20 moves the target by 2-adic order exactly j - 19."""
        rng = random.Random(13)
        for j in range(20, 32):
            for _ in range(500):
                x = rng.getrandbits(32) ^ hc.C5
                d = (g.target(x ^ 1 << j) - g.target(x)) & g.MASK
                self.assertEqual((d & -d).bit_length() - 1, j - 19)

    def test_exact_rules(self):
        ts = hc.templates()
        p = ts[2986]  # xk(u) ; lin2(u,i1) ; shrk(i2) ; lin1(i2,i3)
        self.assertEqual([hc.exact_rule(p, {2: s}) for s in (19, 21, 25, 31)],
                         ['rigid', 'valuation', 'valuation', 'valuation'])
        self.assertFalse(hc.valuation_full(p, 20))

    def test_lemma_a_exact(self):
        """No affine solution for the key C5; eight for each affine-compatible key."""
        self.assertEqual(hc.lemma_a_solutions(hc.C5), 0)
        for key in (0, 1 << 31, g.MASK, (1 << 31) - 1):
            self.assertEqual(hc.lemma_a_solutions(key), 8)

    def test_lemma_a2(self):
        """One case of the two-read condition (lower read at bit 0) for the key C5."""
        self.assertEqual(hc.lemma_a2(0, seconds=600), 'unsat')

    def test_low_pins(self):
        """aff(xk(u)) shifts: the natural head's low bits are among the 32 pinned solutions;
        the head without the C1 XOR is refuted under the pins, and s < 19 by dependency."""
        from codex import grammar4_round0_bit0_863 as b0
        sols = b0.xk_affine_solutions(hc.C5)
        self.assertEqual(len(sols), 32)
        self.assertIn((hc.C5 & 0xFFFFF, 4097, C0 & 0xFFFFF), sols)
        p = (('xk', (0,)), ('aff', (1,)), ('shrk', (2,)), ('xor', (2, 3)))
        for s in (19, 20, 31):
            self.assertEqual(hc.solve(p, 120, {2: s}, low=True)['result'], 'unsat')
        self.assertEqual(hc.solve(p, 10, {2: 18}, low=True)['how'], 'dependency')


if __name__ == '__main__':
    unittest.main()
