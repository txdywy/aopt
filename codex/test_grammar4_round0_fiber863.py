"""Checks for the fiber filter and the low-bit projection (grammar4_round0_fiber863)."""
import random
import unittest

from codex import grammar3_head863 as g
from codex import grammar4_round0_863 as h
from codex import grammar4_round0_fiber863 as fb
from codex import grammar4_round0_pinned863 as q


def breaks_line(bit0, rng, lows=2000):
    """Some u mod 2^19 where bit0(u) ^ u19 is not constant over u >> 19 < 2048."""
    for _ in range(lows):
        low, seen = rng.randrange(1 << 19), set()
        for gg in range(2048):
            seen.add(bit0(low | gg << 19) ^ (gg & 1))
            if len(seen) > 1:
                return True
    return False


def covers(cls, consts, k_at=None, k_width=0):
    """Do concrete constants (consts[n][i]) fall in the class?"""
    return all((consts[n][i] >> lo) % (1 << w) == v for n, i, lo, w, v in fb.class_pins(cls, k_at, k_width))


class FiberTests(unittest.TestCase):
    def test_target_bit0(self):
        """Input bits 20..29 leave the target's bit 0 alone; bit 19 always flips it."""
        rng = random.Random(61)
        for _ in range(4000):
            x = rng.randrange(g.D30)
            t0 = g.target(x) & 1
            self.assertEqual(g.target(x ^ (rng.randrange(1, 1 << 10) << 20)) & 1, t0)
            self.assertNotEqual(g.target(x ^ (1 << 19)) & 1, t0)

    def test_projection_exact(self):
        import z3
        sem, csem = g.ops(z3), g.ops(None)
        rng = random.Random(62)
        ts = h.templates4()
        for idx in (928, 1082, 1083, 1377, 10331, 10336):
            p = ts[idx]
            for _ in range(60):
                amounts = {n: rng.randrange(32) for n in q.shifts(p)}
                k = rng.randrange(1, 9)
                need = fb.projection_widths(p, amounts, k)
                kv = [[amounts[n]] if f == 'shrk' else [rng.getrandbits(32) for _ in range(g.NCONST.get(f, 0))]
                      for n, (f, _) in enumerate(p)]
                consts = [[] if f == 'shrk' else [z3.BitVecVal(x, 32) for x in ks] for ks, (f, _) in zip(kv, p)]
                u = rng.getrandbits(32)
                got = z3.simplify(fb.run_projected(p, amounts, u, consts, need, sem)).as_long()
                self.assertEqual(got, g.run_program(p, u, kv, csem) & ((1 << k) - 1))

    def test_mad_form(self):
        """928, 1082 and 1083 compute a*(a >> p) + (a >> q) for the (p, q) of mad_pq."""
        sem = g.ops(None)
        ts = h.templates4()
        rng = random.Random(63)
        for idx in (928, 1082, 1083):
            p = ts[idx]
            for _ in range(200):
                am = [rng.randrange(1, 20), rng.randrange(1, 12)]
                pq = fb.mad_pq(p, dict(zip(q.shifts(p), am)))
                kv = [[rng.getrandbits(32), rng.getrandbits(32)], [am[0]], [am[1]], []]
                u = rng.getrandbits(32)
                a = (kv[0][0] * u + kv[0][1]) & g.MASK
                self.assertEqual(g.run_program(p, u, kv, sem), (a * (a >> pq[0]) + (a >> pq[1])) & g.MASK)
        self.assertEqual(fb.mad_pq(ts[1082], {1: 19, 2: 11}), (19, 30))
        self.assertEqual(fb.mad_pq(ts[1083], {1: 19, 2: 11}), (30, 19))

    def test_filter_rejects_mad(self):
        """For mu, nu outside every kept class some line breaks bit 0 ^ u19."""
        rng = random.Random(64)
        for pp, qq in ((19, 30), (31, 20), (20, 31), (30, 19)):
            kept = fb.fiber_classes(0, pp, qq)['classes']
            tried = 0
            while tried < 25:
                mu, nu = rng.getrandbits(32), rng.getrandbits(32)
                if rng.random() < 0.3:
                    mu &= ~0x3FF         # also small valuations of the step
                if any(covers(c, [[mu, nu]]) for c in kept):
                    continue
                tried += 1

                def bit0(u, mu=mu, nu=nu):
                    a = (mu * u + nu) & g.MASK
                    return ((a & (a >> pp)) ^ (a >> qq)) & 1
                self.assertTrue(breaks_line(bit0, rng), (pp, qq, mu, nu))

    def test_filter_rejects_window(self):
        rng = random.Random(65)
        kept = fb.fiber_classes(1, 28, 2)['classes']
        tried = 0
        while tried < 25:
            mu, nu, k = rng.getrandbits(32), rng.getrandbits(32), rng.getrandbits(32)
            if rng.random() < 0.5:
                mu = (rng.choice((2047, 2049, 6143, 6145)) + (rng.getrandbits(19) << 13)) & g.MASK
            if any(covers(c, [[mu, nu], [], [k]], 2, 3) for c in kept):
                continue
            tried += 1

            def bit0(u, mu=mu, nu=nu, k=k):
                a = (mu * u + nu) & g.MASK
                return (((a + k * (a >> 28)) & g.MASK) >> 2) & 1
            self.assertTrue(breaks_line(bit0, rng), (mu, nu, k))

    def test_window_lemma(self):
        """mu*u + nu (u < n) inside one block of 2^e mod 2^N forces |mu| * (n - 1) < 2^e (8-bit model)."""
        N = 8
        for e in range(1, 6):
            for n in (2, 3, 1 << e):
                for mu in range(1 << N):
                    ms = mu - (1 << N) if mu >= 1 << (N - 1) else mu
                    for nu in range(1 << N):
                        blocks = {((mu * u + nu) % (1 << N)) >> e for u in range(n)}
                        if len(blocks) == 1:
                            self.assertLess(abs(ms) * (n - 1), 1 << e)
        self.assertEqual(fb.window_steps(2049, 19, 9), [])
        self.assertEqual(fb.window_steps(2, 20, 11), [2])
        self.assertEqual(fb.window_steps(8190, 20, 11), [(1 << 31) - 2])

    def test_class_counts(self):
        counts = {pq: len(fb.fiber_classes(0, *pq)['classes']) for pq in ((19, 30), (21, 31), (30, 19), (31, 21))}
        self.assertEqual(counts, {(19, 30): 3, (21, 31): 2, (30, 19): 6, (31, 21): 2})
        w = fb.fiber_classes(1, 28, 2)
        self.assertEqual((w['kept'], w['window'], len(w['classes'])), (200, 8, 24))


if __name__ == '__main__':
    unittest.main()
