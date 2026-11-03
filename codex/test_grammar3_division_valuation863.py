"""Checks for the 2-adic division rules (grammar3_division_valuation863)."""
import random
import unittest

from codex import grammar3_head863 as g
from codex import grammar3_division_valuation863 as dv

v2 = lambda x: (x & -x).bit_length() - 1 if x else 64


class DivisionValuationTests(unittest.TestCase):
    def test_full_domain_target_valuation(self):
        rng = random.Random(51)
        for _ in range(4000):
            u, j, e = rng.getrandbits(32), rng.randrange(20, 32), rng.choice(g.KEYS)
            m, c = rng.randrange(1, 1 << 32, 2), rng.getrandbits(32)
            t = lambda w: (m * g.target(w ^ e) + c) & g.MASK
            self.assertEqual(v2((t(u ^ (1 << j)) - t(u)) & g.MASK), j - 19)

    def test_top_flip_moves(self):
        rng = random.Random(52)
        for _ in range(3000):
            k = rng.randrange(2, 1 << 31)
            q, r = (1 << 31) // k, (1 << 31) % k
            i1 = rng.randrange(1 << 31)
            self.assertIn((i1 + (1 << 31)) // k - i1 // k, (q, q + 1))
            if r:   # both values occur: i1 = 0 and i1 = k - 1
                self.assertEqual({(x + (1 << 31)) // k - x // k for x in (0, k - 1)}, {q, q + 1})
                d = {(x + (1 << 31)) % k - x % k for x in (0, k - 1)}
                self.assertEqual(d, {r, r - k})
                self.assertNotEqual(v2(r), v2(abs(r - k)))
            cd = lambda a: -(-a // k)
            self.assertIn(cd(i1 + (1 << 31)) - cd(i1), (q, q + 1))
        for kk in range(32):
            i1 = rng.randrange(1 << 31)
            cd = lambda a: -(-a // (1 << kk))
            self.assertEqual(cd(i1 + (1 << 31)) - cd(i1), 1 << (31 - kk))

    def test_bijection_bounds(self):
        """A bijection of a finite range that never decreases (or never increases) is the identity."""
        rng = random.Random(54)
        for _ in range(200):
            n = rng.randrange(2, 40)
            perm = list(range(n))
            rng.shuffle(perm)
            self.assertEqual(all(perm[x] >= x for x in range(n)), perm == list(range(n)))
            self.assertEqual(all(perm[x] <= x for x in range(n)), perm == list(range(n)))

    def test_low_flip_valuations(self):
        for k, (x0, x1) in enumerate(dv.low_flip_witnesses()):
            self.assertNotEqual(dv.flip_valuation(x0, k), dv.flip_valuation(x1, k))
        rng = random.Random(55)
        for _ in range(3000):
            x, k = rng.getrandbits(32), rng.randrange(19, 32)
            self.assertEqual(dv.flip_valuation(x, k), k - 19)

    def test_ork_bits_move(self):
        """Flipping a bit of u that the ork constant has set moves the output by c * 2^k."""
        sem = g.ops(None)
        rng = random.Random(56)
        for idx in (6374, 6412, 6432, 6433, 6450):
            p = g.templates(True)[idx]
            for _ in range(300):
                k = rng.randrange(32)
                kv = [[rng.getrandbits(32) | (1 << k)], [rng.randrange(1, 1 << 32)], [rng.getrandbits(32)]]
                c = kv[2][0] if p[2][0] == 'lin1' else 1
                u = rng.getrandbits(32)
                d = (g.run_program(p, u ^ (1 << k), kv, sem) - g.run_program(p, u, kv, sem)) & g.MASK
                self.assertIn(d, {c * (1 << k) & g.MASK, -c * (1 << k) & g.MASK})

    def test_quotient_image_facts(self):
        self.assertTrue(dv.target_not_affine())
        cdv = lambda a, b: (a + b - 1) // b
        self.assertEqual(cdv(g.MASK, 2), 1 << 31)
        self.assertEqual(cdv(g.MASK, (1 << 31) + 1), 2)
        rng = random.Random(57)
        for _ in range(200):   # beta*u + gamma = 1 has 0 or 2^v2(beta) solutions (12-bit model)
            beta, gamma = rng.randrange(1, 1 << 12), rng.randrange(1 << 12)
            n = sum((beta * u + gamma) % (1 << 12) == 1 for u in range(1 << 12))
            self.assertIn(n, (0, 1 << v2(beta)))
        for _ in range(2000):
            u, k2 = rng.getrandbits(32), rng.getrandbits(32)
            b = (u + (k2 | 1) * (u | 1 << 31)) & g.MASK
            self.assertEqual(b % 2, 0)
        for k, k2 in ((0, rng.getrandbits(32)), (1 << 31, rng.getrandbits(32) & ~1)):
            self.assertEqual((0 + k2 * (0 | k)) & g.MASK, 0)

    def test_classification(self):
        ts = g.templates(True)
        base = dv.base_shift_unsat()
        rules = [dv.classify(ts[i], base)[0] for i in dv.open_items()]
        self.assertEqual(len(rules), 98)
        self.assertEqual(sum(r is not None for r in rules), 98)
        self.assertEqual(sum(r == 'quotient-image' for r in rules), 4)
        self.assertEqual(sum(r.startswith('ork-bits+') for r in rules), 5)
        # the right-shift twin of every div-valuation template is UNSAT in the base grammar
        for i in dv.open_items():
            p = ts[i]
            if dv.classify(p, base)[0] == 'div-valuation':
                self.assertIn(g.canonical([p[0], ('shrk', (1,)), p[2]]), base)


if __name__ == '__main__':
    unittest.main()
