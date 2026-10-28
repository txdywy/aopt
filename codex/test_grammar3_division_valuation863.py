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

    def test_classification(self):
        ts = g.templates(True)
        base = dv.base_shift_unsat()
        rules = [dv.classify(ts[i], base)[0] for i in dv.open_items()]
        self.assertEqual(len(rules), 98)
        self.assertEqual(sum(r is not None for r in rules), 89)
        # the right-shift twin of every div-valuation template is UNSAT in the base grammar
        for i in dv.open_items():
            p = ts[i]
            if dv.classify(p, base)[0] == 'div-valuation':
                self.assertIn(g.canonical([p[0], ('shrk', (1,)), p[2]]), base)


if __name__ == '__main__':
    unittest.main()
