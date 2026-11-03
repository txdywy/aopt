"""Checks for the tail-with-C5 grammar (grammar3_tail_c5_863)."""
import random
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from frozen_problem import HASH_STAGES
from codex import grammar3_head863 as g
from codex import grammar3_tail_c5_863 as t

MASK = g.MASK


def stage(a, k):
    op1, val1, op2, op3, val3 = HASH_STAGES[k]
    fns = {'+': lambda x, y: x + y, '^': lambda x, y: x ^ y,
           '<<': lambda x, y: x << y, '>>': lambda x, y: x >> y}
    return fns[op2](fns[op1](a, val1) & MASK, fns[op3](a, val3) & MASK) & MASK


class TailC5Tests(unittest.TestCase):
    def test_target_is_hash_tail(self):
        rng = random.Random(1)
        for _ in range(2000):
            a = rng.getrandbits(32)
            self.assertEqual(t.target(a), stage(stage(a, 4), 5))

    def test_free_keys(self):
        """Each free key is produced by re-choosing the middle MAD constants."""
        rng = random.Random(2)
        C2, C3 = HASH_STAGES[2][1], HASH_STAGES[3][1]
        keys = t.free_keys()
        self.assertEqual(len(keys), 2048)
        for _ in range(300):
            key, z = rng.choice(keys), rng.getrandbits(32)
            a3 = stage(stage(z, 2), 3)
            comp = bool(key & 1 << 30)  # bits 9..30 of a free key are all equal
            base = key ^ MASK if comp else key
            self.assertEqual(base & ~(511 | 1 << 31), 0)
            k, top = base & 511, base >> 31
            # complement via p's MAD (multiplier -33); k and bit 31 via q's constant
            p = (-33 * z - C2 - C3 - 1 if comp else 33 * z + C2 + C3) & MASK
            q = (16896 * z + 512 * C2 + k + (top << 31)) & MASK
            self.assertEqual(p ^ q, a3 ^ key)

    def test_bit16_flips_bit0(self):
        rng = random.Random(3)
        for _ in range(2000):
            a, key = rng.getrandbits(32), rng.choice(t.free_keys())
            self.assertNotEqual(t.target(a ^ 1 << 16, key) & 1, t.target(a, key) & 1)

    def test_template_counts(self):
        self.assertEqual((len(t.templates()), len(t.templates(full=True))), (6166, 14912))

    def test_controls(self):
        """Without C5 the natural three-instruction tail is found; with C5 it is refuted."""
        natural = (('aff', (0,)), ('shrk', (1,)), ('xor', (1, 2)))
        self.assertEqual(t.solve(natural, 60)['result'], 'unsat')
        saved = t.C5
        try:
            t.C5 = 0
            r = t.solve(natural, 60, free_only=True)
        finally:
            t.C5 = saved
        self.assertEqual(r['result'], 'sat-sampled')

    def test_move_order(self):
        """Flipping input bit j >= 16 moves the target by 2-adic order exactly j - 16."""
        rng = random.Random(4)
        keys = t.free_keys()
        for j in range(16, 32):
            for _ in range(300):
                a, key, okey = rng.getrandbits(32), rng.choice(keys), rng.choice(t.OKEYS)
                d = (t.target(a ^ 1 << j, key, okey) - t.target(a, key, okey)) & MASK
                self.assertEqual((d & -d).bit_length() - 1, j - 16)

    def test_division_moves(self):
        """The u31 flip moves i1 // K by q or q + 1 and i1 % K by r or r - K (both seen
        when r > 0), and cdiv(i1, 2^k) by exactly 2^(31 - k)."""
        rng = random.Random(5)
        for _ in range(2000):
            K = rng.randrange(2, (1 << 31) + 1)
            q, r = divmod(1 << 31, K)
            if r == 0:
                continue
            divs, mods = set(), set()
            for i1 in [0, K - 1] + [rng.randrange(1 << 31) for _ in range(50)]:  # rho = 0 and K - 1
                divs.add((i1 + (1 << 31)) // K - i1 // K)
                mods.add((i1 + (1 << 31)) % K - i1 % K)
            self.assertEqual(divs, {q, q + 1})
            self.assertEqual(mods, {r, r - K})
        for k in range(1, 31):
            for _ in range(50):
                i1 = rng.randrange(1 << 31)
                cd = lambda a: (a + (1 << k) - 1) >> k
                self.assertEqual(cd(i1 + (1 << 31)) - cd(i1), 1 << (31 - k))

    def test_ork_bits_witnesses(self):
        """For every free key and input bit k = 2..15 some input moves the target by an
        order other than k (e chosen so the flip carries into or borrows from bit 16)."""
        inv9 = pow(9, -1, 1 << 32)

        def order(a, k, key, okey):
            d = (t.target(a ^ 1 << k, key, okey) - t.target(a, key, okey)) & MASK
            return (d & -d).bit_length() - 1

        for key in t.free_keys():
            for k in range(2, 16):
                step = max(1, (9 << k) // 64)
                es = [(1 << 16) - (9 << k) + d for d in range(0, 9 << k, step)] + \
                     [(1 << 16) + d for d in range(0, 9 << k, step)]
                self.assertTrue(any(all(order(((inv9 * (e - t.C4)) & MASK) ^ key, k, key, o) != k
                                        for o in t.OKEYS) for e in es), (key, k))

    def test_division_rules(self):
        """Each div-valuation template's power-of-two divisor is a shift template of the
        basic grammar, and every one of those is closed in the stored results."""
        import gzip
        import json
        closed = set()
        with gzip.open(Path(t.__file__).with_name('grammar3_tail_c5_results.jsonl.gz'), 'rt') as f:
            rows = [json.loads(line) for line in f]
        basic = t.templates()
        pins = {}
        for r in rows:
            if not r['full'] and r['run'] == 'main' and r['result'] == 'unsat':
                closed.add(g.canonical(basic[r['idx']]))
            elif not r['full'] and r['run'] == 'pin':
                pins.setdefault(r['idx'], []).append(r['result'])
        closed |= {g.canonical(basic[i]) for i, rs in pins.items() if len(rs) == 32 and set(rs) == {'unsat'}}
        div = t.templates(full=True)
        seen = 0
        for p in div:
            if t.division_rule(p) == 'div-valuation':
                twin = (p[0], ('shrk', p[1][1]), p[2])
                self.assertIn(g.canonical(twin), closed)
                seen += 1
        self.assertGreater(seen, 0)

    def test_pinned_hard_case(self):
        hard = (('aff', (0,)), ('shrk', (1,)), ('lin1', (1, 2)))
        for s in (15, 16, 17):
            self.assertEqual(t.solve(hard, 120, {1: s}, free_only=True)['result'], 'unsat')


if __name__ == '__main__':
    unittest.main()
