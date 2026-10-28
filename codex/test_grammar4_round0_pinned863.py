"""Checks for the pinned-shift search of the round-0 templates and its exact rules."""
import random
import subprocess
import tempfile
import unittest

from codex import grammar3_head863 as g
from codex import grammar4_round0_863 as h
from codex import grammar4_round0_bit0_863 as b0
from codex import grammar4_round0_pinned863 as q
from codex import grammar4_round0_pinned_audit863 as audit

R = h.ROOTS[0]


def move(p, kv, v, j, m):
    sem = g.ops(None)
    return (g.run_program(p, v ^ (1 << j), kv, sem) - g.run_program(p, v, kv, sem)) % m


def random_consts(p, amounts, rng):
    kv = [[rng.randrange(1 << 32) for _ in range(g.NCONST.get(f, 0))] for f, _ in p]
    for k, s in amounts.items():
        kv[k] = [s]
    return kv


class PinnedTests(unittest.TestCase):
    def test_classes(self):
        ts = h.templates4()
        op = q.open_templates()
        self.assertEqual(len(op), 228)
        mask = [i for i in op if q.single_use_mask(ts[i])]
        self.assertEqual(len(mask), 17)
        self.assertEqual(sum(1 for i in op if len(q.shifts(ts[i])) > 1), 75)
        self.assertEqual(sum(1 for i in op if q.xk_aff_input(ts[i], q.shifts(ts[i])[0]) is not None), 13)

    def test_xk_affine_pinning(self):
        self.assertEqual(b0.xk_affine_keys(), (0, 262142, 262144, 524286))
        sols = b0.xk_affine_solutions(R)
        self.assertEqual(len(sols), 32)
        # the natural program K = R, mu = 4097, nu = C0 must stay allowed
        self.assertIn((R & 0xFFFFF, 4097, g.C0 & 0xFFFFF), sols)

    def test_rigid_witnesses(self):
        wit = b0.rigid_witnesses(R)
        self.assertEqual(sorted(wit), list(range(19, 25)))
        for s, (j, v1, v2) in wit.items():
            m = 1 << (32 - s)
            self.assertGreaterEqual(j, s)
            self.assertEqual(v1 % m, v2 % m)
            self.assertEqual(v1 >> j & 1, v2 >> j & 1)
            t = lambda v: (g.target((v ^ R) ^ (1 << j)) - g.target(v ^ R)) % m
            self.assertNotEqual(t(v1), t(v2))

    def test_rigid_rule_numerically(self):
        """Rules 4 and 5 claim equal output moves on the witness pair for any constants."""
        ts = h.templates4()
        rng = random.Random(11)
        wit = b0.rigid_witnesses(R)
        checked = 0
        for i in q.open_templates():
            p = ts[i]
            sh = q.shifts(p)
            if len(sh) == 1 and b0.rigid_shape(p):
                for s, (j, v1, v2) in wit.items():
                    kv = random_consts(p, {sh[0]: s}, rng)
                    m = 1 << (32 - s)
                    self.assertEqual(move(p, kv, v1, j, m), move(p, kv, v2, j, m), (i, s))
                    checked += 1
            elif len(sh) == 2 and i % 7 == 0:
                for _ in range(40):
                    am = dict(zip(sh, (rng.randrange(32), rng.randrange(32))))
                    hit = b0.rigid_any(p, am, R)
                    if hit:
                        e, j = hit
                        v1, v2 = b0.move_witness(e, j, R)
                        kv = random_consts(p, am, rng)
                        self.assertEqual(move(p, kv, v1, j, 1 << e), move(p, kv, v2, j, 1 << e), (i, am))
                        checked += 1
        self.assertGreater(checked, 500)

    def test_rigid_control(self):
        """A target without the XOR, y + (y >> 19), has rigid moves: no witness may exist."""
        def tadd(x):
            y = (4097 * x + g.C0) & g.MASK
            return (y + (y >> 19)) & g.MASK
        rng = random.Random(2)
        m = 1 << 13
        for _ in range(5000):
            v1 = rng.randrange(g.D30)
            v2 = (v1 & (m - 1)) | (v1 & (1 << 29)) | (rng.randrange(g.D30) & ~(m - 1) & ~(1 << 29))
            d = lambda v: (tadd((v ^ R) ^ (1 << 29)) - tadd(v ^ R)) % m
            self.assertEqual(d(v1), d(v2))

    def test_exhaustive_programs_find_known_solutions(self):
        with tempfile.TemporaryDirectory() as d:
            exe = b0._build(b0.BF2_C % dict(C0=g.C0, EXPR='a*(u ^ b) + C0'), 'bf2', d)
            k = (R & 0xFFFFF) % 1024    # the slice holding b = R mod 2^20
            out = subprocess.run([str(exe), str(R), str(k), '1024'], check=True,
                                 capture_output=True, text=True).stdout
            self.assertIn(f'sol 4097 {R & 0xFFFFF} 0 0', out)
            exe = b0._build(b0.QUAD_C % dict(C0=g.C0), 'quad', d)
            root0 = R & ~0xFFFFF    # low 20 bits 0: y19 is bit 19 of 4097u + C0
            out = subprocess.run([str(exe), str(root0), '0', '4096'], check=True,
                                 capture_output=True, text=True).stdout
            self.assertIn(f'sol 0 4097 0 {g.C0 & 0xFFFFF} 1', out)

    def test_mask_rule_facts(self):
        self.assertTrue(audit.mask_rule_holds(R))

    def test_xor_aff_steps(self):
        """The two facts the xor-aff rule rests on, checked exhaustively at a small width."""
        for sw in (6, 7, 8):
            n, half = 1 << sw, 1 << (sw - 1)
            for mu in range(1, n, 2):
                for nu in range(n):
                    img = {(mu * x + nu) % n for x in range(half)}
                    is_half = img in ({*range(half)}, {*range(half, n)})
                    self.assertEqual(is_half, mu in (1, n - 1) and nu % half == (0 if mu == 1 else half - 1),
                                     (sw, mu, nu))
        rng = random.Random(4)
        for _ in range(2000):   # bit s of u ^ (mu u + nu) is A_s ^ carry_s with A = (mu - 1) u + nu
            s, mu, nu, u = rng.randrange(21, 30), rng.randrange(1, 1 << 32, 2), rng.randrange(1 << 32), \
                rng.randrange(g.D30)
            a = ((mu - 1) * u + nu) & g.MASK
            carry = ((u & ((1 << s) - 1)) + (a & ((1 << s) - 1))) >> s
            self.assertEqual((u ^ ((mu * u + nu) & g.MASK)) >> s & 1, (a >> s & 1) ^ carry)

    def test_pair_read_checks(self):
        self.assertEqual(b0.pair_read_solutions(R), ())
        # control: plant y19 ^ u_5 as the target; G1 must recover mu = 4097, nu = C0 with G = u_5
        src = b0.PAIR_C % dict(C0=g.C0)
        planted = src.replace('return ((4097u * (u ^ R) + C0) >> 19) & 1;',
                              'return (((4097u * (u ^ R) + C0) >> 19) & 1) ^ ((u >> 5) & 1);')
        self.assertNotEqual(src, planted)
        with tempfile.TemporaryDirectory() as d:
            exe = b0._build(planted, 'pair', d)
            out = subprocess.run([str(exe), str(R & ~0xFFFFF)], check=True, capture_output=True,
                                 text=True).stdout
        self.assertIn(f'sol 0 5 12 4097 {g.C0 & 0xFFFFF} 1', out)
        wit = audit.psi_witnesses(R)
        y19 = h.y19_of(R)
        for qq, (u, v) in wit.items():
            self.assertEqual(u % (2 << qq), v % (2 << qq))
            self.assertNotEqual(y19(u) ^ (u >> 19 & 1), y19(v) ^ (v >> 19 & 1))
        for (b0_, b19), (u, v) in audit.class_witnesses(R).items():
            self.assertEqual((u & 1, u >> 19 & 1, v & 1, v >> 19 & 1), (b0_, b19, b0_, b19))
            self.assertNotEqual(y19(u), y19(v))

    def test_multi_reductions(self):
        p = [('aff', (0,)), ('shrk', (1,)), ('shrk', (2,)), ('lin2', (0, 3))]
        self.assertEqual(audit.merge_nested_shifts(p, {1: 5, 2: 14}),
                         ([('aff', (0,)), ('shrk', (1,)), ('lin2', (0, 2))], {1: 19}))
        self.assertIsNone(audit.merge_nested_shifts(p, {1: 20, 2: 14}))
        p = [('shrk', (0,)), ('lin1', (0, 1)), ('shrk', (2,)), ('xk', (3,))]
        self.assertEqual(audit.image_bits(p, {0: 1, 2: 3})[-1], 29)   # (... >> 3) has 29 bits
        self.assertEqual(audit.image_bits(p, {0: 1, 2: 2})[-1], 30)
        ts = h.templates4()
        self.assertEqual([audit.two_read_shape(ts[i]) for i in (178, 179, 187, 188, 921)],
                         [(1, 2), (1, 2), (1, 2), (1, 2), None])

    def test_valuation_rule(self):
        """Target moves have valuation exactly j - 19; the shape the rule needs is detected."""
        v2 = lambda x: (x & -x).bit_length() - 1
        rng = random.Random(8)
        for _ in range(3000):
            v, j = rng.randrange(g.D30), rng.randrange(20, 30)
            m, c = rng.randrange(1, 1 << 32, 2), rng.randrange(1 << 32)
            t = lambda w: (m * g.target(w ^ R) + c) & g.MASK
            self.assertEqual(v2((t(v ^ (1 << j)) - t(v)) & g.MASK), j - 19)
        ts = h.templates4()
        self.assertEqual(audit.valuation_bound(ts[2585]), 23)    # lin1(i1, i3)
        self.assertEqual(audit.valuation_bound(ts[2594]), 22)    # xor(i2, i3)
        self.assertIsNone(audit.valuation_bound(ts[921]))        # two shifts
        # the contradiction it derives: no shaped program moves with valuations (1, 10) at j = 20, 29
        sem = g.ops(None)
        shaped = [p for p in ts if audit.valuation_bound(p)]
        for _ in range(4000):
            p = rng.choice(shaped)
            k = len(p) - 2
            am = {k: rng.randrange(audit.valuation_bound(p), 32)}
            kv = random_consts(p, am, rng)
            v = rng.randrange(g.D30)
            run = lambda w: g.run_program(p, w, kv, sem)
            d = [(run(v ^ (1 << j)) - run(v)) & g.MASK for j in (20, 29)]
            self.assertFalse(d[0] and d[1] and (v2(d[0]), v2(d[1])) == (1, 10))

    def test_valuation_sets_contain_observed_moves(self):
        ts = h.templates4()
        sem = g.ops(None)
        rng = random.Random(22)
        v2 = lambda x: (x & -x).bit_length() - 1 if x else audit._INF
        for _ in range(1500):
            p = rng.choice(ts)
            sh = q.shifts(p)
            am = {k: rng.randrange(32) for k in sh}
            j = rng.randrange(20, 30)
            V = audit.move_valuations(p, am, j)
            kv = random_consts(p, am, rng)
            v = rng.randrange(g.D30)
            d = (g.run_program(p, v ^ (1 << j), kv, sem) - g.run_program(p, v, kv, sem)) & g.MASK
            self.assertIn(v2(d), V, (p, am, j))

    def test_window_checks(self):
        self.assertEqual(b0.window_checks(R), ((), (), ()))
        e1, e2, e3 = b0.window_checks(R & ~0xFFFFF)      # control root: the natural mu survives
        self.assertIn(4097, e1)
        self.assertTrue(e3)
        ts = h.templates4()
        self.assertTrue(audit.final_window_shape(ts[1191]) and audit.final_window_facts(R))

    def test_coupled_valuation_rule(self):
        ts = h.templates4()
        sem = g.ops(None)
        rng = random.Random(33)
        v2 = lambda x: (x & -x).bit_length() - 1 if x else 64
        seen = 0
        for _ in range(20000):
            p = rng.choice(ts)
            sh = q.shifts(p)
            am = {k: rng.randrange(32) for k in sh}
            if not audit.coupled_valuation_refuted(p, am):
                continue
            seen += 1
            kv = random_consts(p, am, rng)
            v = rng.randrange(g.D30)
            run = lambda w: g.run_program(p, w, kv, sem)
            d = [v2((run(v ^ (1 << j)) - run(v)) & g.MASK) for j in (20, 29)]
            self.assertNotEqual(d, [1, 10], (p, am))
        self.assertGreater(seen, 200)

    def test_bit20_exhaustive_control(self):
        with tempfile.TemporaryDirectory() as d:
            # bit 20 of 2(a(u ^ b) + C0) is bit 19 of a(u ^ b) + C0: a = 4097, b = R mod 2^20 is y19
            exe = b0._build(b0.BFW_C % dict(C0=g.C0, EXPR='2u*(a*(u ^ b) + C0)', B=20), 'bfw', d)
            k = (R & 0xFFFFF) % 512
            out = subprocess.run([str(exe), str(R), str(k), '512'], check=True, capture_output=True,
                                 text=True).stdout
        self.assertIn(f'sol 4097 {R & 0xFFFFF} 0 0', out)

    def test_drop_zero_shifts(self):
        p = [('aff', (0,)), ('shrk', (1,)), ('shrk', (1,)), ('lin1', (2, 3))]
        r, am = audit.drop_zero_shifts(p, {1: 0, 2: 19})
        self.assertEqual(r, [('aff', (0,)), ('shrk', (1,)), ('lin1', (1, 2))])
        self.assertEqual(am, {1: 19})


if __name__ == '__main__':
    unittest.main()
