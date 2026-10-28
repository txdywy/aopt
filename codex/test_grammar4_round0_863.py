"""Checks for the round-0 four-instruction search and its exact bit-0 arguments."""
import collections
import random
import unittest

from codex import grammar3_head863 as g
from codex import grammar4_round0_863 as h

R = h.ROOTS[0]


class Round0FourTests(unittest.TestCase):
    def test_template_classes(self):
        ts = h.templates4()
        self.assertEqual(len(ts), 14150)
        cnt = collections.Counter()
        for p in ts:
            lk = h.lemma_key(p)
            cnt[h.sub_class(lk[1]) if lk else 'multi'] += 1
        self.assertEqual(cnt, {'bitwise': 10525, 'affine': 942, None: 1117, 'multi': 1566})

    def test_sub_class(self):
        cases = {
            (): 'bitwise',
            (('xk', (0,)),): 'bitwise',
            (('xk', (0,)), ('and', (0, 1))): 'bitwise',
            (('aff', (0,)),): 'affine',
            (('aff', (0,)), ('aff', (1,))): 'affine',
            (('aff', (0,)), ('xk', (1,))): 'affine',      # trailing XOR with a constant
            (('aff', (0,)), ('lin1', (0, 1))): 'affine',
            (('xk', (0,)), ('aff', (1,))): None,          # XOR before the multiply
            (('aff', (0,)), ('xor', (0, 1))): None,
        }
        for sub, want in cases.items():
            self.assertEqual(h.sub_class(list(sub)), want, sub)

    def test_bitwise_witnesses(self):
        y19 = h.y19_of(R)
        wit = h.bitwise_refuted(R)
        self.assertEqual(set(wit), set(range(32)))
        for sh, (v1, v2) in wit.items():
            self.assertLess(max(v1, v2), g.D30)
            for b in {0, sh, 19}:
                self.assertEqual(v1 >> b & 1, v2 >> b & 1)
            self.assertNotEqual(y19(v1), y19(v2))

    def test_affine_check_and_controls(self):
        self.assertEqual(h.affine_refuted(R), ())
        self.assertTrue(h.affine_top_refuted(R))
        # low 20 root bits 0: v ^ R is affine there, so mu = 4097 must be found
        r0 = R & ~((1 << 20) - 1)
        sol = h.affine_refuted(r0)
        self.assertTrue(any(mu == 4097 for mu, _, _ in sol))
        mu, nu, gg = sol[0]
        y19 = h.y19_of(r0)
        for v in random.Random(1).sample(range(1 << 20), 4096):
            self.assertEqual(((mu * v + nu) & 0xFFFFF) >> 19 & 1, y19(v) ^ (gg >> (v & 1) & 1))

    def test_shift_input(self):
        p = [('aff', (0,)), ('xk', (0,)), ('shrk', (1,)), ('xor', (2, 3))]
        self.assertEqual(h.shift_input(p, 2), [('aff', (0,))])
        p = [('xk', (0,)), ('aff', (1,)), ('shrk', (2,)), ('xor', (2, 3))]
        self.assertEqual(h.shift_input(p, 2), [('xk', (0,)), ('aff', (1,))])


class AuditTests(unittest.TestCase):
    def test_dependency_rule(self):
        from codex import grammar4_round0_863 as h4
        from codex.grammar4_round0_audit863 import bit0_support, dependency_excluded, flip_witnesses
        nat = [('xk', (0,)), ('aff', (1,)), ('shrk', (2,)), ('xor', (2, 3))]
        self.assertEqual(bit0_support(nat, {2: 19}), frozenset(range(20)))
        self.assertFalse(dependency_excluded(nat))  # the natural shape must stay open
        thin = [('xk', (0,)), ('shrk', (0,)), ('lin2', (1, 2)), ('shrk', (3,))]
        self.assertEqual(bit0_support(thin, {1: 5, 3: 1}), frozenset({0, 1, 5, 6}))
        self.assertIsNone(bit0_support(thin, {1: 5, 3: 3}))  # final shift >= 3: not injective
        self.assertTrue(dependency_excluded(thin))
        y19 = h4.y19_of(R)
        for j, v in flip_witnesses(R).items():
            self.assertNotEqual(y19(v), y19(v ^ (1 << j)))

    def test_division_rules(self):
        from codex.grammar3_division_audit863 import classify
        cases = {
            (('aff', (0,)), ('aff', (0,)), ('div_ab', (1, 2))): 'injective-a|even-mad-a',
            (('aff', (0,)), ('aff', (1,)), ('div_ab', (1, 2))): 'single-source',
            (('aff', (0,)), ('xk', (0,)), ('div_ba', (1, 2))): 'injective-a',  # numerator u ^ K
            (('aff', (0,)), ('xor', (0, 1)), ('div_ba', (1, 2))): None,    # numerator not unary
            (('aff', (0,)), ('shrk', (1,)), ('xor', (1, 2))): None,       # not a division
            (('shrk', (0,)), ('aff', (0,)), ('cdiv_ab', (1, 2))): 'small-image',
        }
        for p, want in cases.items():
            self.assertEqual(classify(list(p)), want, p)


if __name__ == '__main__':
    unittest.main()
