"""Check the three-instruction head grammar against the frozen ISA and controls."""
import random
import sys
from pathlib import Path
import unittest

import z3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from frozen_problem import Machine
from codex import grammar3_head863 as g

MASK = g.MASK
# form -> (frozen opcode, operand order) over named inputs a, b, c and constants k0, k1
FROZEN = {
    'aff': ('mad', 'a k0 k1'), 'sq': ('mad', 'a a k0'), 'sqx': ('mad', 'a a a'),
    'xk': ('^', 'a k0'), 'andk': ('&', 'a k0'), 'ork': ('|', 'a k0'), 'shrk': ('>>', 'a k0'),
    'kshl': ('<<', 'k0 a'), 'kshr': ('>>', 'k0 a'), 'sshl': ('<<', 'a a'), 'sshr': ('>>', 'a a'),
    'lin1': ('mad', 'a k0 b'), 'lin2': ('mad', 'b k0 a'), 'prod': ('mad', 'a b k0'),
    'xor': ('^', 'a b'), 'and': ('&', 'a b'), 'or': ('|', 'a b'),
    'shl_ab': ('<<', 'a b'), 'shl_ba': ('<<', 'b a'), 'shr_ab': ('>>', 'a b'), 'shr_ba': ('>>', 'b a'),
    'aab': ('mad', 'a a b'), 'bba': ('mad', 'b b a'), 'aba': ('mad', 'a b a'), 'abb': ('mad', 'a b b'),
    'mad_c': ('mad', 'a b c'), 'mad_b': ('mad', 'a c b'), 'mad_a': ('mad', 'b c a'),
    'divk': ('//', 'a k0'), 'kdiv': ('//', 'k0 a'), 'modk': ('%', 'a k0'), 'kmod': ('%', 'k0 a'),
    'cdivk': ('cdiv', 'a k0'), 'kcdiv': ('cdiv', 'k0 a'), 'ltk': ('<', 'a k0'),
    'klt': ('<', 'k0 a'), 'eqk': ('==', 'a k0'),
    'div_ab': ('//', 'a b'), 'div_ba': ('//', 'b a'), 'mod_ab': ('%', 'a b'), 'mod_ba': ('%', 'b a'),
    'cdiv_ab': ('cdiv', 'a b'), 'cdiv_ba': ('cdiv', 'b a'), 'lt_ab': ('<', 'a b'),
    'lt_ba': ('<', 'b a'), 'eq': ('==', 'a b'),
}
ARITY = {**{f: 1 for f in g.UNARY + g.EXTRA_UNARY}, **{f: 2 for f in g.BINARY + g.EXTRA_BINARY},
         **{f: 3 for f in g.TERNARY}}


def frozen(machine, code, args):
    core = machine.cores[0]
    core.scratch[:24] = [args[0]] * 8 + [args[1]] * 8 + [args[2] if len(args) > 2 else 0] * 8
    machine.scratch_write = {}
    if code == 'mad':
        machine.valu(core, 'multiply_add', 24, 0, 8, 16)
    else:
        machine.alu(core, code, 24, 0, 8)
    return machine.scratch_write[24]


class GrammarTests(unittest.TestCase):
    def test_every_form_matches_frozen_machine(self):
        self.assertEqual(set(FROZEN), set(ARITY))
        machine = Machine([], [], None)
        csem, zsem = g.ops(None), g.ops(z3)
        rng = random.Random(863)
        for form, (code, order) in FROZEN.items():
            for trial in range(40):
                vals = {'a': rng.getrandbits(32), 'b': rng.getrandbits(32), 'c': rng.getrandbits(32),
                        'k0': rng.getrandbits(32), 'k1': rng.getrandbits(32)}
                if trial < 4:
                    vals.update(a=[0, 1, MASK, 1 << 31][trial])
                if code in ('<<', '>>'):  # keep amounts small: huge left shifts allocate huge ints
                    names = order.split()
                    vals[names[1]] %= 64 if trial % 2 else 32
                if form == 'shrk':
                    vals['k0'] %= 32
                if code in ('//', '%', 'cdiv'):
                    vals[order.split()[1]] |= 1 << rng.randrange(32)
                args = [vals[n] for n in order.split()]
                want = frozen(machine, code, args)
                data = [vals[n] for n in 'abc'[:ARITY[form]]]
                ks = [vals['k0'], vals['k1']][:g.NCONST.get(form, 0)]
                self.assertEqual(csem[form](*data, ks), want, (form, vals))
                zv = z3.simplify(zsem[form](*[z3.BitVecVal(v, 32) for v in data],
                                            [z3.BitVecVal(v, 32) for v in ks]))
                self.assertEqual(zv.as_long(), want, (form, vals))

    def test_template_counts(self):
        self.assertEqual(len(g.templates()), 6072)
        self.assertEqual(len(g.templates(True)), 20849)
        self.assertEqual(len(g.templates(False, True)), 9394)

    def test_controls(self):
        ts = g.templates()
        std = next(p for p in ts if g.describe(p) == 'i1=aff(u) ; i2=shrk(i1) ; i3=xor(i1,i2)')
        saved = g.C1
        try:
            g.C1 = g.g19(1 << 31)  # absorbable key: the standard program must be found
            r = g.solve(std, 120)
            self.assertEqual(r['result'], 'sat')
            self.assertIsNone(g.verify_full(std, r['consts'], r['m'], r['c'], r['e']))
        finally:
            g.C1 = saved
        self.assertEqual(g.solve(std, 120)['result'], 'unsat')

    def test_c_checker(self):
        saved = g.C1
        try:
            g.C1 = g.g19(1 << 31)
            p = [('aff', (0,)), ('divk', (1,)), ('xor', (1, 2))]
            good = [[4294963199, 2127908117], [1 << 19], []]
            self.assertIsNone(g.verify_full(p, good, 1, 0, (1 << 31) - 1))
            self.assertIsNone(g.verify_full(p, good, 1, 0, (1 << 31) - 1, domain=g.D30))
            self.assertEqual(g.verify_full(p, [good[0], [0], []], 1, 0, (1 << 31) - 1), 0)
            bad = [good[0], [1 << 18], []]
            w = g.verify_full(p, bad, 1, 0, (1 << 31) - 1)
            got = g.run_program(p, w, bad, g.ops(None))
            self.assertNotEqual(got, g.target(w ^ ((1 << 31) - 1)))
        finally:
            g.C1 = saved


if __name__ == '__main__':
    unittest.main()
