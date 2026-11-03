"""Region D solver with shared monomials for an all-MAD pre-shift chain.

Same exact demand model and soundness as rd2. The only change is how values
whose demand exceeds c (the shift's input chain) are encoded: when every
instruction feeding the shift is a MAD, each such value is expanded in Python
into a polynomial in the private constants with concrete, input-dependent
integer coefficients. Monomials such as k0*k0 or k0*k2 are built once as Z3
terms and shared by all samples, so a sample contributes only additions and
multiplications by concrete numbers. The function computed is identical.
"""
import sys
sys.path.insert(0, '/home/user/aopt')
import z3
import rd2
from rd2 import demands, ztrunc


def pre_chain(order, wiring, d, c):
    """Indices (in pool numbering, t_i -> i+1) of values computed above width c."""
    return [i for i in range(len(order)) if d[i + 1] > c]


def all_mad_chain(order, wiring, d, c):
    """True when every value above width c is z, a constant or a MAD of such values.

    A shift output that feeds another shift is also above width c but has no
    polynomial form, so any such chain falls back to rd2.
    """
    hi = set(pre_chain(order, wiring, d, c))
    for i in hi:
        if order[i] != 'mad':
            return False
        for j in wiring[i]:
            if 1 <= j <= i and (j - 1) not in hi:      # an earlier result that is not a hi MAD
                return False
    return True


class Poly:
    __slots__ = ('t', 'm')

    def __init__(self, t, m):
        self.t = t            # {monomial(tuple of var ids, sorted with repetition): coeff}
        self.m = m

    @staticmethod
    def const(v, m):
        return Poly({(): v & m} if v & m else {}, m)

    @staticmethod
    def var(k, m):
        return Poly({(k,): 1}, m)

    def __add__(self, o):
        t = dict(self.t)
        for mono, c in o.t.items():
            v = (t.get(mono, 0) + c) & self.m
            if v:
                t[mono] = v
            else:
                t.pop(mono, None)
        return Poly(t, self.m)

    def __mul__(self, o):
        t = {}
        for m1, c1 in self.t.items():
            for m2, c2 in o.t.items():
                mono = tuple(sorted(m1 + m2))
                v = (t.get(mono, 0) + c1 * c2) & self.m
                if v:
                    t[mono] = v
                else:
                    t.pop(mono, None)
        return Poly(t, self.m)


def solve(order, wiring, s_lo, s_hi, c, seconds, seed=1, control=None):
    d, kd = demands(order, wiring, c, s_hi)
    if not all_mad_chain(order, wiring, d, c):
        return rd2.solve(order, wiring, s_lo, s_hi, c, seconds, seed, control)
    Wmax = max(d)
    hi = set(pre_chain(order, wiring, d, c))
    mono_cache = {}
    sym = {}

    orig_build = rd2.build

    def build(order_, wiring_, ks, ss, x, d_, kd_):
        # x is a concrete BitVecVal of width Wmax when called from the synthesis loop
        if not z3.is_bv_value(x):
            return orig_build(order_, wiring_, ks, ss, x, d_, kd_)
        xv = x.as_long()
        W = Wmax
        m = (1 << W) - 1
        polys = [Poly.const(xv, m)]
        vals = [ztrunc(x, d_[0])]
        for i, op in enumerate(order_):
            w = d_[i + 1]
            idx = wiring_[i]
            if i in hi and op == 'mad':
                pool = polys + [Poly.var((i, 0), m), Poly.var((i, 1), m)]
                a, b, cc = (pool[j] for j in idx)
                p = a * b + cc
                polys.append(p)
                vals.append(ztrunc(to_z3(p, ks, W), w))
                continue
            polys.append(None)
            pool = vals + [ks[i][0], ks[i][1]]
            if op == '>>':
                src = pool[idx[0]]
                v = ztrunc(z3.LShR(src, ztrunc(ss[i], src.size())), w)
            else:
                args = [ztrunc(pool[j], w) for j in idx]
                if op == 'mad':
                    v = args[0] * args[1] + args[2]
                elif op == '^':
                    v = args[0] ^ args[1]
                elif op == '&':
                    v = args[0] & args[1]
                else:
                    v = args[0] | args[1]
            vals.append(v)
        return vals[-1]

    def to_z3(p, ks, W):
        if not p.t:
            return z3.BitVecVal(0, W)
        acc = None
        for mono, coeff in p.t.items():
            if mono not in mono_cache:
                term = None
                for (i, slot) in mono:
                    f = ztrunc(ks[i][slot], W)
                    term = f if term is None else term * f
                mono_cache[mono] = term
            term = mono_cache[mono]
            piece = z3.BitVecVal(coeff, W) if term is None else (
                term if coeff == 1 else z3.BitVecVal(coeff, W) * term)
            acc = piece if acc is None else acc + piece
        return acc

    rd2.build = build
    try:
        return rd2.solve(order, wiring, s_lo, s_hi, c, seconds, seed, control)
    finally:
        rd2.build = orig_build
