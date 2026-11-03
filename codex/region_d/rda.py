"""Fiber reduction for single-SHR wirings whose shifted value is affine in z.

Setting. One SHR at position pos reads a runtime value v (index <= pos) and a private
constant shift s; every other instruction is a T-op. If the T-op wiring makes v an
affine function of z for every constant choice (v = A*z + B mod 2^32; MAD with at most
one non-constant factor, sums), the low c output bits can be decided at width c.

Fiber lemma. Fix a class z0 mod 2^c and let z = z0 + 2^c*y range over the fiber. All
T-op values are fixed mod 2^c on the fiber; only u = v >> s varies. The output mod 2^c
is H(z0, u mod 2^c) for a T-function H, and must equal tau(z0) = F(z0^K)^D mod 2^c.
So H(z0, w) == tau(z0) for every w in W(z0) = {u(z) mod 2^c : z in the fiber}.

Write A = 2^e*A1 (A1 odd) or A = 0, B = B_lo + 2^e*B_h with B_lo < 2^e, and
m(z0) = (A1*z0 + B_h) mod 2^c, delta = s - e. Then v mod 2^c = (B_lo + 2^e*m) mod 2^c,
and on the fiber v = B_lo + 2^e*m + 2^(c+e)*h where h takes every value in
[0, 2^(32-c-e)) (A1 odd).  Three exhaustive cases:
  case 1, delta >= c : u mod 2^c = h >> (s-c-e): every value in [0, 2^min(c, 32-s)).
  case 2, delta <= 0 (A == 0 mod 2^s, includes A = 0 and s = 0):
          u = (A'*z + B') mod 2^(32-s) with A' = A >> s, B' = B >> s, and
          v mod 2^c = ((u << s) + (B mod 2^s)) mod 2^c.
  case 3, 1 <= delta <= c-1 : u mod 2^c = (m' >> delta) | (r << (c-delta)) for every
          r in [0, 2^min(delta, 32-c-e)), where m' = m mod 2^min(c, 32-e) (bits [e, 32)
          of v hold only 32-e bits of m when e > 32-c).
s >= 32 gives u = 0 (case 1 with s = 32). Each case is a finite c-bit problem over
(z0, r); UNSAT of all three cases excludes the wiring at c output bits for every
32-bit constant choice and every shift amount. The case parameters (s, e, delta) are
symbolic; consistency v_T(z0) == model(z0) ties the model to the T-op constants.
"""
import random
import sys
import time
sys.path.insert(0, '/home/user/aopt')
import numpy as np
import z3
from rd2 import K1, K2, C4, D


def affine_kinds(order, wiring, upto):
    """Kind of each runtime value t_0..t_{upto-1}: 'c' const, 'a' affine in z, 'o' other."""
    kinds = ['a']                                   # z
    for i in range(upto):
        op, idx = order[i], wiring[i]
        pool = kinds + ['c', 'c']
        ks = [pool[j] for j in idx]
        if op == 'mad':
            a, b, c = ks
            if a == 'c' and b == 'c':
                p = 'c'
            elif 'o' in (a, b) or (a == 'a' and b == 'a'):
                p = 'o'
            else:
                p = 'a'
            if p == 'o' or c == 'o':
                k = 'o'
            elif p == 'c' and c == 'c':
                k = 'c'
            else:
                k = 'a'
        elif op == '>>':
            k = 'o'
        else:
            k = 'c' if all(x == 'c' for x in ks) else 'o'
        kinds.append(k)
    return kinds


def applicable(order, wiring):
    if list(order).count('>>') != 1:
        return False
    pos = order.index('>>')
    vi, ki = wiring[pos]
    if vi > pos or ki <= pos:                       # value must be runtime, amount constant
        return False
    return affine_kinds(order, wiring, pos)[vi] == 'a'


def tau_np(z, K, cm):
    y = z ^ K
    t = (9 * (((33 * y + (K1 & cm)) & cm) ^ ((16896 * y + (K2 & cm)) & cm)) + (C4 & cm)) & cm
    return t ^ (D & cm)


def tau_z3(z, K, cm):
    y = z ^ K
    return (9 * ((33 * y + (K1 & cm)) ^ (16896 * y + (K2 & cm))) + (C4 & cm)) ^ (D & cm)


def t_op(op, a, b, c, cm, is_np):
    if op == 'mad':
        v = a * b + c
        return v & cm if is_np else v
    if op == '^':
        return a ^ b
    if op == '&':
        return a & b
    return a | b


def run_prefix(order, wiring, ks, z, pos, cm, is_np):
    vals = [z]
    for i in range(pos):
        pool = vals + [ks[i][0], ks[i][1]]
        idx = wiring[i]
        args = [pool[j] for j in idx] + [None] * (3 - len(idx))
        vals.append(t_op(order[i], args[0], args[1], args[2], cm, is_np))
    return vals


def run_suffix(order, wiring, ks, vals, u, pos, cm, is_np):
    vals = vals + [u]
    for i in range(pos + 1, len(order)):
        pool = vals + [ks[i][0], ks[i][1]]
        idx = wiring[i]
        args = [pool[j] for j in idx] + [None] * (3 - len(idx))
        vals.append(t_op(order[i], args[0], args[1], args[2], cm, is_np))
    return vals[-1]


class Case:
    """One fiber case with symbolic parameters; builds z3 constraints and numpy checks."""

    def __init__(self, kind, c, S):
        self.kind, self.c, self.S = kind, c, S
        self.cm = (1 << c) - 1
        w = 8
        self.s = z3.BitVec('s', w)
        self.e = z3.BitVec('e', w)
        self.dl = z3.BitVec('dl', w)
        self.A1 = z3.BitVec('A1', c)
        self.Bh = z3.BitVec('Bh', c)
        self.Bl = z3.BitVec('Bl', c)
        s, e, dl = self.s, self.e, self.dl
        if kind == 1:
            S.add(z3.UGE(s, c), z3.ULE(s, 32))
        elif kind == 2:
            S.add(z3.ULE(s, 31))
            S.add(z3.LShR(self.Bl, self.cw(s)) == 0)
        else:
            S.add(z3.UGE(dl, 1), z3.ULE(dl, c - 1), z3.ULE(e, 31), z3.ULE(e + dl, 31), s == e + dl)
            S.add(z3.Extract(0, 0, self.A1) == 1)
            S.add(z3.LShR(self.Bl, self.cw(e)) == 0)

    def cw(self, x):
        return z3.ZeroExt(self.c - 8, x) if self.c > 8 else z3.Extract(self.c - 1, 0, x)

    def lim(self):
        """Number of free low bits of r (case 1: of u)."""
        c = self.c
        s, e, dl = self.s, self.e, self.dl
        if self.kind == 1:
            room = z3.If(z3.ULE(s, 32 - c), z3.BitVecVal(c, 8), 32 - s)
            return room
        if self.kind == 3:
            room = z3.If(z3.ULE(e, 32 - c), (32 - c) - e, z3.BitVecVal(0, 8))
            return z3.If(z3.ULE(dl, room), dl, room)
        return z3.BitVecVal(0, 8)

    def u_and_v(self, z, r):
        """z3 values of u and of the modelled v (None when unconstrained) for sample (z, r)."""
        c = self.c
        if self.kind == 1:
            return r, None
        if self.kind == 2:
            s = self.cw(self.s)
            u = self.A1 * z + self.Bh                       # A1 = A', Bh = B' here (any values)
            keep = z3.If(z3.UGT(self.s, 32 - c), z3.LShR(z3.BitVecVal(self.cm, c), s - (32 - c)),
                         z3.BitVecVal(self.cm, c))
            u = u & keep
            return u, (u << s) + self.Bl
        e, dl = self.cw(self.e), self.cw(self.dl)
        m = self.A1 * z + self.Bh
        # bits [e, 32) of v hold only 32-e bits of m when e > 32-c
        keep = z3.If(z3.UGT(self.e, 32 - c), z3.LShR(z3.BitVecVal(self.cm, c), e - (32 - c)),
                     z3.BitVecVal(self.cm, c))
        mu = m & keep
        u = z3.LShR(mu, dl) | (r << (c - dl))
        return u, self.Bl + (m << e)

    # numpy side ------------------------------------------------------------
    def np_params(self, mo):
        g = lambda x: mo.eval(x, model_completion=True).as_long()
        return dict(s=g(self.s), e=g(self.e), dl=g(self.dl), A1=g(self.A1), Bh=g(self.Bh), Bl=g(self.Bl))

    def np_lim(self, p):
        c = self.c
        if self.kind == 1:
            return c if p['s'] <= 32 - c else 32 - p['s']
        if self.kind == 3:
            room = (32 - c) - p['e'] if p['e'] <= 32 - c else 0
            return min(p['dl'], room)
        return 0

    def np_u_v(self, p, z, r):
        c, cm = self.c, self.cm
        if self.kind == 1:
            return r, None
        if self.kind == 2:
            s = p['s']
            u = (p['A1'] * z + p['Bh']) & cm
            if s > 32 - c:
                u = u & (cm >> (s - (32 - c)))
            v = (((u << s) & cm) + p['Bl']) & cm if s < 64 else p['Bl'] & cm
            return u, v
        m = (p['A1'] * z + p['Bh']) & cm
        mu = m & (cm >> (p['e'] - (32 - c))) if p['e'] > 32 - c else m
        u = (mu >> p['dl']) | ((r << (c - p['dl'])) & cm)
        v = (p['Bl'] + ((m << p['e']) & cm if p['e'] < 64 else 0)) & cm
        return u, v


def solve_case(order, wiring, kind, c=11, seconds=60.0, seed=1, target=None):
    """target: optional uint array of length 2^c to match instead of tau (K fixed 0)."""
    pos = order.index('>>')
    vi = wiring[pos][0]
    cm = (1 << c) - 1
    S = z3.SolverFor('QF_BV')
    case = Case(kind, c, S)
    ks = [(z3.BitVec(f'k{i}_0', c), z3.BitVec(f'k{i}_1', c)) for i in range(len(order))]
    K = z3.BitVec('K', c)
    if target is not None:
        S.add(K == 0)
    lim = case.lim()
    seen_z = set()

    def add(zv, rv):
        z = z3.BitVecVal(zv, c)
        r = z3.BitVecVal(rv, c)
        vals = run_prefix(order, wiring, ks, z, pos, cm, False)
        u, vmodel = case.u_and_v(z, r)
        if vmodel is not None and zv not in seen_z:
            S.add(vals[vi] == vmodel)
        seen_z.add(zv)
        out = run_suffix(order, wiring, ks, vals, u, pos, cm, False)
        want = z3.BitVecVal(int(target[zv]), c) if target is not None else tau_z3(z, K, cm)
        ok = out == want
        if kind == 2:
            S.add(ok)
        else:
            S.add(z3.Implies(z3.LShR(r, case.cw(lim)) == 0, ok))

    rng = random.Random(seed)
    pts = [(0, 0), (1, 0), (2, 0), (3, 0), (cm, 0), (1 << (c - 1), 0), (5, 1), (7, cm), (1, cm), (0, cm)]
    pts += [(rng.getrandbits(c), rng.getrandbits(c)) for _ in range(14)]
    pts += [(rng.getrandbits(c), 0) for _ in range(8)]
    for zv, rv in pts:
        add(zv, rv)
    zs = np.arange(1 << c, dtype=np.uint64).reshape(-1, 1)
    t0 = time.monotonic()
    it = 0
    while True:
        left = seconds - (time.monotonic() - t0)
        if left <= 0:
            return dict(result='timeout', case=kind, c=c, it=it)
        S.set(timeout=max(1, int(left * 1000)))
        st = S.check()
        it += 1
        if st == z3.unsat:
            return dict(result='unsat', case=kind, c=c, it=it, sec=round(time.monotonic() - t0, 3))
        if st == z3.unknown:
            return dict(result='timeout', case=kind, c=c, it=it)
        mo = S.model()
        p = case.np_params(mo)
        kv = [(np.uint64(mo.eval(a, model_completion=True).as_long()),
               np.uint64(mo.eval(b, model_completion=True).as_long())) for a, b in ks]
        Kv = mo.eval(K, model_completion=True).as_long()
        L = case.np_lim(p)
        pnp = {k: np.uint64(v) if k in ('A1', 'Bh', 'Bl') else v for k, v in p.items()}
        vals = run_prefix(order, wiring, kv, zs, pos, np.uint64(cm), True)
        if kind != 2:
            _, vmodel = case.np_u_v(pnp, zs, np.zeros((1, 1), np.uint64))
        else:
            _, vmodel = case.np_u_v(pnp, zs, None)
        if vmodel is not None:
            bad_v = np.nonzero((vals[vi] & np.uint64(cm)).ravel() != (vmodel & np.uint64(cm)).ravel())[0]
            if len(bad_v):
                zb = int(bad_v[rng.randrange(len(bad_v))])
                add(zb, 0)
                continue
        want = (np.asarray(target, dtype=np.uint64).reshape(-1, 1) if target is not None
                else tau_np(zs, np.uint64(Kv), np.uint64(cm)))
        vals_b = [np.broadcast_to(x, (1 << c, 1)) if not np.isscalar(x) else x for x in vals]
        nr = 1 << L if kind != 2 else 1
        blk = max(1, (1 << 20) >> c)                 # bound memory: <= 2^20 entries per block
        cex = None
        for r0 in range(0, nr, blk):
            rs = np.arange(r0, min(nr, r0 + blk), dtype=np.uint64).reshape(1, -1)
            u, _ = case.np_u_v(pnp, zs, rs)
            out = run_suffix(order, wiring, kv, vals_b, u, pos, np.uint64(cm), True)
            out = np.broadcast_to(out, (1 << c, rs.shape[1])) & np.uint64(cm)
            bad = np.argwhere(out != want)
            if len(bad):
                j = bad[rng.randrange(len(bad))]
                cex = (int(j[0]), int(rs[0, j[1]]) if kind != 2 else 0)
                break
        if cex is None:
            return dict(result='sat', case=kind, c=c, it=it, params=p, K=Kv,
                        consts=[(int(a), int(b)) for a, b in kv])
        add(*cex)


def solve(order, wiring, c=11, seconds=60.0, seed=1, target=None):
    res = {}
    for kind in (1, 2, 3):
        r = solve_case(order, wiring, kind, c, seconds, seed, target)
        res[kind] = r
    agg = 'unsat' if all(r['result'] == 'unsat' for r in res.values()) else (
        'sat' if any(r['result'] == 'sat' for r in res.values()) else 'timeout')
    return dict(result=agg, cases=res)


if __name__ == '__main__':
    o, w = ('mad', '>>', 'mad', 'mad'), ((0, 1, 2), (1, 2), (0, 1, 2), (0, 0, 3))
    print(applicable(o, w))
    t = time.time()
    r = solve(o, w, 11, 120)
    print(r['result'], {k: (v['result'], v.get('it'), v.get('sec')) for k, v in r['cases'].items()},
          '%.1fs' % (time.time() - t))
