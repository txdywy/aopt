"""Region D, demand-driven exact model.

Target: P(z) == F(z ^ K) ^ O, four instructions, at least one SHR with a constant
shift amount s in a band [s_lo, s_hi]. O is any of the four absorbable keys; all
agree with D below bit 15, so checking c <= 15 low output bits covers all four.

Soundness. Demand d(v) = number of low bits of v that must be exact.
  d(output) = c.  A T-op (MAD, XOR, AND, OR) with demand d needs each input at d.
  SHR(v, s) with demand d needs v at min(32, d + s_hi).
Each value is computed at width d(v) from its inputs truncated to d(v). The low d
bits of a T-op depend only on the low d bits of its inputs, and the low d bits of
v >> s are bits [s, s + d) of v, exact when v is exact to d + s_hi bits; at full
width 32 every value is exact. Hence the computed low c output bits equal those of
every uint32 program with the same wiring, constants and shift in the band, and an
UNSAT over all inputs in [0, 2^Wmax) excludes all of them.
"""
import sys, time, random
sys.path.insert(0, '/home/user/aopt')
import numpy as np
import z3
from codex.cegis_c1_absorb863 import K1, K2, C4

C5 = 0xB55A4F09
D = C5 ^ (C5 >> 16)


def F(x, m):
    return (9 * (((33 * x + K1) & m) ^ ((16896 * x + K2) & m)) + C4) & m


def operands(op, idx):
    """Pool indices that are data inputs (the SHR amount is a literal, not data)."""
    return idx[:1] if op == '>>' else idx


def demands(order, wiring, c, s_hi):
    n = len(order)
    d = [0] * (n + 1)                       # d[0] = z, d[i+1] = t_i
    d[n] = c
    kd = [[0, 0] for _ in range(n)]         # demand on private constants
    for i in range(n - 1, -1, -1):
        out = d[i + 1]
        if out == 0:
            continue
        op = order[i]
        need = min(32, out + s_hi) if op == '>>' else out
        for j in operands(op, wiring[i]):
            if j <= i:                      # z or an earlier result
                d[j] = max(d[j], need)
            else:                           # private constant k0 (j == i+1) or k1
                kd[i][j - i - 1] = max(kd[i][j - i - 1], need)
    return d, kd


def ztrunc(v, w):
    """Resize a bitvector to width w (truncate or zero-extend)."""
    if v.size() == w:
        return v
    if v.size() > w:
        return z3.Extract(w - 1, 0, v)
    return z3.ZeroExt(w - v.size(), v)


def build(order, wiring, ks, ss, x, d, kd):
    vals = [ztrunc(x, d[0]) if d[0] else None]
    for i, op in enumerate(order):
        w = d[i + 1]
        if w == 0:
            vals.append(None)
            continue
        pool = vals + [ks[i][0], ks[i][1]]
        idx = wiring[i]
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


def npsim(order, wiring, consts, xs, m):
    vals = [xs]
    for i, op in enumerate(order):
        k0, k1 = consts[i]
        pool = vals + [np.full_like(xs, k0), np.full_like(xs, k1)]
        idx = wiring[i]
        if op == 'mad':
            a, b, cc = (pool[j] for j in idx)
            v = (a * b + cc) & m
        elif op == '>>':
            v = pool[idx[0]] >> np.uint64(k0)
        else:
            a, b = (pool[j] for j in idx)
            v = a ^ b if op == '^' else a & b if op == '&' else a | b
        vals.append(v)
    return vals[-1]


def solve(order, wiring, s_lo, s_hi, c, seconds, seed=1, control=None):
    d, kd = demands(order, wiring, c, s_hi)
    Wmax = max(d)
    m = (1 << Wmax) - 1
    cm = (1 << c) - 1
    xor_key = 0 if control else D & cm
    if control is None:
        table = np.array([F(x, cm) for x in range(1 << c)], dtype=np.uint64)
    else:
        table = np.array([control(x) & cm for x in range(1 << c)], dtype=np.uint64)
    S = z3.Then('simplify', 'solve-eqs', 'bit-blast', 'sat').solver()
    ks, ss = [], [None] * len(order)
    for i, op in enumerate(order):
        w = max(1, max(kd[i]) or d[i + 1] or 1)
        a = z3.BitVec(f'k{i}_0', max(w, 6) if op == '>>' else w)
        b = z3.BitVec(f'k{i}_1', w)
        ks.append((a, b))
        if op == '>>':
            ss[i] = a
            S.add(z3.UGE(a, s_lo), z3.ULE(a, s_hi))
    K = z3.BitVec('K', c)

    def tgt(xc):
        y = xc ^ K
        if control is not None:
            return control(y)
        return 9 * ((33 * y + (K1 & cm)) ^ (16896 * y + (K2 & cm))) + (C4 & cm)

    def constrain(p):
        x = z3.BitVecVal(p, max(Wmax, 1))
        out = build(order, wiring, ks, ss, x, d, kd)
        S.add(ztrunc(out, c) == tgt(z3.BitVecVal(p & cm, c)) ^ z3.BitVecVal(xor_key, c))

    rng = random.Random(seed)
    pts = list(dict.fromkeys([0, 1, 2, 3, 4, 5, 7, 8, 16, m, m - 1, 1 << (Wmax - 1)]
                             + [rng.getrandbits(Wmax) for _ in range(16)]))
    for p in pts:
        constrain(p)
    t0 = time.monotonic()
    it = 0
    xs = np.arange(1 << Wmax, dtype=np.uint64) if Wmax <= 22 else None
    while True:
        left = seconds - (time.monotonic() - t0)
        if left <= 0:
            return dict(result='timeout', Wmax=Wmax, c=c, it=it)
        S.set(timeout=max(1, int(left * 1000)))
        st = S.check()
        it += 1
        if st == z3.unsat:
            return dict(result='unsat', Wmax=Wmax, c=c, it=it)
        if st == z3.unknown:
            return dict(result='timeout', Wmax=Wmax, c=c, it=it)
        mo = S.model()
        ev = lambda v: mo.eval(v, model_completion=True).as_long()
        consts = [(ev(a), ev(b)) for a, b in ks]
        k = ev(K)
        if xs is not None:
            got = npsim(order, wiring, consts, xs, np.uint64(m)) & np.uint64(cm)
            want = table[(xs & np.uint64(cm)) ^ np.uint64(k)] ^ np.uint64(xor_key)
            bad = np.nonzero(got != want)[0]
            if len(bad) == 0:
                return dict(result='sat', Wmax=Wmax, c=c, it=it, consts=consts, K=k)
            p = int(bad[rng.randrange(len(bad))])
        else:
            V = z3.SolverFor('QF_BV')
            xv = z3.BitVec('xv', Wmax)
            kv = [(z3.BitVecVal(a, ka.size()), z3.BitVecVal(b, kb.size()))
                  for (a, b), (ka, kb) in zip(consts, ks)]
            sv = [kv[i][0] if order[i] == '>>' else None for i in range(len(order))]
            out = build(order, wiring, kv, sv, xv, d, kd)
            xc = ztrunc(xv, c) ^ z3.BitVecVal(k, c)
            t = control(xc) if control is not None else \
                9 * ((33 * xc + (K1 & cm)) ^ (16896 * xc + (K2 & cm))) + (C4 & cm)
            V.add(ztrunc(out, c) != t ^ z3.BitVecVal(xor_key, c))
            V.set(timeout=max(1, int(left * 1000)))
            vr = V.check()
            if vr == z3.unsat:
                return dict(result='sat', Wmax=Wmax, c=c, it=it, consts=consts, K=k)
            if vr == z3.unknown:
                return dict(result='timeout', Wmax=Wmax, c=c, it=it)
            p = V.model().eval(xv, model_completion=True).as_long()
        pts.append(p)
        constrain(p)
