"""General exact solver: any number of SHRs, each SHR(a, b) with a, b runtime values or
the private constant k0 (never both constant).

Every SHR operand (value, amount, or constant) is demanded in full 32 bits and the
shift is Z3 LShR at 32 bits (0 for amounts >= 32, identity for 0), so the model is
exact for all constants; no band restriction on constant shift amounts is needed.
All other values use rd2's demand rule. UNSAT on the low c output bits over all
inputs excludes the wiring for every constant choice.
"""
import random
import sys
import time
sys.path.insert(0, '/home/user/aopt')
import z3
from rd2 import ztrunc, K1, K2, C4, D


def demands_g(order, wiring, c):
    n = len(order)
    d = [0] * (n + 1)
    d[n] = c
    kd = [[0, 0] for _ in range(n)]
    for i in range(n - 1, -1, -1):
        out = d[i + 1]
        if out == 0:
            continue
        idx = wiring[i]
        needs = [(j, 32) for j in idx] if order[i] == '>>' else [(j, out) for j in idx]
        for j, need in needs:
            if j <= i:
                d[j] = max(d[j], need)
            else:
                kd[i][j - i - 1] = max(kd[i][j - i - 1], need)
    return d, kd


def build_g(order, wiring, ks, x, d):
    vals = [ztrunc(x, d[0]) if d[0] else None]
    for i, op in enumerate(order):
        w = d[i + 1]
        if w == 0:
            vals.append(None)
            continue
        pool = vals + [ks[i][0], ks[i][1]]
        idx = wiring[i]
        if op == '>>':
            a = ztrunc(pool[idx[0]], 32)
            b = ztrunc(pool[idx[1]], 32)
            v = ztrunc(z3.LShR(a, b), w)
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


def solve(order, wiring, c, seconds, seed=1, control=None, extra=None):
    d, kd = demands_g(order, wiring, c)
    W = max(d)
    cm = (1 << c) - 1
    key = 0 if control else D & cm
    S = z3.Then('simplify', 'solve-eqs', 'bit-blast', 'sat').solver()
    ks = [(z3.BitVec(f'k{i}_0', max(1, max(kd[i]) or d[i + 1] or 1)),
           z3.BitVec(f'k{i}_1', max(1, max(kd[i]) or d[i + 1] or 1))) for i in range(len(order))]
    K = z3.BitVec('K', c)
    if extra is not None:
        extra(S, ks)                       # optional restriction, e.g. fix a shift literal

    def tgt(xc):
        y = xc ^ K
        if control is not None:
            return control(y)
        return 9 * ((33 * y + (K1 & cm)) ^ (16896 * y + (K2 & cm))) + (C4 & cm)

    def constrain(p):
        x = z3.BitVecVal(p, W)
        S.add(ztrunc(build_g(order, wiring, ks, x, d), c) == tgt(z3.BitVecVal(p & cm, c)) ^ z3.BitVecVal(key, c))

    rng = random.Random(seed)
    m = (1 << W) - 1
    pts = list(dict.fromkeys([0, 1, 2, 3, 4, 5, 7, 8, 16, 31, 32, 33, m, m - 1, 1 << (W - 1)]
                             + [rng.getrandbits(W) for _ in range(16)] + [rng.getrandbits(6) for _ in range(6)]))
    for p in pts:
        constrain(p)
    t0 = time.monotonic()
    it = 0
    while True:
        left = seconds - (time.monotonic() - t0)
        if left <= 0:
            return dict(result='timeout', W=W, c=c, it=it)
        S.set(timeout=max(1, int(left * 1000)))
        st = S.check()
        it += 1
        if st == z3.unsat:
            return dict(result='unsat', W=W, c=c, it=it)
        if st == z3.unknown:
            return dict(result='timeout', W=W, c=c, it=it)
        mo = S.model()
        consts = [(mo.eval(a, model_completion=True).as_long(), mo.eval(b, model_completion=True).as_long()) for a, b in ks]
        k = mo.eval(K, model_completion=True).as_long()
        V = z3.SolverFor('QF_BV')
        xv = z3.BitVec('xv', W)
        kv = [(z3.BitVecVal(a, ka.size()), z3.BitVecVal(b, kb.size())) for (a, b), (ka, kb) in zip(consts, ks)]
        out = build_g(order, wiring, kv, xv, d)
        xc = ztrunc(xv, c) ^ z3.BitVecVal(k, c)
        t = control(xc) if control is not None else 9 * ((33 * xc + (K1 & cm)) ^ (16896 * xc + (K2 & cm))) + (C4 & cm)
        V.add(ztrunc(out, c) != t ^ z3.BitVecVal(key, c))
        V.set(timeout=max(1, int(left * 1000)))
        vr = V.check()
        if vr == z3.unsat:
            return dict(result='sat', W=W, c=c, it=it, consts=consts, K=k)
        if vr == z3.unknown:
            return dict(result='timeout', W=W, c=c, it=it)
        pts.append(V.model().eval(xv, model_completion=True).as_long())
        constrain(pts[-1])
