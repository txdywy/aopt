"""Fiber model for single constant-shift wirings whose shifted value v is ANY polynomial in
z (MAD-only cone, degree <= 4 for two MADs before the shift), v = sum_i a_i z^i mod 2^32
with the a_i polynomials in the private constants (computed once per query).

Exact fiber relation for c >= 11 (3c >= 32): for z = z0 + 2^c*y,
    v(z) = V0 + 2^c*y*D1 + 2^(2c)*y^2*D2   (mod 2^32),
    V0 = v(z0), D1 = v'(z0) = sum i a_i z0^(i-1), D2 = v''(z0)/2 = sum C(i,2) a_i z0^(i-2)
(Taylor expansion of an integer polynomial; terms of order >= 3 carry 2^(3c) == 0).
Per class let d = v2(D1 mod 2^c). If d < c, y -> y*D1 + 2^c*y^2*D2 is 2^d times a
T-function of y with odd linear part and even quadratic part, hence a bijection of
Z/2^(32-c-d): on the fiber v = (V0 mod 2^(c+d)) + 2^(c+d)*h for EVERY h in
[0, 2^(32-c-d)). If D1 == 0 mod 2^c the explicit formula is used with y itself.
Samples (z0, h, y); the shift literal s is free in [0, 32] (32 = all s >= 32) or split.
UNSAT excludes the wiring at c output bits for every 32-bit constant choice and shift.
"""
import random
import sys
import time
sys.path.insert(0, '/home/user/aopt')
import z3
from rda import tau_z3

MAXDEG = 4


def _op(op, a, b, c):
    if op == 'mad':
        return a * b + c
    if op == '^':
        return a ^ b
    if op == '&':
        return a & b
    return a | b


def degrees(order, wiring, upto):
    deg = [1]
    for i in range(upto):
        pool = deg + [0, 0]
        ds = [pool[j] for j in wiring[i]]
        if order[i] != 'mad' or any(d is None for d in ds):
            deg.append(None)
        else:
            deg.append(max(ds[0] + ds[1], ds[2]))
    return deg


def analyse(order, wiring):
    if list(order).count('>>') != 1:
        return None
    pos = order.index('>>')
    vi, ki = wiring[pos]
    if vi > pos or ki <= pos:
        return None
    d = degrees(order, wiring, pos)[vi]
    if d is None or d > MAXDEG:
        return None
    return dict(pos=pos, vi=vi, ki=ki, deg=d)


def applicable(order, wiring):
    return analyse(order, wiring) is not None


def coeffs(order, wiring, k32, upto):
    """Coefficient lists (a_0..a_MAXDEG) of each polynomial value; None if not polynomial."""
    zero = z3.BitVecVal(0, 32)
    one = z3.BitVecVal(1, 32)

    def const(k):
        return [k] + [zero] * MAXDEG
    P = [[zero, one] + [zero] * (MAXDEG - 1)]
    for i in range(upto):
        pool = P + [const(k32[i][0]), const(k32[i][1])]
        if order[i] != 'mad':
            P.append(None)
            continue
        a, b, c = [pool[j] for j in wiring[i]]
        if a is None or b is None or c is None:
            P.append(None)
            continue
        prod = [zero] * (MAXDEG + 1)
        for p in range(MAXDEG + 1):
            for q in range(MAXDEG + 1 - p):
                if z3.is_bv_value(a[p]) and a[p].as_long() == 0:
                    continue
                if z3.is_bv_value(b[q]) and b[q].as_long() == 0:
                    continue
                prod[p + q] = prod[p + q] + a[p] * b[q]
        P.append([z3.simplify(prod[t] + c[t]) for t in range(MAXDEG + 1)])
    return P


def jets(cof, z0v):
    """(V0, D1, D2) at the concrete class representative z0v (Python int), as 32-bit terms."""
    M = (1 << 32) - 1
    V0 = D1 = D2 = z3.BitVecVal(0, 32)
    for i, a in enumerate(cof):
        if z3.is_bv_value(a) and a.as_long() == 0:
            continue
        V0 = V0 + a * z3.BitVecVal(pow(z0v, i, 1 << 32), 32)
        if i >= 1:
            D1 = D1 + a * z3.BitVecVal((i * pow(z0v, i - 1, 1 << 32)) & M, 32)
        if i >= 2:
            D2 = D2 + a * z3.BitVecVal(((i * (i - 1) // 2) * pow(z0v, i - 2, 1 << 32)) & M, 32)
    return V0, D1, D2


def solve(order, wiring, c=11, seconds=120.0, seed=1, target=None, s_range=(0, 32)):
    assert c >= 11, 'the exact fiber relation needs 3c >= 32'
    an = analyse(order, wiring)
    assert an is not None
    pos, vi, ki = an['pos'], an['vi'], an['ki']
    cm = (1 << c) - 1
    hw = 32 - c
    S = z3.SolverFor('QF_BV')
    k32 = [(z3.BitVec(f'k{i}_0', 32), z3.BitVec(f'k{i}_1', 32)) for i in range(len(order))]
    K = z3.BitVec('K', c)
    if target is not None:
        S.add(K == 0)
    slot = ki - pos - 1
    S.add(z3.UGE(k32[pos][slot], s_range[0]))
    if s_range[1] < 32:
        S.add(z3.ULE(k32[pos][slot], s_range[1]))
    co = coeffs(order, wiring, k32, pos)[vi]

    def ctz(x):
        d = z3.BitVecVal(c, 32)
        for b in range(c - 1, -1, -1):
            d = z3.If(z3.Extract(b, b, x) == 1, z3.BitVecVal(b, 32), d)
        return d

    def build(kk, cof, z0v, z0, h, y):
        kc = [(z3.Extract(c - 1, 0, a), z3.Extract(c - 1, 0, b)) for a, b in kk]
        vals = [z0]
        for i in range(pos):
            pool = vals + [kc[i][0], kc[i][1]]
            a = [pool[j] for j in wiring[i]] + [None] * (3 - len(wiring[i]))
            vals.append(_op(order[i], a[0], a[1], a[2]))
        V0, D1, D2 = jets(cof, z0v)
        Dl = z3.Extract(c - 1, 0, D1)
        d = ctz(Dl)
        width = d + c
        lowmask = z3.If(z3.UGE(width, 32), z3.BitVecVal(0xFFFFFFFF, 32), (z3.BitVecVal(1, 32) << width) - 1)
        Hx = z3.ZeroExt(c, h)
        Yx = z3.ZeroExt(c, y)
        v_free = (V0 & lowmask) + (Hx << width)
        v_exact = V0 + ((Yx * D1) << c) + ((Yx * Yx * D2) << (2 * c))
        sh = kk[pos][slot]
        outs = []
        for v in (v_free, v_exact):
            u = z3.Extract(c - 1, 0, z3.LShR(v, sh))
            vv = vals + [u]
            for i in range(pos + 1, len(order)):
                pool = vv + [kc[i][0], kc[i][1]]
                a = [pool[j] for j in wiring[i]] + [None] * (3 - len(wiring[i]))
                vv.append(_op(order[i], a[0], a[1], a[2]))
            outs.append(vv[-1])
        room = z3.BitVecVal(32 - c, 32) - d
        h_ok = z3.If(z3.UGE(d, 32 - c), Hx == 0, z3.LShR(Hx, room) == 0)
        cons = [vals[vi] == z3.Extract(c - 1, 0, V0)]
        return outs, Dl, h_ok, cons

    def constrain(zv, hv, yv):
        z0 = z3.BitVecVal(zv, c)
        (of, oe), Dl, h_ok, cons = build(k32, co, zv, z0, z3.BitVecVal(hv, hw), z3.BitVecVal(yv, hw))
        S.add(*cons)
        w = z3.BitVecVal(int(target[zv]), c) if target is not None else tau_z3(z0, K, cm)
        S.add(z3.If(Dl != 0, z3.Implies(h_ok, of == w), oe == w))

    rng = random.Random(seed)
    hm = (1 << hw) - 1
    pts = [(0, 0, 0), (1, 0, 1), (2, 1, 0), (3, 0, hm), (cm, hm, 1), (1, 1, 2), (5, 2, 3), (7, hm, hm)]
    pts += [(rng.getrandbits(c), rng.getrandbits(hw), rng.getrandbits(hw)) for _ in range(14)]
    pts += [(rng.getrandbits(c), rng.getrandbits(rng.randrange(1, hw)), rng.getrandbits(4)) for _ in range(8)]
    for p in pts:
        constrain(*p)
    t0 = time.monotonic()
    it = 0
    while True:
        left = seconds - (time.monotonic() - t0)
        if left <= 0:
            return dict(result='timeout', it=it)
        S.set(timeout=max(1, int(left * 1000)))
        st = S.check()
        it += 1
        if st == z3.unsat:
            return dict(result='unsat', it=it)
        if st == z3.unknown:
            return dict(result='timeout', it=it)
        mo = S.model()
        kk = [(mo.eval(a, model_completion=True), mo.eval(b, model_completion=True)) for a, b in k32]
        Kv = mo.eval(K, model_completion=True)
        sub = [(a, va) for (a, b), (va, vb) in zip(k32, kk)] + [(b, vb) for (a, b), (va, vb) in zip(k32, kk)]
        cof = [z3.simplify(z3.substitute(x, *sub)) for x in co]
        # verification: the class representative must stay symbolic, so evaluate the jets
        # with a symbolic z0 via the polynomial itself (cof are now constants)
        V = z3.SolverFor('QF_BV')
        zv = z3.BitVec('zv', c)
        hv = z3.BitVec('hv', hw)
        yv = z3.BitVec('yv', hw)
        Z = z3.ZeroExt(32 - c, zv)
        V0 = D1 = D2 = z3.BitVecVal(0, 32)
        for i, a in enumerate(cof):
            V0 = V0 + a * pw(Z, i)
            if i >= 1:
                D1 = D1 + a * i * pw(Z, i - 1)
            if i >= 2:
                D2 = D2 + a * (i * (i - 1) // 2) * pw(Z, i - 2)
        kc = [(z3.Extract(c - 1, 0, a), z3.Extract(c - 1, 0, b)) for a, b in kk]
        vals = [zv]
        for i in range(pos):
            pool = vals + [kc[i][0], kc[i][1]]
            a = [pool[j] for j in wiring[i]] + [None] * (3 - len(wiring[i]))
            vals.append(_op(order[i], a[0], a[1], a[2]))
        Dl = z3.Extract(c - 1, 0, D1)
        d = ctz(Dl)
        width = d + c
        lowmask = z3.If(z3.UGE(width, 32), z3.BitVecVal(0xFFFFFFFF, 32), (z3.BitVecVal(1, 32) << width) - 1)
        Hx = z3.ZeroExt(c, hv)
        Yx = z3.ZeroExt(c, yv)
        v_free = (V0 & lowmask) + (Hx << width)
        v_exact = V0 + ((Yx * D1) << c) + ((Yx * Yx * D2) << (2 * c))
        outs = []
        for v in (v_free, v_exact):
            u = z3.Extract(c - 1, 0, z3.LShR(v, kk[pos][slot]))
            vv = vals + [u]
            for i in range(pos + 1, len(order)):
                pool = vv + [kc[i][0], kc[i][1]]
                a = [pool[j] for j in wiring[i]] + [None] * (3 - len(wiring[i]))
                vv.append(_op(order[i], a[0], a[1], a[2]))
            outs.append(vv[-1])
        room = z3.BitVecVal(32 - c, 32) - d
        h_ok = z3.If(z3.UGE(d, 32 - c), Hx == 0, z3.LShR(Hx, room) == 0)
        if target is not None:
            tz = z3.Function('tz', z3.BitVecSort(c), z3.BitVecSort(c))
            for zz in range(1 << c):
                V.add(tz(z3.BitVecVal(zz, c)) == int(target[zz]))
            want = tz(zv)
        else:
            want = tau_z3(zv, Kv, cm)
        V.add(z3.Or(vals[vi] != z3.Extract(c - 1, 0, V0),
                    z3.And(Dl != 0, h_ok, outs[0] != want), z3.And(Dl == 0, outs[1] != want)))
        V.set(timeout=max(1, int(left * 1000)))
        vr = V.check()
        if vr == z3.unsat:
            return dict(result='sat', it=it, consts=[(a.as_long(), b.as_long()) for a, b in kk], K=Kv.as_long())
        if vr == z3.unknown:
            return dict(result='timeout', it=it)
        vm = V.model()
        g = lambda x: vm.eval(x, model_completion=True).as_long()
        constrain(g(zv), g(hv), g(yv))


def pw(Z, i):
    r = z3.BitVecVal(1, 32)
    for _ in range(i):
        r = r * Z
    return r


def solve_split(order, wiring, c=11, seconds=120.0, seed=1):
    parts = []
    for lo in range(0, 33, 4):
        sr = (lo, min(lo + 3, 32))
        r = solve(order, wiring, c, seconds, seed, None, sr)
        parts.append((sr, r['result']))
        if r['result'] == 'sat':
            return dict(result='sat', parts=parts)
    return dict(result='unsat' if all(p[1] == 'unsat' for p in parts) else 'timeout', parts=parts)


if __name__ == '__main__':
    for o, w in [(('mad', 'mad', '>>', 'mad'), ((0, 0, 1), (0, 1, 2), (2, 3), (3, 4, 1))),
                 (('mad', 'mad', '>>', '^'), ((0, 1, 2), (0, 1, 2), (2, 3), (3, 4)))]:
        t = time.time()
        r = solve(o, w, 11, 600)
        print(o, w, r['result'], r.get('it'), '%.1fs' % (time.time() - t), flush=True)
