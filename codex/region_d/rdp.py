"""Fiber model for single constant-shift wirings whose shifted value v is a polynomial of
degree <= 2 in z (MAD-only prefix: v = a2*z^2 + a1*z + a0 mod 2^32, the a_i being
polynomials in the private constants, computed once per query).

Exact fiber relation (c <= 16): for z = z0 + 2^c*y,
    v(z) = V0 + 2^c*y*D + 2^(2c)*a2*y^2   (mod 2^32),   V0 = v(z0), D = 2*a2*z0 + a1.
Let d = v2(D mod 2^c) (per class). If d < c then y -> y*D + 2^c*a2*y^2 is 2^d times a
bijection of Z/2^(32-c-d) (odd linear part, even quadratic part), so on the fiber
    v = (V0 mod 2^(c+d)) + 2^(c+d)*h   for EVERY h in [0, 2^(32-c-d)).
If D == 0 mod 2^c the explicit formula is used with y itself. Samples are (z0, h, y);
the constraint for a class uses the free form when d < c and the explicit one otherwise.
The shift s is symbolic in [0, 32] (32 standing for every s >= 32), or split.
UNSAT excludes the wiring at c output bits for every 32-bit constant choice and shift.
"""
import random
import sys
import time
sys.path.insert(0, '/home/user/aopt')
import z3
from rda import tau_z3


def _op(op, a, b, c):
    if op == 'mad':
        return a * b + c
    if op == '^':
        return a ^ b
    if op == '&':
        return a & b
    return a | b


def poly_degrees(order, wiring, upto):
    """Degree in z of each runtime value if polynomial (MAD-only), else None."""
    deg = [1]
    for i in range(upto):
        pool = deg + [0, 0]
        ds = [pool[j] for j in wiring[i]]
        if order[i] != 'mad' or any(d is None for d in ds):
            deg.append(None if order[i] != 'mad' or any(d is None for d in ds) else None)
            continue
        p = ds[0] + ds[1]
        deg.append(max(p, ds[2]))
    return deg


def analyse(order, wiring):
    if list(order).count('>>') != 1:
        return None
    pos = order.index('>>')
    vi, ki = wiring[pos]
    if vi > pos or ki <= pos:
        return None
    deg = poly_degrees(order, wiring, pos)
    if deg[vi] is None or deg[vi] > 2:
        return None
    # every intermediate on vi's cone must have degree <= 2 (products of degree-1 factors)
    return dict(pos=pos, vi=vi, ki=ki)


def applicable(order, wiring):
    return analyse(order, wiring) is not None


def coeffs(order, wiring, k32, upto):
    """Polynomial triples (a0, a1, a2) of z for each value, as 32-bit z3 terms."""
    one = z3.BitVecVal(1, 32)
    zero = z3.BitVecVal(0, 32)
    P = [(zero, one, zero)]
    for i in range(upto):
        pool = P + [(k32[i][0], zero, zero), (k32[i][1], zero, zero)]
        if order[i] != 'mad':
            P.append(None)
            continue
        a, b, c = [pool[j] for j in wiring[i]]
        if a is None or b is None or c is None:
            P.append(None)
            continue
        # (a0 + a1 z + a2 z^2)(b0 + b1 z + b2 z^2) truncated to degree 2 (valid: degree <= 2)
        p0 = a[0] * b[0]
        p1 = a[0] * b[1] + a[1] * b[0]
        p2 = a[0] * b[2] + a[1] * b[1] + a[2] * b[0]
        P.append((p0 + c[0], p1 + c[1], p2 + c[2]))
    return P


def solve(order, wiring, c=11, seconds=120.0, seed=1, target=None, s_range=(0, 32)):
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
    s = k32[pos][ki - pos - 1]
    S.add(z3.UGE(s, s_range[0]))
    if s_range[1] < 32:
        S.add(z3.ULE(s, s_range[1]))
    co = coeffs(order, wiring, k32, pos)[vi]

    def ctz(x):
        d = z3.BitVecVal(c, 32)
        for b in range(c - 1, -1, -1):
            d = z3.If(z3.Extract(b, b, x) == 1, z3.BitVecVal(b, 32), d)
        return d

    def build(kk, cof, z0, h, y):
        kc = [(z3.Extract(c - 1, 0, a), z3.Extract(c - 1, 0, b)) for a, b in kk]
        vals = [z0]
        for i in range(pos):
            pool = vals + [kc[i][0], kc[i][1]]
            a = [pool[j] for j in wiring[i]] + [None] * (3 - len(wiring[i]))
            vals.append(_op(order[i], a[0], a[1], a[2]))
        a0, a1, a2 = cof
        Z = z3.ZeroExt(32 - c, z0)
        V0 = a2 * Z * Z + a1 * Z + a0
        D = a2 * Z * 2 + a1
        Dl = z3.Extract(c - 1, 0, D)
        d = ctz(Dl)
        width = d + c
        lowmask = z3.If(z3.UGE(width, 32), z3.BitVecVal(0xFFFFFFFF, 32), (z3.BitVecVal(1, 32) << width) - 1)
        Hx = z3.ZeroExt(c, h)
        Yx = z3.ZeroExt(c, y)
        v_free = (V0 & lowmask) + (Hx << width)
        v_exact = V0 + ((Yx * D) << c) + ((Yx * Yx * a2) << (2 * c))
        sh = kk[pos][ki - pos - 1]
        outs = []
        for v in (v_free, v_exact):
            u = z3.Extract(c - 1, 0, z3.LShR(v, sh))
            vv = vals + [u]
            for i in range(pos + 1, len(order)):
                pool = vv + [kc[i][0], kc[i][1]]
                a = [pool[j] for j in wiring[i]] + [None] * (3 - len(wiring[i]))
                vv.append(_op(order[i], a[0], a[1], a[2]))
            outs.append(vv[-1])
        room = z3.BitVecVal(32 - c, 32) - d                 # free bits of h when d < c
        h_ok = z3.If(z3.UGE(d, 32 - c), Hx == 0, z3.LShR(Hx, room) == 0)
        cons = [vals[vi] == z3.Extract(c - 1, 0, V0)]
        return outs, Dl, h_ok, cons

    def want(z0, zv, Kx):
        if target is not None:
            return z3.BitVecVal(int(target[zv]), c)
        return tau_z3(z0, Kx, cm)

    def constrain(zv, hv, yv):
        z0 = z3.BitVecVal(zv, c)
        (of, oe), Dl, h_ok, cons = build(k32, co, z0, z3.BitVecVal(hv, hw), z3.BitVecVal(yv, hw))
        S.add(*cons)
        w = want(z0, zv, K)
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
        cof = [z3.simplify(z3.substitute(x, *[(a, va) for (a, b), (va, vb) in zip(k32, kk)],
                                         *[(b, vb) for (a, b), (va, vb) in zip(k32, kk)])) for x in co]
        V = z3.SolverFor('QF_BV')
        zv = z3.BitVec('zv', c)
        hv = z3.BitVec('hv', hw)
        yv = z3.BitVec('yv', hw)
        (of, oe), Dl, h_ok, cons = build(kk, cof, zv, hv, yv)
        if target is not None:
            tz = z3.Function('tz', z3.BitVecSort(c), z3.BitVecSort(c))
            for zz in range(1 << c):
                V.add(tz(z3.BitVecVal(zz, c)) == int(target[zz]))
            w = tz(zv)
        else:
            w = tau_z3(zv, Kv, cm)
        V.add(z3.Or(z3.Not(z3.And(*cons)), z3.And(Dl != 0, h_ok, of != w), z3.And(Dl == 0, oe != w)))
        V.set(timeout=max(1, int(left * 1000)))
        vr = V.check()
        if vr == z3.unsat:
            return dict(result='sat', it=it, consts=[(a.as_long(), b.as_long()) for a, b in kk], K=Kv.as_long())
        if vr == z3.unknown:
            return dict(result='timeout', it=it)
        vm = V.model()
        g = lambda x: vm.eval(x, model_completion=True).as_long()
        constrain(g(zv), g(hv), g(yv))


if __name__ == '__main__':
    for o, w in [(('mad', 'mad', '>>', '^'), ((0, 1, 2), (0, 1, 2), (2, 3), (3, 4))),
                 (('mad', 'mad', '>>', '^'), ((0, 1, 2), (0, 1, 2), (2, 3), (2, 3)))]:
        t = time.time()
        r = solve(o, w, 11, 600)
        print(o, w, r['result'], r.get('it'), '%.1fs' % (time.time() - t), flush=True)
