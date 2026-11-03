"""Exact fiber model for single-SHR wirings u = a >> b where a and b are each either the
affine root w (w = A*z + B) or a square-type value MAD(w, w, x) = w*w + x with x a
private constant or w itself, at least one of them being square-type (all other
instructions T-ops). Covers e.g. u = t0 >> (t0*t0 + k).

Fiber z = z0 + 2^c*y, A = 2^e*A1 (A1 odd, A = 0 is e = 32):
    w = T + 2^(c+e)*H,  T = B_lo + 2^e*m (mod 2^32), m = (A1*z0 + B_h) mod 2^min(c, 32-e),
with H taking every value in [0, 2^(32-c-e)) (H = 0 only when e >= 32-c). Then exactly
    w^2 = B_lo^2 + 2^(e+1)*B_lo*m + 2^(2e)*m^2 + 2^(c+e+1)*T*H + 2^(2c+2e)*H^2  (mod 2^32),
so every per-sample product is (e-bit x c-bit), (c x c) or (32-bit x constant H); B_lo^2
is shared. Both shift operands are evaluated exactly at 32 bits, everything else at c
bits, for all (z0, H). Split: e >= 32-c symbolic in one query (H = 0); each e = E in
[0, 31-c] as its own query with every shift by e a constant.
UNSAT of all parts excludes the wiring at c output bits for every 32-bit constant choice.
"""
import random
import sys
import time
sys.path.insert(0, '/home/user/aopt')
import z3
from rda import affine_kinds, tau_z3


def _op(op, a, b, c):
    if op == 'mad':
        return a * b + c
    if op == '^':
        return a ^ b
    if op == '&':
        return a & b
    return a | b


def analyse(order, wiring):
    if list(order).count('>>') != 1:
        return None
    pos = order.index('>>')
    ai, bi = wiring[pos]
    if ai > pos or bi > pos:
        return None
    kinds = affine_kinds(order, wiring, pos)

    def classify(x):
        if kinds[x] == 'a':
            return ('w', x)
        i = x - 1
        if x == 0 or order[i] != 'mad':
            return None
        p, q, r = wiring[i]
        if p != q or p > i or kinds[p] != 'a':
            return None
        if r == p:
            return ('sq_w', p)
        if r > i:
            return ('sq_k', p, i, r - i - 1)
        return None
    ca, cb = classify(ai), classify(bi)
    if ca is None or cb is None:
        return None
    if ca[0] == 'w' and cb[0] == 'w':
        return None                                   # affine/affine: rdvf
    roots = {ca[1], cb[1]}
    if len(roots) != 1:
        return None
    return dict(pos=pos, a=ca, b=cb, w=roots.pop())


def applicable(order, wiring):
    return analyse(order, wiring) is not None


def _solve(order, wiring, an, c, seconds, seed, E, target=None):
    """E = None: e symbolic in [32-c, 32] (H = 0); else e fixed to E <= 31-c."""
    pos, w = an['pos'], an['w']
    cm = (1 << c) - 1
    hw = 32 - c
    S = z3.SolverFor('QF_BV')
    k32 = [(z3.BitVec(f'k{i}_0', 32), z3.BitVec(f'k{i}_1', 32)) for i in range(len(order))]
    K = z3.BitVec('K', c)
    if target is not None:
        S.add(K == 0)
    A1 = z3.BitVec('A1', c)
    Bh = z3.BitVec('Bh', c)
    S.add(z3.Extract(0, 0, A1) == 1)
    if E is None:
        e8 = z3.BitVec('e', 8)
        S.add(z3.UGE(e8, 32 - c), z3.ULE(e8, 32))
        e32 = z3.ZeroExt(24, e8)
        Blo = z3.BitVec('Blo', 32)
        S.add(z3.LShR(Blo, e32) == 0)
        keep = z3.LShR(z3.BitVecVal(cm, c), z3.Extract(c - 1, 0, e32) - (32 - c))
    else:
        e32 = z3.BitVecVal(E, 32)
        Blo = z3.ZeroExt(32 - E, z3.BitVec('Blo', E)) if E > 0 else z3.BitVecVal(0, 32)
        keep = z3.BitVecVal(cm, c)

    def build(z0, H, kk):
        kc = [(z3.Extract(c - 1, 0, a), z3.Extract(c - 1, 0, b)) for a, b in kk]
        vals = [z0]
        for i in range(pos):
            pool = vals + [kc[i][0], kc[i][1]]
            a = [pool[j] for j in wiring[i]] + [None] * (3 - len(wiring[i]))
            vals.append(_op(order[i], a[0], a[1], a[2]))
        m = z3.ZeroExt(32 - c, (A1 * z0 + Bh) & keep)
        T = Blo + (m << e32)
        cons = [vals[w] == z3.Extract(c - 1, 0, T)]
        if E is None:
            w32 = T
            sq = Blo * Blo + ((z3.ZeroExt(32 - c, z3.Extract(c - 1, 0, Blo)) * m) << (e32 + 1))
        else:
            Hx = z3.ZeroExt(c, H)
            w32 = T + (Hx << (E + c))
            sq = (Blo * Blo + ((Blo * m) << (E + 1)) + ((m * m) << (2 * E))
                  + ((T * Hx) << (c + E + 1)) + ((Hx * Hx) << (2 * c + 2 * E)))

        def val(cl):
            if cl[0] == 'w':
                return w32
            if cl[0] == 'sq_w':
                return sq + w32
            return sq + kk[cl[2]][cl[3]]
        u = z3.Extract(c - 1, 0, z3.LShR(val(an['a']), val(an['b'])))
        vals = vals + [u]
        for i in range(pos + 1, len(order)):
            pool = vals + [kc[i][0], kc[i][1]]
            a = [pool[j] for j in wiring[i]] + [None] * (3 - len(wiring[i]))
            vals.append(_op(order[i], a[0], a[1], a[2]))
        return vals[-1], cons

    def h_ok(H):
        if E is None:
            return H == 0
        room = 32 - c - E
        return z3.LShR(H, room) == 0

    def constrain(zv, hv):
        z0 = z3.BitVecVal(zv, c)
        H = z3.BitVecVal(hv, hw)
        out, cons = build(z0, H, k32)
        S.add(*cons)
        want = z3.BitVecVal(int(target[zv]), c) if target is not None else tau_z3(z0, K, cm)
        S.add(z3.Implies(h_ok(H), out == want))

    rng = random.Random(seed)
    hmax = (1 << (32 - c - E)) - 1 if E is not None else 0
    pts = [(0, 0), (1, 0), (2, 0), (3, 0), (cm, 0), (5, 0), (7, 0)]
    if hmax:
        pts += [(1, 1), (5, 2 & hmax), (7, hmax), (0, hmax), (1 << (c - 1), hmax >> 1)]
        pts += [(rng.getrandbits(c), rng.randrange(hmax + 1)) for _ in range(12)]
    pts += [(rng.getrandbits(c), 0) for _ in range(8)]
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
        ev = lambda x: mo.eval(x, model_completion=True)
        kk = [(ev(a), ev(b)) for a, b in k32]
        V = z3.SolverFor('QF_BV')
        zv = z3.BitVec('zv', c)
        Hv = z3.BitVec('Hv', hw)
        subst = [(A1, ev(A1)), (Bh, ev(Bh)), (K, ev(K))]
        if E is None:
            subst += [(e8, ev(e8)), (Blo, ev(Blo))]
        elif E > 0:
            bl = z3.BitVec('Blo', E)
            subst += [(bl, ev(bl))]
        out, cons = build(zv, Hv, kk)
        out = z3.substitute(out, *subst)
        cons = [z3.substitute(x, *subst) for x in cons]
        if target is not None:
            tz = z3.Function('tz', z3.BitVecSort(c), z3.BitVecSort(c))
            for zz in range(1 << c):
                V.add(tz(z3.BitVecVal(zz, c)) == int(target[zz]))
            want = tz(zv)
        else:
            want = tau_z3(zv, ev(K), cm)
        V.add(z3.Or(z3.Not(z3.And(*cons)), z3.And(h_ok(Hv), out != want)))
        V.set(timeout=max(1, int(left * 1000)))
        vr = V.check()
        if vr == z3.unsat:
            return dict(result='sat', it=it, E=E)
        if vr == z3.unknown:
            return dict(result='timeout', it=it)
        vm = V.model()
        constrain(vm.eval(zv, model_completion=True).as_long(), vm.eval(Hv, model_completion=True).as_long())


def solve_all(order, wiring, c=11, seconds=120.0, seed=1, target=None):
    an = analyse(order, wiring)
    assert an is not None
    parts = []
    for E in [None] + list(range(0, 32 - c)):
        r = _solve(order, wiring, an, c, seconds, seed, E, target)
        parts.append(('B' if E is None else E, r['result'], r.get('it')))
        if r['result'] == 'sat':
            return dict(result='sat', parts=parts)
    res = 'unsat' if all(p[1] == 'unsat' for p in parts) else 'timeout'
    return dict(result=res, parts=parts)


if __name__ == '__main__':
    o, w = ('mad', 'mad', '>>', 'mad'), ((0, 1, 2), (1, 1, 2), (1, 2), (0, 3, 4))
    print(analyse(o, w))
    t = time.time()
    r = solve_all(o, w, 11, 300)
    print(r['result'], r['parts'], '%.1fs' % (time.time() - t))
