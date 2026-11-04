"""Exact fiber model for single CONSTANT-shift wirings whose shifted value is
v = MAD(w, w, x) = w*w + x, w an affine runtime value, x a private constant or w itself.
(Covers the even-derivative case w*w + k that rdq.py, which needs an odd derivative,
does not; rdp2's per-class valuation abstraction is slow on it.)

Fiber z = z0 + 2^c*y, A = 2^e*A1 (A1 odd; A = 0 is e = 32): w = T + 2^(c+e)*H with
T = B_lo + 2^e*m(z0) and H taking every value in [0, 2^(32-c-e)) (H = 0 if e >= 32-c).
Exactly (mod 2^32):
    w^2 = B_lo^2 + 2^(e+1)*B_lo*m + 2^(2e)*m^2 + 2^(c+e+1)*T*H + 2^(2c+2e)*H^2,
so v and u = v >> s are evaluated exactly for every (z0, H) with only (e x c)-bit,
(c x c)-bit and (32-bit x constant H) products per sample. Exact at every c (no 3c >= 32
condition), so it can also replay known c=8 solutions. Split: e = E fixed for each
E in [0, 31-c] (all shifts by e constant) plus one query for e in [32-c, 32] (H = 0).
UNSAT of every part excludes the wiring at c output bits for every 32-bit constant
choice and every shift s in [0, 32] (32 standing for all s >= 32).
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
    vi, ki = wiring[pos]
    if vi > pos or ki <= pos or vi == 0:
        return None
    i = vi - 1
    if order[i] != 'mad':
        return None
    p, q, r = wiring[i]
    if p != q or p > i:
        return None
    kinds = affine_kinds(order, wiring, pos)
    if kinds[p] != 'a':
        return None
    if r == p:
        x = ('w',)
    elif r > i:
        x = ('k', i, r - i - 1)
    else:
        return None
    return dict(pos=pos, vi=vi, ki=ki, w=p, x=x)


def applicable(order, wiring):
    return analyse(order, wiring) is not None


def _solve(order, wiring, an, c, seconds, seed, E, target=None, s_range=(0, 32)):
    pos, w = an['pos'], an['w']
    cm = (1 << c) - 1
    hw = 32 - c
    S = z3.SolverFor('QF_BV')
    k32 = [(z3.BitVec(f'k{i}_0', 32), z3.BitVec(f'k{i}_1', 32)) for i in range(len(order))]
    K = z3.BitVec('K', c)
    if target is not None:
        S.add(K == 0)
    slot = an['ki'] - pos - 1
    sh = k32[pos][slot]
    S.add(z3.UGE(sh, s_range[0]))
    if s_range[1] < 32:
        S.add(z3.ULE(sh, s_range[1]))
    A1 = z3.BitVec('A1', c)
    Bh = z3.BitVec('Bh', c)
    S.add(z3.Extract(0, 0, A1) == 1)
    if E is None:
        e8 = z3.BitVec('e', 8)
        S.add(z3.UGE(e8, 32 - c), z3.ULE(e8, 32))
        e32 = z3.ZeroExt(24, e8)
        Blo = z3.BitVec('Blo', 32)
        S.add(z3.LShR(Blo, e32) == 0)
        keep = z3.LShR(z3.BitVecVal(cm, c), z3.Extract(c - 1, 0, e32) - (32 - c)) if c < 32 else None
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
        x32 = w32 if an['x'][0] == 'w' else kk[an['x'][1]][an['x'][2]]
        v32 = sq + x32
        u = z3.Extract(c - 1, 0, z3.LShR(v32, kk[pos][slot]))
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
        return H == 0 if room <= 0 else z3.LShR(H, room) == 0

    def constrain(zv, hv):
        z0 = z3.BitVecVal(zv, c)
        H = z3.BitVecVal(hv, hw)
        out, cons = build(z0, H, k32)
        S.add(*cons)
        want = z3.BitVecVal(int(target[zv]), c) if target is not None else tau_z3(z0, K, cm)
        S.add(z3.Implies(h_ok(H), out == want))

    rng = random.Random(seed)
    hmax = (1 << (32 - c - E)) - 1 if (E is not None and 32 - c - E > 0) else 0
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
        subst = [(A1, ev(A1)), (Bh, ev(Bh)), (K, ev(K))]
        if E is None:
            subst += [(e8, ev(e8)), (Blo, ev(Blo))]
        elif E > 0:
            bl = z3.BitVec('Blo', E)
            subst += [(bl, ev(bl))]
        V = z3.SolverFor('QF_BV')
        zv = z3.BitVec('zv', c)
        Hv = z3.BitVec('Hv', hw)
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
    for E in [None] + list(range(0, max(0, 32 - c))):
        r = _solve(order, wiring, an, c, seconds, seed, E, target)
        parts.append(('B' if E is None else E, r['result'], r.get('it')))
        if r['result'] == 'sat':
            return dict(result='sat', parts=parts)
    res = 'unsat' if all(p[1] == 'unsat' for p in parts) else 'timeout'
    return dict(result=res, parts=parts)


if __name__ == '__main__':
    o, w = ('mad', 'mad', '>>', 'mad'), ((0, 1, 2), (1, 1, 2), (2, 3), (0, 3, 2))
    print(analyse(o, w))
    t = time.time()
    r = solve_all(o, w, 11, 120)
    print(r['result'], r['parts'], '%.1fs' % (time.time() - t), flush=True)
