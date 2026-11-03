"""Analytic fiber model for v = w*w + w (one MAD(w, w, w) on an affine root w), constant
shift, single SHR.  (w*w + w has odd derivative 2w + 1.)

With w = A*z + B, A = 2^e*A1 (A1 odd; A = 0 is e = 32), on the fiber z = z0 + 2^c*y:
  w = T + 2^(c+e)*A1*y,   T = w(z0),
  v = T^2 + T + 2^(c+e)*A1*y*(2T+1) + 2^(2c+2e)*A1^2*y^2.
y -> A1*y*(2T+1) + 2^(c+e)*A1^2*y^2 is a bijection of Z/2^(32-c-e) (odd linear part),
so on the fiber  v = V_low + 2^(c+e)*h  with h taking EVERY value in [0, 2^(32-c-e)) and
V_low = (T^2 + T) mod 2^(c+e). With T mod 2^(c+e) = B_lo + 2^e*m (B_lo < 2^e,
m = (A1*z0 + B_h) mod 2^min(c, 32-e)):
  V_low = [B_lo^2 + B_lo + 2^e*(m*(2B_lo+1) mod 2^c) + 2^(2e)*(m^2 mod 2^c)] mod 2^min(32, c+e)
(the dropped bits land at or above bit c+e, inside the free part; for e >= 32-c there is
no free part and the formula is exact mod 2^32 because 2^(2e) == 0 there).
All per-sample products are c-bit; B_lo^2 is one shared 32-bit product.
UNSAT excludes the wiring at c output bits for every 32-bit constant choice and every
constant shift s in [0, 32] (s = 32 standing for all s >= 32).

Case split (solve_all):
  A: e <= 31-c and s >= c+e: u = h >> (s-c-e) takes every value in [0, 2^min(c, 32-s)),
     in particular 0 on every class -> a <= 3 T-op program matches mod 2^11 (closure3):
     excluded without search (requires c >= 11).
  B: e = E fixed for each E in [32-c, 32] (no free part), s in [0, 32].
  C: e = E fixed for each E in [0, 31-c], s in [0, c+E-1].
  Every query has all shifts by e as constants.
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
    a, b, cc = wiring[i]
    if not (a == b == cc and a <= i):
        return None
    kinds = affine_kinds(order, wiring, pos)
    if kinds[a] != 'a':
        return None
    return dict(pos=pos, vi=vi, w=a)


def applicable(order, wiring):
    return analyse(order, wiring) is not None


def solve(order, wiring, c=11, seconds=120.0, seed=1, target=None, s_range=None, e_fixed=None, e_range=None):
    an = analyse(order, wiring)
    assert an is not None
    pos, vi, w = an['pos'], an['vi'], an['w']
    cm = (1 << c) - 1
    hw = 32 - c
    S = z3.SolverFor('QF_BV')
    ks = [(z3.BitVec(f'k{i}_0', c), z3.BitVec(f'k{i}_1', c)) for i in range(len(order))]
    P = dict(e=z3.BitVec('e', 8), s=z3.BitVec('s', 8), A1=z3.BitVec('A1', c), Bh=z3.BitVec('Bh', c),
             Blo=z3.BitVec('Blo', 32), K=z3.BitVec('K', c), ks=ks)
    S.add(z3.ULE(P['e'], 32), z3.ULE(P['s'], 32), z3.Extract(0, 0, P['A1']) == 1,
          z3.LShR(P['Blo'], z3.ZeroExt(24, P['e'])) == 0)
    if s_range is not None:
        S.add(z3.UGE(P['s'], s_range[0]), z3.ULE(P['s'], s_range[1]))
    if e_range is not None:
        S.add(z3.UGE(P['e'], e_range[0]), z3.ULE(P['e'], e_range[1]))
    if e_fixed is not None:
        S.add(P['e'] == e_fixed)
    if target is not None:
        S.add(P['K'] == 0)

    def build(P, z0, h):
        vals = [z0]
        for i in range(pos):
            pool = vals + [P['ks'][i][0], P['ks'][i][1]]
            a = [pool[j] for j in wiring[i]] + [None] * (3 - len(wiring[i]))
            vals.append(_op(order[i], a[0], a[1], a[2]))
        e = P['e'] if e_fixed is None else z3.BitVecVal(e_fixed, 8)
        e32 = z3.ZeroExt(24, e) if e_fixed is None else z3.BitVecVal(e_fixed, 32)
        if e_fixed is None:
            keep = z3.If(z3.UGT(e, 32 - c), z3.LShR(z3.BitVecVal(cm, c), z3.Extract(c - 1, 0, e32) - (32 - c)),
                         z3.BitVecVal(cm, c))
        else:
            keep = z3.BitVecVal(cm >> max(0, e_fixed - (32 - c)), c)
        m = (P['A1'] * z0 + P['Bh']) & keep
        Blo = P['Blo']
        cons = [vals[w] == z3.Extract(c - 1, 0, Blo + (z3.ZeroExt(32 - c, m) << e32))]
        blc = z3.Extract(c - 1, 0, Blo)
        p1 = m * (blc + blc + 1)
        p2 = m * m
        V = Blo * Blo + Blo + (z3.ZeroExt(32 - c, p1) << e32) + (z3.ZeroExt(32 - c, p2) << (e32 + e32))
        width = e32 + c
        if e_fixed is None:
            lowmask = z3.If(z3.UGE(e, 32 - c), z3.BitVecVal(0xFFFFFFFF, 32),
                            (z3.BitVecVal(1, 32) << width) - 1)
        else:
            lowmask = z3.BitVecVal(0xFFFFFFFF if e_fixed >= 32 - c else (1 << (e_fixed + c)) - 1, 32)
        v32 = (V & lowmask) + (z3.ZeroExt(c, h) << width)
        u = z3.Extract(c - 1, 0, z3.LShR(v32, z3.ZeroExt(24, P['s'])))
        vals = vals + [u]
        for i in range(pos + 1, len(order)):
            pool = vals + [P['ks'][i][0], P['ks'][i][1]]
            a = [pool[j] for j in wiring[i]] + [None] * (3 - len(wiring[i]))
            vals.append(_op(order[i], a[0], a[1], a[2]))
        return vals[-1], cons

    def h_ok(P, h):
        if e_fixed is not None:
            room = 32 - c - e_fixed
            return h == 0 if room <= 0 else z3.LShR(h, room) == 0
        room = z3.BitVecVal(32 - c, 8) - P['e']
        return z3.If(z3.UGE(P['e'], 32 - c), h == 0, z3.LShR(h, z3.ZeroExt(hw - 8, room)) == 0)

    def want(P, z0, zv):
        if target is not None:
            return z3.BitVecVal(int(target[zv]), c)
        return tau_z3(z0, P['K'], cm)

    def constrain(zv, hv):
        z0 = z3.BitVecVal(zv, c)
        h = z3.BitVecVal(hv, hw)
        out, cons = build(P, z0, h)
        S.add(*cons)
        S.add(z3.Implies(h_ok(P, h), out == want(P, z0, zv)))

    rng = random.Random(seed)
    hm = (1 << hw) - 1
    pts = [(0, 0), (1, 0), (2, 0), (3, 0), (cm, 0), (1, 1), (5, 2), (7, hm), (0, hm), (1 << (c - 1), 1 << (hw - 1))]
    pts += [(rng.getrandbits(c), rng.getrandbits(hw)) for _ in range(10)]
    pts += [(rng.getrandbits(c), 0) for _ in range(10)]
    for p in pts:
        constrain(*p)
    t0 = time.monotonic()
    it = 0
    while True:
        left = seconds - (time.monotonic() - t0)
        if left <= 0:
            return dict(result='timeout', c=c, it=it)
        S.set(timeout=max(1, int(left * 1000)))
        st = S.check()
        it += 1
        if st == z3.unsat:
            return dict(result='unsat', c=c, it=it, sec=round(time.monotonic() - t0, 3))
        if st == z3.unknown:
            return dict(result='timeout', c=c, it=it)
        mo = S.model()
        ev = lambda x: mo.eval(x, model_completion=True)
        Pc = dict(e=ev(P['e']), s=ev(P['s']), A1=ev(P['A1']), Bh=ev(P['Bh']), Blo=ev(P['Blo']), K=ev(P['K']),
                  ks=[(ev(a), ev(b)) for a, b in ks])
        V = z3.SolverFor('QF_BV')
        zv = z3.BitVec('zv', c)
        hv = z3.BitVec('hv', hw)
        out, cons = build(Pc, zv, hv)
        if target is not None:
            tz = z3.Function('tz', z3.BitVecSort(c), z3.BitVecSort(c))
            for zz in range(1 << c):
                V.add(tz(z3.BitVecVal(zz, c)) == int(target[zz]))
            wv = tz(zv)
        else:
            wv = tau_z3(zv, Pc['K'], cm)
        V.add(z3.Or(z3.Not(z3.And(*cons)), z3.And(h_ok(Pc, hv), out != wv)))
        V.set(timeout=max(1, int(left * 1000)))
        vr = V.check()
        if vr == z3.unsat:
            return dict(result='sat', c=c, it=it, e=Pc['e'].as_long(), s=Pc['s'].as_long(), K=Pc['K'].as_long(),
                        A1=Pc['A1'].as_long(), Bh=Pc['Bh'].as_long(), Blo=Pc['Blo'].as_long(),
                        consts=[(a.as_long(), b.as_long()) for a, b in Pc['ks']])
        if vr == z3.unknown:
            return dict(result='timeout', c=c, it=it)
        vm = V.model()
        constrain(vm.eval(zv, model_completion=True).as_long(), vm.eval(hv, model_completion=True).as_long())


def solve_all(order, wiring, c=11, seconds=120.0, seed=1):
    """Every e in [0, 32] as its own query (all shifts by e constant); for e <= 31-c only
    s < c+e is searched (s >= c+e is case A, excluded by the closure argument).
    'unsat' iff every query is UNSAT."""
    assert c >= 11
    parts = []
    for E in range(0, 33):
        sr = (0, c + E - 1) if E <= 31 - c else (0, 32)
        r = solve(order, wiring, c, seconds, seed, e_fixed=E, s_range=sr)
        parts.append((E, r['result'], r.get('it')))
        if r['result'] == 'sat':
            return dict(result='sat', parts=parts, sat=r)
    res = 'unsat' if all(p[1] == 'unsat' for p in parts) else 'timeout'
    return dict(result=res, parts=parts)


if __name__ == '__main__':
    for o, w in [(('mad', 'mad', '>>', '^'), ((0, 1, 2), (1, 1, 1), (2, 3), (3, 4))),
                 (('mad', 'mad', '>>', '^'), ((0, 1, 2), (1, 1, 1), (2, 3), (2, 3))),
                 (('mad', 'mad', '>>', 'mad'), ((0, 1, 2), (1, 1, 1), (2, 3), (3, 4, 1)))]:
        t = time.time()
        r = solve_all(o, w, 11, 300)
        print(o, w, r['result'], r['parts'], '%.1fs' % (time.time() - t), flush=True)
