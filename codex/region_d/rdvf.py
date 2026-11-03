"""Fiber reduction for single-SHR wirings u = a >> b with a RUNTIME AFFINE shift amount
b = A_b*z + B_b (mod 2^32). The value a is either a private constant ('k' type) or
G(w) for one affine runtime value w and a bitwise composition G (as in rda2.py).

Fiber z = z0 + 2^c*y (c >= 11). Write A_b = 2^e_b * odd (or A_b = 0).
 (i) e_b <= 31-c (A_b != 0): b = b(z0) + 2^(c+e_b)*h_b with h_b taking every value in
     [0, 2^(32-c-e_b)) on the fiber; h_b = 1 gives b >= 2^c >= 32, so u = 0 at some z of
     EVERY class. There the output equals H(z0, 0), computed by the other three T-ops with
     u replaced by the constant 0, so a <= 3 T-op program would match F(z^K)^D mod 2^11 for
     all z0: impossible by the 3-instruction closure (closure3.py). Excluded, no search.
 (ii) e_b >= 32-c (includes A_b = 0): b is constant on every fiber,
     b(z0) = B_b + ((A' * z0) mod 2^c) << (32-c), A' = A_b >> (32-c) (any c-bit value).
     The value a: a constant K32; or G(w) with w = b itself (then a is constant on the
     fiber too); or G(w) with w != b, whose high part is free on the fiber exactly as in
     rda: w = B_lo + 2^e*m(z0) + 2^(c+e)*h, h in [0, 2^(32-c-e)), independent of b because
     b does not move on the fiber. a and the shift are evaluated exactly at 32 bits, the
     rest at c bits. Searched here with CEGIS over (z0, h); verification by z3.
UNSAT of (ii) excludes the wiring at c >= 11 output bits for every 32-bit constant choice.
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
    """Return dict(pos, bi, vi, kind_value, w, path) or None."""
    if list(order).count('>>') != 1:
        return None
    pos = order.index('>>')
    vi, bi = wiring[pos]
    if bi > pos:
        return None                                      # constant amount: rda / rda2
    kinds = affine_kinds(order, wiring, pos)
    if kinds[bi] != 'a':
        return None
    if vi > pos:
        return dict(pos=pos, bi=bi, vi=vi, value='const', w=None, path=[])
    # value: bitwise composition of a single affine root
    if kinds[vi] == 'a':
        return dict(pos=pos, bi=bi, vi=vi, value='same' if vi == bi else 'free', w=vi, path=[])
    if kinds[vi] != 'o':
        return None
    path, roots = set(), set()

    def visit(x):
        if kinds[x] == 'a':
            roots.add(x)
            return True
        if kinds[x] == 'c':
            return True
        i = x - 1
        if order[i] not in ('^', '&', '|'):
            return False
        path.add(i)
        return all(visit(j) if j <= i else True for j in wiring[i])
    if not visit(vi) or len(roots) != 1:
        return None
    w = next(iter(roots))
    return dict(pos=pos, bi=bi, vi=vi, value='same' if w == bi else 'free', w=w, path=sorted(path))


def applicable(order, wiring):
    return analyse(order, wiring) is not None


def solve(order, wiring, c=11, seconds=120.0, seed=1, target=None):
    assert c >= 11 or target is not None, 'branch (i) relies on closure3 at 11 bits'
    an = analyse(order, wiring)
    assert an is not None
    pos, bi, vi, w, path = an['pos'], an['bi'], an['vi'], an['w'], an['path']
    cm = (1 << c) - 1
    hw = 32 - c
    S = z3.SolverFor('QF_BV')
    k32 = [(z3.BitVec(f'k{i}_0', 32), z3.BitVec(f'k{i}_1', 32)) for i in range(len(order))]
    kc = [(z3.Extract(c - 1, 0, a), z3.Extract(c - 1, 0, b)) for a, b in k32]
    K = z3.BitVec('K', c)
    if target is not None:
        S.add(K == 0)
    Bb = z3.BitVec('Bb', 32)
    Ap = z3.BitVec('Ap', c)
    e = z3.BitVec('e', 8)
    A1 = z3.BitVec('A1', c)
    Bh = z3.BitVec('Bh', c)
    Blo = z3.BitVec('Blo', 32)
    S.add(z3.ULE(e, 32), z3.Extract(0, 0, A1) == 1, z3.LShR(Blo, z3.ZeroExt(24, e)) == 0)
    P = dict(k32=k32, kc=kc, K=K, Bb=Bb, Ap=Ap, e=e, A1=A1, Bh=Bh, Blo=Blo)

    def build(P, z0, h):
        vals = [z0]
        for i in range(pos):
            pool = vals + [P['kc'][i][0], P['kc'][i][1]]
            a = [pool[j] for j in wiring[i]] + [None] * (3 - len(wiring[i]))
            vals.append(_op(order[i], a[0], a[1], a[2]))
        b32 = P['Bb'] + (z3.ZeroExt(32 - c, P['Ap'] * z0) << (32 - c))
        cons = [vals[bi] == z3.Extract(c - 1, 0, b32)]
        if an['value'] == 'const':
            a32 = P['k32'][pos][vi - pos - 1]
        else:
            if an['value'] == 'same':
                w32 = b32
            else:
                e32 = z3.ZeroExt(24, P['e'])
                keep = z3.If(z3.UGT(P['e'], 32 - c),
                             z3.LShR(z3.BitVecVal(cm, c), z3.Extract(c - 1, 0, e32) - (32 - c)),
                             z3.BitVecVal(cm, c))
                m = (P['A1'] * z0 + P['Bh']) & keep
                w32 = P['Blo'] + (z3.ZeroExt(32 - c, m) << e32) + (z3.ZeroExt(c, h) << (e32 + c))
                cons.append(vals[w] == z3.Extract(c - 1, 0, w32))
            sv = {w: w32}
            for i in path:
                args = [sv[j] if j <= i else P['k32'][i][j - i - 1] for j in wiring[i]]
                sv[i + 1] = _op(order[i], args[0], args[1], None)
            a32 = sv[vi]
        u = z3.Extract(c - 1, 0, z3.LShR(a32, b32))
        vals = vals + [u]
        for i in range(pos + 1, len(order)):
            pool = vals + [P['kc'][i][0], P['kc'][i][1]]
            a = [pool[j] for j in wiring[i]] + [None] * (3 - len(wiring[i]))
            vals.append(_op(order[i], a[0], a[1], a[2]))
        return vals[-1], cons

    def h_ok(P, h):
        if an['value'] != 'free':
            return h == 0
        room = z3.BitVecVal(32 - c, 8) - P['e']
        return z3.If(z3.UGE(P['e'], 32 - c), h == 0, z3.LShR(h, z3.ZeroExt(hw - 8, room)) == 0)

    def want(P, z0, zv=None):
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
        Pc = dict(k32=[(ev(a), ev(b)) for a, b in k32], K=ev(K), Bb=ev(Bb), Ap=ev(Ap), e=ev(e), A1=ev(A1),
                  Bh=ev(Bh), Blo=ev(Blo))
        Pc['kc'] = [(z3.Extract(c - 1, 0, a), z3.Extract(c - 1, 0, b)) for a, b in Pc['k32']]
        V = z3.SolverFor('QF_BV')
        zv = z3.BitVec('zv', c)
        hv = z3.BitVec('hv', hw)
        out, cons = build(Pc, zv, hv)
        if target is not None:
            import numpy as np
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
            return dict(result='sat', c=c, it=it, K=Pc['K'].as_long(),
                        consts=[(a.as_long(), b.as_long()) for a, b in Pc['k32']],
                        Bb=Pc['Bb'].as_long(), Ap=Pc['Ap'].as_long(), e=Pc['e'].as_long())
        if vr == z3.unknown:
            return dict(result='timeout', c=c, it=it)
        vm = V.model()
        constrain(vm.eval(zv, model_completion=True).as_long(), vm.eval(hv, model_completion=True).as_long())


if __name__ == '__main__':
    for o, w in [(('mad', 'mad', '>>', 'mad'), ((0, 1, 2), (0, 2, 3), (1, 2), (0, 3, 2))),
                 (('mad', 'mad', '>>', 'mad'), ((0, 1, 2), (0, 2, 3), (2, 1), (3, 4, 2)))]:
        print(o, w, analyse(o, w), flush=True)
        t = time.time()
        r = solve(o, w, 11, 300)
        print('  ', r['result'], r.get('it'), '%.1fs' % (time.time() - t), flush=True)
