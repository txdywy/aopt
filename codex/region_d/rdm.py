"""Bit-sliced fiber model for programs with ONE OR MORE SHRs whose operands all derive
from a single affine runtime value w (w = A*z + B) through SHR, XOR/AND/OR with private
constants, or are private constants. Every other instruction is a T-op.

Fiber z = z0 + 2^c*y, A = 2^e*A1 (A1 odd; A = 0 is e = 32): w = T + 2^(c+e)*h with
T = B_lo + 2^e*m(z0), m = (A1*z0 + B_h) mod 2^min(c, 32-e), h taking every value in
[0, 2^(32-c-e)) (fiber lemma, rda.py). All values on the w-cone are functions of w and
private constants only, so they are evaluated EXACTLY at 32 bits from w32 for every
(z0, h); T-op values outside the cone are fixed mod 2^c on the fiber and are evaluated
at c bits. No multiplication touches h, so samples are cheap. Shift literals are
32-bit symbols (any value; >= 32 gives 0). e is fixed per query (E in [0, 31-c]) plus
one query for e in [32-c, 32] (no free part).
UNSAT of every part excludes the wiring at c output bits for every 32-bit constant
choice and every shift literal.
"""
import random
import sys
import time
sys.path.insert(0, '/home/user/aopt')
import z3
from rda import affine_kinds, tau_z3


def _op(op, a, b, c=None):
    if op == 'mad':
        return a * b + c
    if op == '^':
        return a ^ b
    if op == '&':
        return a & b
    if op == '|':
        return a | b
    return z3.LShR(a, b)


def analyse(order, wiring):
    n = len(order)
    if '>>' not in order[:n - 1]:
        return None
    upto = n - 1
    kinds = ['a']                       # z is affine
    for i in range(n):
        op, idx = order[i], wiring[i]
        pool = kinds + ['c', 'c']
        ks = [pool[j] for j in idx]
        if op == 'mad':
            a, b, c = ks
            if all(k == 'c' for k in ks):
                k = 'c'
            elif 'o' in ks or (a != 'c' and b != 'c'):
                k = 'o'
            else:
                k = 'a'
        else:
            k = 'c' if all(x == 'c' for x in ks) else 'o'
        kinds.append(k)
    # w-cone: the closure of SHR operands under SHR / bitwise ops
    cone, roots = set(), set()

    def visit(x):
        if x == 0 or kinds[x] == 'a':
            roots.add(x)
            return True
        if kinds[x] == 'c':
            return True
        i = x - 1
        if order[i] not in ('^', '&', '|', '>>'):
            return False
        cone.add(i)
        return all(visit(j) if j <= i else True for j in wiring[i])
    for i, op in enumerate(order):
        if op == '>>':
            cone.add(i)
            for j in wiring[i]:
                if j <= i and not visit(j):
                    return None
    if len(roots) != 1:
        return None
    w = next(iter(roots))
    # every value that a T-op outside the cone reads from the cone is a cone value; fine.
    return dict(w=w, cone=sorted(cone))


def applicable(order, wiring):
    return analyse(order, wiring) is not None


def _solve(order, wiring, an, c, seconds, seed, E, target=None):
    n = len(order)
    w, cone = an['w'], set(an['cone'])
    cm = (1 << c) - 1
    hw = 32 - c
    S = z3.SolverFor('QF_BV')
    k32 = [(z3.BitVec(f'k{i}_0', 32), z3.BitVec(f'k{i}_1', 32)) for i in range(n)]
    K = z3.BitVec('K', c)
    if target is not None:
        S.add(K == 0)
    A1 = z3.BitVec('A1', c)
    Bh = z3.BitVec('Bh', c)
    S.add(z3.Extract(0, 0, A1) == 1)
    if w == 0:
        S.add(A1 == 1, Bh == 0)        # the root is z itself
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
        m = z3.ZeroExt(32 - c, (A1 * z0 + Bh) & keep)
        T = Blo + (m << e32)
        if E is None:
            w32 = T
        else:
            w32 = T + (z3.ZeroExt(c, H) << (E + c))
        v32 = {}                      # 32-bit values on the cone (and the root)
        vc = [z0]                     # c-bit values
        cons = []
        for i in range(n):
            op, idx = order[i], wiring[i]
            if i in cone:
                args = []
                for j in idx:
                    if j <= i:
                        if j == w:
                            args.append(w32)
                        else:
                            args.append(v32[j])
                    else:
                        args.append(kk[i][j - i - 1])
                v = _op(op, *args) if op != 'mad' else None
                v32[i + 1] = v
                vc.append(z3.Extract(c - 1, 0, v))
            else:
                pool = vc + [kc[i][0], kc[i][1]]
                a = [pool[j] for j in idx] + [None] * (3 - len(idx))
                vc.append(_op(op, a[0], a[1], a[2]) if op != '>>' else None)
            if i + 1 == w:
                cons.append(vc[w] == z3.Extract(c - 1, 0, T))
        if w == 0:
            pass
        return vc[-1], cons

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
        V.add(z3.Or(z3.Not(z3.And(*cons)) if cons else z3.BoolVal(False), z3.And(h_ok(Hv), out != want)))
        V.set(timeout=max(1, int(left * 1000)))
        vr = V.check()
        if vr == z3.unsat:
            return dict(result='sat', it=it, E=E)
        if vr == z3.unknown:
            return dict(result='timeout', it=it)
        vm = V.model()
        constrain(vm.eval(zv, model_completion=True).as_long(), vm.eval(Hv, model_completion=True).as_long())


def solve_all(order, wiring, c=11, seconds=120.0, seed=1, target=None, stop_on_fail=True):
    an = analyse(order, wiring)
    assert an is not None
    parts = []
    Es = [None] + list(range(0, max(0, 32 - c)))
    if an['w'] == 0:
        Es = [0]                       # the root is z itself: e = 0, A1 = 1 (covered by E=0)
    for E in Es:
        r = _solve(order, wiring, an, c, seconds, seed, E, target)
        parts.append(('B' if E is None else E, r['result'], r.get('it')))
        if r['result'] == 'sat':
            return dict(result='sat', parts=parts)
        if r['result'] != 'unsat' and stop_on_fail:
            break
    res = 'unsat' if all(p[1] == 'unsat' for p in parts) and len(parts) == len(Es) else 'timeout'
    return dict(result=res, parts=parts)


if __name__ == '__main__':
    for o, w in [(('mad', '>>', '>>', 'mad'), ((0, 1, 2), (1, 2), (1, 3), (0, 2, 3))),
                 (('mad', '>>', '>>', 'mad'), ((0, 1, 2), (1, 2), (1, 3), (0, 3, 2)))]:
        print(o, w, analyse(o, w), flush=True)
        t = time.time()
        r = solve_all(o, w, 11, 120)
        print('  ', r['result'], r['parts'][-3:], '%.1fs' % (time.time() - t), flush=True)
