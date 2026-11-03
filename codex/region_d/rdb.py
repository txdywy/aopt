"""Bit-sliced fiber model for single-SHR wirings whose shifted value is v = g(w), where
w is ONE affine runtime value (w = A*z + B mod 2^32) and g uses only w and private
constants (e.g. v = (A*z+B) ^ k, (A*z+B) & k, w*w + w).

Fiber lemma (as in rda.py). On the fiber z = z0 + 2^c*y, with A = 2^e*A1 (A1 odd):
    w = B_lo + 2^e*m(z0) + 2^(c+e)*h,   m(z0) = (A1*z0 + B_h) mod 2^min(c, 32-e),
B_lo < 2^e, and h takes every value in [0, 2^(32-c-e)) (h = 0 only when e >= 32-c;
A = 0 is e = 32, w = B_lo). Every other runtime value is a T-function of z, fixed mod
2^c on the fiber. Hence the low c output bits match tau for all 32-bit z iff they match
for every (z0, h) when w is replaced by the expression above, g and the shift are
evaluated exactly at 32 bits, and the rest at c bits. Constants of instructions on the
g path are 32-bit symbols whose low c bits are the c-bit constants. Parameters e, s are
symbolic (s in [0, 32], s = 32 meaning every s >= 32); one query covers every shift.
UNSAT excludes the wiring at c output bits for every 32-bit constant choice.
"""
import random
import sys
import time
sys.path.insert(0, '/home/user/aopt')
import z3
from rda import affine_kinds, tau_z3


def roots(order, wiring, upto):
    """For each value: frozenset of affine runtime values its non-affine part depends on."""
    kinds = affine_kinds(order, wiring, upto)
    rts = [frozenset([0])]
    for i in range(upto):
        if kinds[i + 1] == 'a':
            rts.append(frozenset([i + 1]))
            continue
        if kinds[i + 1] == 'c':
            rts.append(frozenset())
            continue
        acc = set()
        for j in wiring[i]:
            if j <= i:
                acc |= rts[j]
        rts.append(frozenset(acc))
    return kinds, rts


def g_path(order, wiring, pos):
    """(w, path) if v = g(w) for a single affine runtime value w, else None.
    path = instruction indices i (< pos) whose result lies between w and v."""
    if list(order).count('>>') != 1:
        return None
    vi, ki = wiring[pos]
    if vi > pos or ki <= pos:
        return None
    kinds, rts = roots(order, wiring, pos)
    if kinds[vi] != 'o' or len(rts[vi]) != 1:
        return None
    w = next(iter(rts[vi]))
    path = set()

    def visit(x):
        if x == w or x == 0 and w == 0 or kinds[x] != 'o':
            return
        i = x - 1
        if i in path:
            return
        path.add(i)
        for j in wiring[i]:
            if j <= i:
                visit(j)
    visit(vi)
    # every non-constant leaf of the path must be w itself
    for i in path:
        for j in wiring[i]:
            if j <= i and j != w and kinds[j] != 'c' and (j - 1) not in path:
                return None
    return w, sorted(path)


def applicable(order, wiring):
    return g_path(order, wiring, order.index('>>')) is not None if '>>' in order else False


def _op(op, a, b, c):
    if op == 'mad':
        return a * b + c
    if op == '^':
        return a ^ b
    if op == '&':
        return a & b
    return a | b


def build(order, wiring, pos, w, path, P, z0, h, c):
    """Return (out_c, w_c, w32_low_c) for sample (z0, h) as z3 expressions."""
    K32 = P['k32']
    kc = [(z3.Extract(c - 1, 0, a), z3.Extract(c - 1, 0, b)) for a, b in K32]
    vals = [z0]
    for i in range(pos):
        pool = vals + [kc[i][0], kc[i][1]]
        a = [pool[j] for j in wiring[i]] + [None] * (3 - len(wiring[i]))
        vals.append(_op(order[i], a[0], a[1], a[2]))
    # 32-bit fiber value of w and the g path
    e = P['e']
    e32 = z3.ZeroExt(24, e)
    cmask = z3.If(z3.UGT(e, 32 - c), z3.LShR(z3.BitVecVal((1 << c) - 1, c), z3.Extract(c - 1, 0, e32) - (32 - c)),
                  z3.BitVecVal((1 << c) - 1, c))
    m = (P['A1'] * z0 + P['Bh']) & cmask
    w32 = P['Blo'] + (z3.ZeroExt(32 - c, m) << e32) + (z3.ZeroExt(c, h) << (e32 + c))
    v32 = {w: w32}
    for i in path:
        pool32 = {}
        for j in wiring[i]:
            pass
        args = []
        for j in wiring[i]:
            if j <= i:
                args.append(v32[j] if j in v32 else z3.ZeroExt(32 - c, vals[j]))
            else:
                args.append(K32[i][j - i - 1])
        args += [None] * (3 - len(args))
        v32[i + 1] = _op(order[i], args[0], args[1], args[2])
    vi = wiring[pos][0]
    s32 = z3.ZeroExt(24, P['s'])
    u = z3.Extract(c - 1, 0, z3.LShR(v32[vi], s32))
    vals = vals + [u]
    for i in range(pos + 1, len(order)):
        pool = vals + [kc[i][0], kc[i][1]]
        a = [pool[j] for j in wiring[i]] + [None] * (3 - len(wiring[i]))
        vals.append(_op(order[i], a[0], a[1], a[2]))
    return vals[-1], vals[w], z3.Extract(c - 1, 0, w32)


def h_ok(P, h, c):
    e = P['e']
    room = z3.BitVecVal(32 - c, 8) - e                        # valid when e <= 32-c
    return z3.If(z3.UGE(e, 32 - c), h == 0,
                 z3.LShR(h, z3.ZeroExt(c - 8, room) if c > 8 else z3.Extract(c - 1, 0, room)) == 0)


def solve(order, wiring, c=11, seconds=120.0, seed=1, fix_e=None, fix_s=None, s_range=None):
    pos = order.index('>>')
    gp = g_path(order, wiring, pos)
    assert gp is not None, 'not a g(w) wiring'
    w, path = gp
    cm = (1 << c) - 1
    hw = 32 - c                                               # h width
    S = z3.SolverFor('QF_BV')
    P = dict(e=z3.BitVec('e', 8), s=z3.BitVec('s', 8), A1=z3.BitVec('A1', c), Bh=z3.BitVec('Bh', c),
             Blo=z3.BitVec('Blo', 32), K=z3.BitVec('K', c),
             k32=[(z3.BitVec(f'k{i}_0', 32), z3.BitVec(f'k{i}_1', 32)) for i in range(len(order))])
    S.add(z3.ULE(P['e'], 32), z3.ULE(P['s'], 32), z3.Extract(0, 0, P['A1']) == 1,
          z3.LShR(P['Blo'], z3.ZeroExt(24, P['e'])) == 0)
    if fix_e is not None:
        S.add(P['e'] == fix_e)
    if fix_s is not None:
        S.add(P['s'] == fix_s)
    if s_range is not None:
        S.add(z3.UGE(P['s'], s_range[0]), z3.ULE(P['s'], s_range[1]))

    def constrain(z0v, hv):
        z0 = z3.BitVecVal(z0v, c)
        h = z3.BitVecVal(hv, hw)
        out, wc, w32c = build(order, wiring, pos, w, path, P, z0, h, c)
        S.add(wc == w32c)                                     # consistency (h-independent)
        S.add(z3.Implies(h_ok(P, h, hw), out == tau_z3(z0, P['K'], cm)))

    rng = random.Random(seed)
    hm = (1 << hw) - 1
    pts = [(0, 0), (1, 0), (2, 0), (3, 0), (cm, 0), (1, 1), (5, 2), (7, hm), (0, hm), (1 << (c - 1), 1 << (hw - 1))]
    pts += [(rng.getrandbits(c), rng.getrandbits(hw)) for _ in range(12)]
    pts += [(rng.getrandbits(c), rng.getrandbits(rng.randrange(1, hw))) for _ in range(8)]
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
                  k32=[(ev(a), ev(b)) for a, b in P['k32']])
        V = z3.SolverFor('QF_BV')
        zv = z3.BitVec('zv', c)
        hv = z3.BitVec('hv', hw)
        out, wc, w32c = build(order, wiring, pos, w, path, Pc, zv, hv, c)
        V.add(z3.Or(wc != w32c, z3.And(h_ok(Pc, hv, hw), out != tau_z3(zv, Pc['K'], cm))))
        V.set(timeout=max(1, int(left * 1000)))
        vr = V.check()
        if vr == z3.unsat:
            return dict(result='sat', c=c, it=it, e=Pc['e'].as_long(), s=Pc['s'].as_long(),
                        consts=[(a.as_long(), b.as_long()) for a, b in Pc['k32']], K=Pc['K'].as_long(),
                        A1=Pc['A1'].as_long(), Bh=Pc['Bh'].as_long(), Blo=Pc['Blo'].as_long())
        if vr == z3.unknown:
            return dict(result='timeout', c=c, it=it)
        vm = V.model()
        constrain(vm.eval(zv, model_completion=True).as_long(), vm.eval(hv, model_completion=True).as_long())


if __name__ == '__main__':
    tests = [(('mad', '^', '>>', 'mad'), ((0, 1, 2), (1, 2), (2, 3), (0, 3, 2))),
             (('mad', 'mad', '>>', '^'), ((0, 1, 2), (1, 1, 1), (2, 3), (3, 4)))]
    for o, wi in tests:
        print(o, wi, g_path(o, wi, o.index('>>')))
        t = time.time()
        r = solve(o, wi, 11, 300)
        print(' ', r['result'], r.get('it'), '%.1fs' % (time.time() - t), flush=True)
