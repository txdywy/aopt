"""rdvf generalised: single-SHR wirings u = a >> b whose runtime shift amount is
b = G_b(w_b): w_b an affine runtime value (w_b = A_b*z + B_b) and G_b a bitwise
composition (XOR/AND/OR) with private constants (G_b = identity is rdvf.py). The value a
is a private constant, or G_a(w_a) for an affine root w_a and a bitwise G_a.

Fiber z = z0 + 2^c*y (c >= 11), A_b = 2^e*A1 (A1 odd; A_b = 0 is e = 32):
  w_b = T + 2^(c+e)*h, T = B_lo + 2^e*m(z0), h free in [0, 2^(32-c-e)) (h = 0 if e >= 32-c).
  Bitwise G acts bit by bit, so the bits of b at positions >= c+e are G applied to h and
  are the same function of h on every class. Let HB = (G_b(0) | G_b(~0)) >> (c+e).
  (i) e <= 31-c and HB != 0: some bit j >= c+e of b is non-zero for a suitable h, on every
      class: b >= 2^c >= 32, u = 0 there, the output equals H(z0, 0), and a <= 3 T-op
      program would match F(z^K)^D mod 2^11 on all classes: impossible (closure3).
  (ii) otherwise (e >= 32-c, or HB == 0): b = G_b(T) is constant on each fiber. The value:
      a constant; or G_a(w_a) with w_a = w_b (then G_a(T + 2^(c+e)*h) with the same free h);
      or w_a another affine root, whose own high part is free on the fiber, independent of
      b. Everything on the two shift operands is evaluated exactly at 32 bits.
Branch (ii) is searched with CEGIS over (z0, h). UNSAT excludes the wiring at c >= 11
output bits for every 32-bit constant choice.
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


def bitwise_root(order, wiring, kinds, x):
    """(root, path) if value x is a bitwise composition of ONE affine runtime value."""
    if kinds[x] == 'a':
        return x, []
    if kinds[x] != 'o':
        return None
    path, roots = set(), set()

    def visit(v):
        if kinds[v] == 'a':
            roots.add(v)
            return True
        if kinds[v] == 'c':
            return True
        i = v - 1
        if order[i] not in ('^', '&', '|'):
            return False
        path.add(i)
        return all(visit(j) if j <= i else True for j in wiring[i])
    if not visit(x) or len(roots) != 1:
        return None
    return next(iter(roots)), sorted(path)


def analyse(order, wiring):
    if list(order).count('>>') != 1:
        return None
    pos = order.index('>>')
    ai, bi = wiring[pos]
    if bi > pos:
        return None
    kinds = affine_kinds(order, wiring, pos)
    rb = bitwise_root(order, wiring, kinds, bi)
    if rb is None:
        return None
    if ai > pos:
        va = None
    else:
        va = bitwise_root(order, wiring, kinds, ai)
        if va is None:
            return None
    return dict(pos=pos, ai=ai, bi=bi, b_root=rb[0], b_path=rb[1],
                a_root=None if va is None else va[0], a_path=[] if va is None else va[1])


def applicable(order, wiring):
    return analyse(order, wiring) is not None


class Root:
    """Fiber parameters of one affine root."""

    def __init__(self, tag, c, S):
        self.c = c
        self.e = z3.BitVec(f'e_{tag}', 8)
        self.A1 = z3.BitVec(f'A1_{tag}', c)
        self.Bh = z3.BitVec(f'Bh_{tag}', c)
        self.Blo = z3.BitVec(f'Blo_{tag}', 32)
        S.add(z3.ULE(self.e, 32), z3.Extract(0, 0, self.A1) == 1,
              z3.LShR(self.Blo, z3.ZeroExt(24, self.e)) == 0)

    def params(self, mo):
        g = lambda x: mo.eval(x, model_completion=True)
        return dict(e=g(self.e), A1=g(self.A1), Bh=g(self.Bh), Blo=g(self.Blo))

    def sym(self):
        return dict(e=self.e, A1=self.A1, Bh=self.Bh, Blo=self.Blo)


def root_value(p, c, z0, h):
    """(T, w32): class part and full fiber value of an affine root."""
    cm = (1 << c) - 1
    e32 = z3.ZeroExt(24, p['e'])
    keep = z3.If(z3.UGT(p['e'], 32 - c), z3.LShR(z3.BitVecVal(cm, c), z3.Extract(c - 1, 0, e32) - (32 - c)),
                 z3.BitVecVal(cm, c))
    m = (p['A1'] * z0 + p['Bh']) & keep
    T = p['Blo'] + (z3.ZeroExt(32 - c, m) << e32)
    w32 = T + (z3.ZeroExt(c, h) << (e32 + c))
    return T, w32


def h_free(p, c, h):
    room = z3.BitVecVal(32 - c, 8) - p['e']
    return z3.If(z3.UGE(p['e'], 32 - c), h == 0, z3.LShR(h, z3.ZeroExt(32 - c - 8, room)) == 0)


def eval_path(order, wiring, path, root, x0, k32):
    sv = {root: x0}
    for i in path:
        args = [sv[j] if j <= i else k32[i][j - i - 1] for j in wiring[i]]
        sv[i + 1] = _op(order[i], args[0], args[1], None)
    return sv


def solve(order, wiring, c=11, seconds=120.0, seed=1, target=None, allow_low_c=False):
    assert c >= 11 or target is not None or allow_low_c
    an = analyse(order, wiring)
    assert an is not None
    pos, ai, bi = an['pos'], an['ai'], an['bi']
    cm = (1 << c) - 1
    hw = 32 - c
    S = z3.SolverFor('QF_BV')
    k32 = [(z3.BitVec(f'k{i}_0', 32), z3.BitVec(f'k{i}_1', 32)) for i in range(len(order))]
    K = z3.BitVec('K', c)
    if target is not None:
        S.add(K == 0)
    RB = Root('b', c, S)
    same = an['a_root'] is not None and an['a_root'] == an['b_root']
    RA = Root('a', c, S) if (an['a_root'] is not None and not same) else None
    ones = z3.BitVecVal(0xFFFFFFFF, 32)
    zero = z3.BitVecVal(0, 32)

    def branch_ii(P, kk):
        g0 = eval_path(order, wiring, an['b_path'], an['b_root'], zero, kk)[bi]
        g1 = eval_path(order, wiring, an['b_path'], an['b_root'], ones, kk)[bi]
        e = P['b']['e']
        sh = z3.ZeroExt(24, e) + c
        return z3.Or(z3.UGE(e, 32 - c), z3.LShR(g0 | g1, sh) == 0)

    def build(P, kk, z0, h):
        kc = [(z3.Extract(c - 1, 0, a), z3.Extract(c - 1, 0, b)) for a, b in kk]
        vals = [z0]
        for i in range(pos):
            pool = vals + [kc[i][0], kc[i][1]]
            a = [pool[j] for j in wiring[i]] + [None] * (3 - len(wiring[i]))
            vals.append(_op(order[i], a[0], a[1], a[2]))
        Tb, wb32 = root_value(P['b'], c, z0, h)
        cons = [vals[an['b_root']] == z3.Extract(c - 1, 0, Tb)]
        b32 = eval_path(order, wiring, an['b_path'], an['b_root'], Tb, kk)[bi]
        if an['a_root'] is None:
            a32 = kk[pos][ai - pos - 1]
            hok = h == 0
        elif same:
            a32 = eval_path(order, wiring, an['a_path'], an['a_root'], wb32, kk)[ai]
            hok = h_free(P['b'], c, h)
        else:
            Ta, wa32 = root_value(P['a'], c, z0, h)
            cons.append(vals[an['a_root']] == z3.Extract(c - 1, 0, Ta))
            a32 = eval_path(order, wiring, an['a_path'], an['a_root'], wa32, kk)[ai]
            hok = h_free(P['a'], c, h)
        u = z3.Extract(c - 1, 0, z3.LShR(a32, b32))
        vals = vals + [u]
        for i in range(pos + 1, len(order)):
            pool = vals + [kc[i][0], kc[i][1]]
            a = [pool[j] for j in wiring[i]] + [None] * (3 - len(wiring[i]))
            vals.append(_op(order[i], a[0], a[1], a[2]))
        return vals[-1], cons, hok

    P = dict(b=RB.sym(), a=RA.sym() if RA else None)
    S.add(branch_ii(P, k32))

    def constrain(zv, hv):
        z0 = z3.BitVecVal(zv, c)
        h = z3.BitVecVal(hv, hw)
        out, cons, hok = build(P, k32, z0, h)
        S.add(*cons)
        want = z3.BitVecVal(int(target[zv]), c) if target is not None else tau_z3(z0, K, cm)
        S.add(z3.Implies(hok, out == want))

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
        Pc = dict(b=RB.params(mo), a=RA.params(mo) if RA else None)
        Kv = mo.eval(K, model_completion=True)
        V = z3.SolverFor('QF_BV')
        zv = z3.BitVec('zv', c)
        hv = z3.BitVec('hv', hw)
        out, cons, hok = build(Pc, kk, zv, hv)
        if target is not None:
            tz = z3.Function('tz', z3.BitVecSort(c), z3.BitVecSort(c))
            for zz in range(1 << c):
                V.add(tz(z3.BitVecVal(zz, c)) == int(target[zz]))
            want = tz(zv)
        else:
            want = tau_z3(zv, Kv, cm)
        V.add(z3.Or(z3.Not(z3.And(*cons)), z3.And(hok, out != want)))
        V.set(timeout=max(1, int(left * 1000)))
        vr = V.check()
        if vr == z3.unsat:
            return dict(result='sat', it=it, consts=[(a.as_long(), b.as_long()) for a, b in kk], K=Kv.as_long())
        if vr == z3.unknown:
            return dict(result='timeout', it=it)
        vm = V.model()
        constrain(vm.eval(zv, model_completion=True).as_long(), vm.eval(hv, model_completion=True).as_long())


if __name__ == '__main__':
    for o, w in [(('mad', '^', '>>', 'mad'), ((0, 1, 2), (1, 2), (0, 2), (0, 3, 2))),
                 (('mad', '^', '>>', 'mad'), ((0, 1, 2), (1, 2), (1, 2), (0, 3, 2))),
                 (('mad', 'mad', '>>', 'mad'), ((0, 1, 2), (0, 2, 3), (1, 2), (0, 3, 2)))]:
        print(o, w, analyse(o, w), flush=True)
        t = time.time()
        r = solve(o, w, 11, 300)
        print('  ', r['result'], r.get('it'), '%.1fs' % (time.time() - t), flush=True)
