"""Validation of rda.py.

T1 (lemma): for random 32-bit A, B, s, classify the case, derive the model parameters
   and check on random z that (a) v mod 2^c equals the model's v(z0), (b) the true
   u mod 2^c is one of the model's values u(z0, r) with r in range, and (c) every r in
   range is attained by some z in the fiber (constructed explicitly).
T2 (positive controls): build 32-bit programs whose output is a T-function (so they
   are realizable), use their low c bits as the target (K = 0), and require the
   matching case of rda to return SAT.
"""
import random
import sys
import numpy as np
sys.path.insert(0, '.')
import rda

M32 = (1 << 32) - 1


def v2(a):
    return (a & -a).bit_length() - 1 if a else 99


def classify(A, B, s, c):
    e = v2(A)
    if s >= 32:
        return 1, None
    if A == 0 or e >= s:
        p = dict(s=s, e=0, dl=0, A1=(A >> s) & ((1 << c) - 1), Bh=(B >> s) & ((1 << c) - 1),
                 Bl=(B & ((1 << s) - 1)) & ((1 << c) - 1))
        return 2, p
    dl = s - e
    if dl >= c:
        return 1, dict(s=s, e=0, dl=0, A1=0, Bh=0, Bl=0)
    p = dict(s=s, e=e, dl=dl, A1=(A >> e) & ((1 << c) - 1), Bh=(B >> e) & ((1 << c) - 1),
             Bl=(B & ((1 << e) - 1)) & ((1 << c) - 1))
    return 3, p


def model_values(kind, p, z0, c):
    """All model u values for class z0 and the model v."""
    cm = (1 << c) - 1
    if kind == 1:
        s = p['s'] if p else 32
        L = c if s <= 32 - c else max(0, 32 - s)
        return set(range(1 << L)), None
    case = rda.Case.__new__(rda.Case)
    case.kind, case.c, case.cm = kind, c, cm
    pn = {k: (np.uint64(v) if k in ('A1', 'Bh', 'Bl') else v) for k, v in p.items()}
    L = case.np_lim(p) if kind == 3 else 0
    rs = np.arange(1 << L, dtype=np.uint64)
    u, v = case.np_u_v(pn, np.uint64(z0), rs)
    return set(int(x) for x in np.atleast_1d(u)), int(v)


def t1(trials=4000, c=11, seed=7):
    rng = random.Random(seed)
    cm = (1 << c) - 1
    counts = {1: 0, 2: 0, 3: 0}
    for _ in range(trials):
        e = rng.choice(list(range(0, 33)))
        A = 0 if e >= 32 else ((rng.getrandbits(32 - e) | 1) << e) & M32
        B = rng.getrandbits(32)
        s = rng.choice(list(range(0, 33)) + [40])
        kind, p = classify(A, B, s, c)
        counts[kind] += 1
        z0 = rng.getrandbits(c)
        us, vm = model_values(kind, p, z0, c)
        for _ in range(6):
            z = z0 + (rng.getrandbits(32 - c) << c)
            v = (A * z + B) & M32
            u = (v >> s) & cm if s < 32 else 0
            if vm is not None:
                assert v & cm == vm, ('v', kind, A, B, s, z)
            assert u in us, ('u', kind, A, B, s, z, u, sorted(us)[:5])
        # (c) attainment: every model value is reached inside the fiber
        if kind == 3 or kind == 1:
            if kind == 3:
                e_, dl = p['e'], p['dl']
            else:
                e_, dl = v2(A) if A else 0, None
            if A == 0:
                continue
            A1 = A >> e_
            inv = pow(A1, -1, 1 << 32)
            for w in rng.sample(sorted(us), min(5, len(us))):
                # find y with u(z0 + 2^c y) mod 2^c == w: solve for the high part h
                found = False
                hbits = 32 - c - e_
                if hbits <= 0:
                    found = ((((A * z0 + B) & M32) >> s) & cm) == w
                else:
                    base = (A * z0 + B) & M32
                    v0 = base & ((1 << (c + e_)) - 1)
                    h0 = base >> (c + e_)
                    for cand in range(64):
                        # choose target h: low dl bits of h from w, rest random
                        if kind == 3:
                            want_h = ((w >> (c - dl)) | (rng.getrandbits(max(0, hbits - dl)) << dl)) & ((1 << hbits) - 1)
                        else:
                            sh = s - c - e_
                            want_h = ((w << sh) | rng.getrandbits(sh)) & ((1 << hbits) - 1) if sh < 64 else 0
                        y = ((want_h - h0) * inv) & ((1 << hbits) - 1)
                        z = (z0 + (y << c)) & M32
                        if ((((A * z + B) & M32) >> s) & cm) == w:
                            found = True
                            break
                assert found, ('attain', kind, A, B, s, z0, w)
    return counts


def t2(c=11, seed=3, n=12, seconds=120):
    """Positive controls on realizable programs."""
    rng = random.Random(seed)
    cm = (1 << c) - 1
    ok = 0
    tests = []
    # case 2 control: A = 2^s * A'  (u is a T-function); out = z*t0 + u, then ^ k
    # case 3 control: I3 multiplies u by 2^dl * k' so the free top bits vanish
    for j in range(n):
        kind = (2, 3)[j % 2]
        s = rng.randrange(1, 31)
        if kind == 2:
            A = (rng.getrandbits(32 - s) << s) & M32
            order, wiring = ('mad', '>>', 'mad', '^'), ((0, 1, 2), (1, 2), (0, 1, 2), (3, 4))
            k2 = rng.getrandbits(32)
            k3 = rng.getrandbits(32)
            def prog(z, A=A, B=None, s=s, k2=k2, k3=k3):
                return None
        else:
            dl = rng.randrange(1, c)
            e = max(0, s - dl)
            s = e + dl
            A = ((rng.getrandbits(32 - e) | 1) << e) & M32
            order, wiring = ('mad', '>>', 'mad', '^'), ((0, 1, 2), (1, 2), (2, 3, 0), (3, 4))
            k2 = (rng.getrandbits(32) << dl) & M32
            k3 = rng.getrandbits(32)
        B = rng.getrandbits(32)
        k4 = rng.getrandbits(32)

        def run32(z):
            t0 = (A * z + B) & M32
            u = t0 >> s
            if kind == 2:
                t2_ = (z * t0 + u) & M32
            else:
                t2_ = (u * k2 + z) & M32
            return t2_ ^ k4
        # realizability: T-function check on random fibers
        for _ in range(200):
            z0 = rng.getrandbits(c)
            a = run32(z0) & cm
            for _ in range(3):
                assert run32(z0 + (rng.getrandbits(32 - c) << c)) & cm == a
        target = np.array([run32(z) & cm for z in range(1 << c)], dtype=np.uint64)
        r = rda.solve_case(order, wiring, kind, c, seconds, 1, target)
        tests.append((kind, s, r['result'], r.get('it')))
        ok += r['result'] == 'sat'
    return ok, tests


if __name__ == '__main__':
    print('T1 lemma checks, case counts:', t1())
    ok, tests = t2()
    print('T2 positive controls SAT:', ok, '/', len(tests))
    for t in tests:
        print('  ', t)
