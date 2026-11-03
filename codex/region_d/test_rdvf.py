"""Validation of rdvf.
(a) branch (i): v2(A_b) <= 31-c  =>  every class z0 has y in {0,1} with b >= 2^c >= 32.
(b) branch (ii): v2(A_b) >= 32-c =>  b(z0 + 2^c y) == B_b + ((A' z0) mod 2^c) << (32-c).
(c) positive controls: realizable programs (output provably a T-function) as targets;
    rdvf must return SAT. Value kinds: free root, same root, constant.
"""
import random, sys
import numpy as np
sys.path.insert(0, '.')
import rdvf
M = (1 << 32) - 1
c = 11
cm = (1 << c) - 1
rng = random.Random(21)


def v2(a):
    return (a & -a).bit_length() - 1 if a else 99


def part_a(n=20000):
    for _ in range(n):
        e = rng.randrange(0, 32 - c)
        Ab = ((rng.getrandbits(32 - e) | 1) << e) & M
        Bb = rng.getrandbits(32)
        z0 = rng.getrandbits(c)
        ok = any(((Ab * (z0 + (y << c)) + Bb) & M) >= 32 for y in (0, 1))
        assert ok, (Ab, Bb, z0)
    return n


def part_b(n=20000):
    for _ in range(n):
        e = rng.randrange(32 - c, 33)
        Ab = 0 if e == 32 else ((rng.getrandbits(32 - e) | 1) << e) & M
        Bb = rng.getrandbits(32)
        Ap = (Ab >> (32 - c)) & cm
        z = rng.getrandbits(32)
        z0 = z & cm
        assert (Ab * z + Bb) & M == (Bb + (((Ap * z0) & cm) << (32 - c))) & M
    return n


def run32(order, wiring, consts, z):
    vals = [z]
    for i, op in enumerate(order):
        pool = vals + list(consts[i])
        a = [pool[j] for j in wiring[i]]
        if op == 'mad': v = (a[0] * a[1] + a[2]) & M
        elif op == '^': v = a[0] ^ a[1]
        elif op == '&': v = a[0] & a[1]
        elif op == '|': v = a[0] | a[1]
        else: v = (a[0] >> a[1]) if a[1] < 32 else 0
        vals.append(v)
    return vals[-1]


def part_c():
    res = []
    controls = []
    # free root: u = t0 >> t1, t0 = A z + B with v2(A) = 8, t1 = 2^31 z + 5 (b in {5, 2^31+5})
    o = ('mad', 'mad', '>>', '^'); w = ((0, 1, 2), (0, 2, 3), (1, 2), (3, 4))
    for trial in range(4):
        A = ((rng.getrandbits(24) | 1) << 8) & M
        consts = [(A, rng.getrandbits(32)), (1 << 31, rng.randrange(0, 8)), (0, 0), (rng.getrandbits(32), 0)]
        controls.append(('free', o, w, consts))
    # free root through a bitwise path: u = (t0 ^ k) >> t2 ... use ('mad','^','mad','>>') is SHR last: skip;
    # value t1 = t0 ^ k, amount t0' affine: ('mad','^','>>','^') with SHR (2, 1)? amount t0 = value root -> 'same'
    o2 = ('mad', '^', '>>', '^'); w2 = ((0, 1, 2), (1, 2), (2, 1), (3, 4))
    for trial in range(3):
        # t0 = 2^22*z*Ap + B (fixed per class), t1 = t0 ^ k, u = t1 >> t0
        Ap = rng.getrandbits(10) << 22
        consts = [(Ap, rng.randrange(0, 40)), (rng.getrandbits(32), 0), (0, 0), (rng.getrandbits(32), 0)]
        controls.append(('same', o2, w2, consts))
    # constant value: u = K >> t1 with t1 affine class-fixed
    o3 = ('mad', 'mad', '>>', 'mad'); w3 = ((0, 1, 2), (0, 2, 3), (3, 2), (3, 4, 0))
    for trial in range(3):
        consts = [(rng.getrandbits(32), rng.getrandbits(32)), (rng.getrandbits(11) << 21, rng.randrange(0, 40)),
                  (0, rng.getrandbits(32)), (rng.getrandbits(32), 0)]
        controls.append(('const', o3, w3, consts))
    for kind, o, w, consts in controls:
        # realizability check: output mod 2^c constant on fibers
        tgt = np.array([run32(o, w, consts, z) & cm for z in range(1 << c)], dtype=np.uint64)
        for _ in range(300):
            z = rng.getrandbits(32)
            assert run32(o, w, consts, z) & cm == tgt[z & cm], (kind, o, w, consts)
        an = rdvf.analyse(o, w)
        assert an['value'] == kind, (an, kind)
        r = rdvf.solve(o, w, c, 300, 1, tgt)
        res.append((kind, r['result'], r.get('it')))
    return res


if __name__ == '__main__':
    print('a', part_a(), 'b', part_b(), flush=True)
    out = part_c()
    print('c positive controls:', out)
    print('all SAT:', all(r[1] == 'sat' for r in out))
