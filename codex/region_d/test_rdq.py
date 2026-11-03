"""Validation of rdq: (1) V_low formula, (2) freedom of the high part on fibers,
(3) positive controls (realizable programs with s <= e) must be SAT."""
import random, sys
import numpy as np
sys.path.insert(0, '.')
import rdq
M = (1 << 32) - 1
c = 11; cm = (1 << c) - 1
rng = random.Random(33)


def vlow_formula(A, B, z0):
    e = (A & -A).bit_length() - 1 if A else 32
    A1 = (A >> e) if A else 0
    Blo = B & ((1 << e) - 1) if e < 32 else B
    Bh = (B >> e) & cm if e < 32 else 0
    keep = cm >> (e - (32 - c)) if e > 32 - c else cm
    m = ((A1 * z0 + Bh) & cm) & keep if e < 32 else 0
    blc = Blo & cm
    p1 = (m * (2 * blc + 1)) & cm
    p2 = (m * m) & cm
    V = (Blo * Blo + Blo + (p1 << e) + (p2 << (2 * e))) & M
    width = min(32, c + e)
    return V & ((1 << width) - 1), width, e


def part1(n=20000):
    for _ in range(n):
        e = rng.randrange(0, 33)
        A = 0 if e == 32 else ((rng.getrandbits(32 - e) | 1) << e) & M
        B = rng.getrandbits(32)
        z = rng.getrandbits(32)
        V, width, _ = vlow_formula(A, B, z & cm)
        t = (A * z + B) & M
        v = (t * t + t) & M
        assert v & ((1 << width) - 1) == V, (A, B, z)
    return n


def part2(n=300):
    for _ in range(n):
        e = rng.randrange(32 - c - 8, 32 - c)
        A = ((rng.getrandbits(32 - e) | 1) << e) & M
        B = rng.getrandbits(32)
        z0 = rng.getrandbits(c)
        hb = 32 - c - e
        seen = set()
        for y in range(1 << hb):
            z = z0 + (y << c)
            t = (A * z + B) & M
            seen.add(((t * t + t) & M) >> (c + e))
        assert len(seen) == 1 << hb, (A, B, z0, len(seen))
    return n


def run32(o, w, consts, z):
    vals = [z]
    for i, op in enumerate(o):
        pool = vals + list(consts[i])
        a = [pool[j] for j in w[i]]
        if op == 'mad': v = (a[0] * a[1] + a[2]) & M
        elif op == '^': v = a[0] ^ a[1]
        elif op == '&': v = a[0] & a[1]
        elif op == '|': v = a[0] | a[1]
        else: v = (a[0] >> a[1]) if a[1] < 32 else 0
        vals.append(v)
    return vals[-1]


def part3():
    o, w = ('mad', 'mad', '>>', '^'), ((0, 1, 2), (1, 1, 1), (2, 3), (3, 4))
    res = []
    for t in range(6):
        e = rng.randrange(1, 22)
        s = rng.randrange(0, e + 1)
        A = ((rng.getrandbits(32 - e) | 1) << e) & M
        consts = [(A, rng.getrandbits(32)), (0, 0), (s, 0), (rng.getrandbits(32), 0)]
        tgt = np.array([run32(o, w, consts, z) & cm for z in range(1 << c)], dtype=np.uint64)
        for _ in range(300):
            z = rng.getrandbits(32)
            assert run32(o, w, consts, z) & cm == tgt[z & cm]
        r = rdq.solve(o, w, c, 300, 1, tgt)
        res.append((e, s, r['result'], r.get('it')))
    # s > e with u insensitive: out = u*2^c... not expressible with ^; use s >= 32 (u = 0)
    consts = [(rng.getrandbits(32) | 1, rng.getrandbits(32)), (0, 0), (40, 0), (rng.getrandbits(32), 0)]
    tgt = np.array([run32(o, w, consts, z) & cm for z in range(1 << c)], dtype=np.uint64)
    r = rdq.solve(o, w, c, 300, 1, tgt)
    res.append(('u=0', 40, r['result'], r.get('it')))
    return res


if __name__ == '__main__':
    print('part1', part1(), 'part2', part2(), flush=True)
    r = part3()
    print('part3', r, 'all SAT:', all(x[2] == 'sat' for x in r))
