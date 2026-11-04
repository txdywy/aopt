"""Validation of rdp2 (no c=8 replay: its exact relation needs 3c >= 32).
(a) Taylor identity for random degree-4 polynomials; (b) exhaustive fiber freedom for
d in [8, 10]; (c) coeffs() equals the program's own value on random constants and z;
(d) realizable positive controls must be SAT."""
import random, sys
import numpy as np
sys.path.insert(0, '.')
import z3
import rdp2, test_rdq
from drive import all_wirings
M = (1 << 32) - 1
c = 11; cm = (1 << c) - 1
rng = random.Random(2024)


def jets_py(a, z0):
    V0 = sum(ai * pow(z0, i, 1 << 32) for i, ai in enumerate(a)) & M
    D1 = sum(i * ai * pow(z0, i - 1, 1 << 32) for i, ai in enumerate(a) if i >= 1) & M
    D2 = sum((i * (i - 1) // 2) * ai * pow(z0, i - 2, 1 << 32) for i, ai in enumerate(a) if i >= 2) & M
    return V0, D1, D2


def part_a(n=20000):
    for _ in range(n):
        a = [rng.getrandbits(32) for _ in range(5)]
        z0 = rng.getrandbits(c); y = rng.getrandbits(32 - c)
        f = lambda z: sum(ai * pow(z, i, 1 << 32) for i, ai in enumerate(a)) & M
        V0, D1, D2 = jets_py(a, z0)
        assert f(z0 + (y << c)) == (V0 + ((y * D1) << c) + ((y * y * D2) << (2 * c))) & M
    return n


def part_b(n=150):
    cnt = 0
    while cnt < n:
        a = [rng.getrandbits(32) for _ in range(5)]
        z0 = rng.getrandbits(c)
        d = rng.randrange(8, c)
        _, D1, _ = jets_py(a, z0)
        # adjust a1 so that v2(D1 mod 2^c) == d
        target = (1 << d) | (rng.getrandbits(32) << (d + 1))
        a[1] = (a[1] + target - D1) & M
        V0, D1, D2 = jets_py(a, z0)
        assert ((D1 & cm) & -(D1 & cm)).bit_length() - 1 == d
        f = lambda z: sum(ai * pow(z, i, 1 << 32) for i, ai in enumerate(a)) & M
        nb = 32 - c - d
        seen = set()
        for y in range(1 << nb):
            v = f(z0 + (y << c))
            assert v & ((1 << (c + d)) - 1) == V0 & ((1 << (c + d)) - 1)
            seen.add(v >> (c + d))
        assert len(seen) == 1 << nb
        cnt += 1
    return cnt


def part_c(n=300):
    ws = [(o, w) for o, w in all_wirings(1, 2) if rdp2.applicable(o, w)]
    for _ in range(n):
        o, w = rng.choice(ws)
        an = rdp2.analyse(o, w)
        consts = [(rng.getrandbits(32), rng.getrandbits(32)) for _ in range(4)]
        k32 = [(z3.BitVecVal(a, 32), z3.BitVecVal(b, 32)) for a, b in consts]
        cof = [z3.simplify(x).as_long() for x in rdp2.coeffs(o, w, k32, an['pos'])[an['vi']]]
        z = rng.getrandbits(32)
        vals = [z]
        for i in range(an['pos']):
            pool = vals + list(consts[i])
            args = [pool[j] for j in w[i]]
            op = o[i]
            if op == 'mad': vals.append((args[0] * args[1] + args[2]) & M)
            elif op == '^': vals.append(args[0] ^ args[1])
            elif op == '&': vals.append(args[0] & args[1])
            else: vals.append(args[0] | args[1])
        assert sum(ai * pow(z, i, 1 << 32) for i, ai in enumerate(cof)) & M == vals[an['vi']], (o, w)
    return n


def part_d():
    out = []
    o, w = ('mad', 'mad', '>>', '^'), ((0, 0, 1), (1, 2, 3), (2, 3), (3, 4))
    for t in range(5):
        k2 = ((rng.getrandbits(26) | 1) << 6) & M          # v = k2*z^2 + const, d >= 7
        s = rng.randrange(0, 8)
        consts = [(0, rng.getrandbits(32)), (k2, rng.getrandbits(32)), (s, 0), (rng.getrandbits(32), 0)]
        tgt = np.array([test_rdq.run32(o, w, consts, z) & cm for z in range(1 << c)], dtype=np.uint64)
        for _ in range(300):
            z = rng.getrandbits(32)
            assert test_rdq.run32(o, w, consts, z) & cm == tgt[z & cm]
        r = rdp2.solve(o, w, c, 300, 1, tgt)
        out.append((s, r['result'], r.get('it')))
    return out


if __name__ == '__main__':
    print('a', part_a(), 'b', part_b(), 'c', part_c(), flush=True)
    r = part_d()
    print('d', r, 'all SAT:', all(x[1] == 'sat' for x in r), flush=True)
