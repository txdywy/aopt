"""Validation of rdvq.
(1) square expansion identity on random 32-bit values (T, H, E).
(2) positive controls: realizable u = t0 >> (t0*t0 + k) programs; the matching part
    (E fixed, or case B for e >= 21) must be SAT."""
import random, sys
import numpy as np
sys.path.insert(0, '.')
import rdvq, test_rdq
M = (1 << 32) - 1
c = 11; cm = (1 << c) - 1
rng = random.Random(99)


def part1(n=20000):
    for _ in range(n):
        E = rng.randrange(0, 21)
        Blo = rng.getrandbits(E) if E else 0
        m = rng.getrandbits(c)
        H = rng.getrandbits(32 - c - E)
        T = Blo + (m << E)
        w = (T + (H << (c + E))) & M
        sq = (Blo * Blo + ((Blo * m) << (E + 1)) + ((m * m) << (2 * E)) + ((T * H) << (c + E + 1)) + ((H * H) << (2 * c + 2 * E))) & M
        assert sq == (w * w) & M
    return n


def part2():
    o, w = ('mad', 'mad', '>>', '^'), ((0, 1, 2), (1, 1, 2), (1, 2), (3, 4))
    an = rdvq.analyse(o, w)
    res = []
    for t in range(6):
        if t < 4:
            E = rng.randrange(10, 21)
            A = ((rng.getrandbits(32 - E) | 1) << E) & M
            B = (rng.getrandbits(32 - E) << E) & M          # B_lo = 0
            k = rng.randrange(0, E + 1)
        else:
            E = None
            e = rng.randrange(21, 33)
            A = 0 if e == 32 else ((rng.getrandbits(32 - e) | 1) << e) & M
            B = rng.getrandbits(32)
            k = rng.getrandbits(32)
        consts = [(A, B), (0, k), (0, 0), (rng.getrandbits(32), 0)]
        tgt = np.array([test_rdq.run32(o, w, consts, z) & cm for z in range(1 << c)], dtype=np.uint64)
        ok = True
        for _ in range(400):
            z = rng.getrandbits(32)
            ok &= test_rdq.run32(o, w, consts, z) & cm == tgt[z & cm]
        assert ok, ('control not realizable', E, A, B, k)
        r = rdvq._solve(o, w, an, c, 300, 1, E, tgt)
        res.append((E, k if E is not None else 'k', r['result'], r.get('it')))
    return res


if __name__ == '__main__':
    print('part1', part1(), flush=True)
    r = part2()
    print('part2', r, 'all SAT:', all(x[2] == 'sat' for x in r))
