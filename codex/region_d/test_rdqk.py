"""Validation of rdqk: (a) expansion identity incl. + x; (b) realizable positive controls;
(c) every known c=8 solution (band SAT records) of an rdqk-applicable wiring must be SAT
under rdqk at c=8 (rdqk is exact at every c)."""
import glob, json, random, sys
import numpy as np
sys.path.insert(0, '.')
import rdqk, test_rdq
from rd2 import K1, K2, C4, D
M = (1 << 32) - 1
rng = random.Random(4242)


def part_a(n=20000, c=11):
    for _ in range(n):
        E = rng.randrange(0, 32 - c)
        Blo = rng.getrandbits(E) if E else 0
        m = rng.getrandbits(c); H = rng.getrandbits(32 - c - E); x = rng.getrandbits(32)
        T = Blo + (m << E)
        w = (T + (H << (c + E))) & M
        sq = (Blo * Blo + ((Blo * m) << (E + 1)) + ((m * m) << (2 * E)) + ((T * H) << (c + E + 1)) + ((H * H) << (2 * c + 2 * E))) & M
        assert (sq + x) & M == (w * w + x) & M
    return n


def part_b(c=11):
    cm = (1 << c) - 1
    out = []
    o, w = ('mad', 'mad', '>>', '^'), ((0, 1, 2), (1, 1, 2), (2, 3), (3, 4))
    an = rdqk.analyse(o, w)
    for t in range(5):
        if t < 3:
            E = rng.randrange(16, 21); s = rng.randrange(0, 11)
            A = ((rng.getrandbits(32 - E) | 1) << E) & M
        else:
            E = None; e = rng.randrange(21, 33); s = rng.randrange(0, 33)
            A = 0 if e == 32 else ((rng.getrandbits(32 - e) | 1) << e) & M
        consts = [(A, rng.getrandbits(32)), (0, rng.getrandbits(32)), (s, 0), (rng.getrandbits(32), 0)]
        tgt = np.array([test_rdq.run32(o, w, consts, z) & cm for z in range(1 << c)], dtype=np.uint64)
        for _ in range(300):
            z = rng.getrandbits(32)
            assert test_rdq.run32(o, w, consts, z) & cm == tgt[z & cm]
        r = rdqk._solve(o, w, an, c, 300, 1, E, tgt)
        out.append((E, s, r['result'], r.get('it')))
    return out


def part_c():
    rows, seen = [], set()
    for fn in sorted(glob.glob('cs_p*_s*.jsonl')):
        for line in open(fn):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('result') != 'sat' or r.get('c') != 8:
                continue
            o = tuple(r['order']); w = tuple(tuple(x) for x in r['wiring'])
            if (o, w) in seen or not rdqk.applicable(o, w):
                continue
            seen.add((o, w)); rows.append((fn, r['i'], o, w))
    bad = 0
    for fn, i, o, w in rows:
        x = rdqk.solve_all(o, w, 8, 120)
        if x['result'] != 'sat':
            bad += 1
            print('PROBLEM', fn, i, o, w, x['result'], x['parts'][-3:], flush=True)
    return len(rows), bad


if __name__ == '__main__':
    print('a', part_a(), flush=True)
    r = part_b(); print('b', r, 'all SAT:', all(x[2] == 'sat' for x in r), flush=True)
    print('c (records, problems)', part_c(), flush=True)
