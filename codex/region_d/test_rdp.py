"""Validation of rdp. (a) fiber lemma for random quadratics; (b) every verified c=8
SAT record whose shifted value is a polynomial of degree <= 2 must be SAT under rdp at
c=8; (c) agreement with rda (UNSAT) on a few affine wirings at c=11."""
import glob, json, random, sys
sys.path.insert(0, '.')
import rdp, rda
M = (1 << 32) - 1
rng = random.Random(123)


def part_a(n=300, c=11):
    for _ in range(n):
        a2, a1, a0 = rng.getrandbits(32), rng.getrandbits(32), rng.getrandbits(32)
        z0 = rng.getrandbits(c)
        f = lambda z: (a2 * z * z + a1 * z + a0) & M
        V0 = f(z0); D = (2 * a2 * z0 + a1) & M
        for _ in range(4):
            y = rng.getrandbits(32 - c)
            assert f(z0 + (y << c)) == (V0 + ((y * D) << c) + ((y * y * a2) << (2 * c))) & M
        Dl = D & ((1 << c) - 1)
        if Dl == 0:
            continue
        d = (Dl & -Dl).bit_length() - 1
        nb = 32 - c - d
        if nb > 13:
            # force a small case: choose a1 so that d is large enough
            continue
        seen = set()
        for y in range(1 << nb):
            v = f(z0 + (y << c))
            assert v & ((1 << (c + d)) - 1) == V0 & ((1 << (c + d)) - 1)
            seen.add(v >> (c + d))
        assert len(seen) == 1 << nb, (a2, a1, a0, z0, d, len(seen))
    return n


def part_a2(n=200, c=11):
    """Same, with a1 chosen so that d is in [8, 10] (small fibers to enumerate)."""
    cnt = 0
    while cnt < n:
        a2, a0 = rng.getrandbits(32), rng.getrandbits(32)
        z0 = rng.getrandbits(c)
        d = rng.randrange(8, c)
        # pick a1 with v2(2 a2 z0 + a1 mod 2^c) == d
        a1 = ((1 << d) | (rng.getrandbits(32) << (d + 1))) - 2 * a2 * z0
        a1 &= M
        f = lambda z: (a2 * z * z + a1 * z + a0) & M
        D = (2 * a2 * z0 + a1) & M
        Dl = D & ((1 << c) - 1)
        assert (Dl & -Dl).bit_length() - 1 == d
        nb = 32 - c - d
        V0 = f(z0)
        seen = set()
        for y in range(1 << nb):
            v = f(z0 + (y << c))
            assert v & ((1 << (c + d)) - 1) == V0 & ((1 << (c + d)) - 1)
            seen.add(v >> (c + d))
        assert len(seen) == 1 << nb
        cnt += 1
    return cnt


def part_b():
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
            if (o, w) in seen or not rdp.applicable(o, w):
                continue
            seen.add((o, w))
            rows.append((fn, r['i'], o, w))
    bad = 0
    for fn, i, o, w in rows:
        x = rdp.solve(o, w, 8, 240)
        if x['result'] != 'sat':
            bad += 1
            print('PROBLEM', fn, i, o, w, x['result'], flush=True)
    return len(rows), bad


def part_c():
    out = []
    for o, w in [(('mad', '>>', 'mad', 'mad'), ((0, 1, 2), (1, 2), (0, 1, 2), (0, 0, 3))),
                 (('mad', 'mad', '>>', 'mad'), ((0, 1, 2), (0, 2, 1), (2, 3), (0, 3, 2)))]:
        out.append((rda.solve(o, w, 11, 300)['result'], rdp.solve(o, w, 11, 300)['result']))
    return out


if __name__ == '__main__':
    print('a', part_a(), 'a2', part_a2(), flush=True)
    print('b (records, problems)', part_b(), flush=True)
    print('c (rda, rdp)', part_c(), flush=True)
