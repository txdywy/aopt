"""Validation of rdm: (b) realizable multi-SHR positive controls; (c) known c=8 multi-SHR
solutions (rdg SAT records in cg_multi / cs_m2) of rdm-applicable wirings must be SAT
under rdm at c=8 (rdm uses no closure argument, so it is exact at every c)."""
import json, random, sys
import numpy as np
sys.path.insert(0, '.')
import rdm, test_rdq
M = (1 << 32) - 1
rng = random.Random(31337)


def part_b(c=11):
    cm = (1 << c) - 1
    out = []
    o, w = ('mad', '>>', '>>', 'mad'), ((0, 1, 2), (1, 2), (1, 3), (2, 3, 4))   # out = u1*u2 + k
    an = rdm.analyse(o, w)
    for t in range(5):
        if t < 3:
            E = rng.randrange(8, 21)
            s1, s2 = rng.randrange(0, E + 1), rng.randrange(0, E + 1)
            A = ((rng.getrandbits(32 - E) | 1) << E) & M
        else:
            E = None; e = rng.randrange(21, 33)
            s1, s2 = rng.randrange(0, 40), rng.randrange(0, 40)
            A = 0 if e == 32 else ((rng.getrandbits(32 - e) | 1) << e) & M
        consts = [(A, rng.getrandbits(32)), (s1, 0), (s2, 0), (rng.getrandbits(32), 0)]
        tgt = np.array([test_rdq.run32(o, w, consts, z) & cm for z in range(1 << c)], dtype=np.uint64)
        for _ in range(300):
            z = rng.getrandbits(32)
            assert test_rdq.run32(o, w, consts, z) & cm == tgt[z & cm]
        r = rdm._solve(o, w, an, c, 300, 1, E, tgt)
        out.append((E, s1, s2, r['result'], r.get('it')))
    return out


def part_c():
    rows, seen = [], set()
    for fn in ('cg_multi.jsonl', 'cs_m2_s1_31.jsonl'):
        for line in open(fn):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('result') != 'sat' or r.get('c', 8) != 8:
                continue
            o = tuple(r['order']); w = tuple(tuple(x) for x in r['wiring'])
            if (o, w) in seen or not rdm.applicable(o, w):
                continue
            seen.add((o, w)); rows.append((fn, r['i'], o, w))
    bad = 0
    for fn, i, o, w in rows:
        x = rdm.solve_all(o, w, 8, 120, stop_on_fail=False)
        if x['result'] != 'sat':
            bad += 1
            print('PROBLEM', fn, i, o, w, x['result'], flush=True)
    return len(rows), bad


if __name__ == '__main__':
    r = part_b(); print('b', r, 'all SAT:', all(x[3] == 'sat' for x in r), flush=True)
    print('c (records, problems)', part_c(), flush=True)
