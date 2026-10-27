"""Two-op shift families for stage one under affine re-encodings at both ends.

For a = 4097*x + C0 (bijective in x), MAD0 can emit any odd affine image
w = mu*a + nu, and the middle hash absorbs any odd affine image of its input.
For each shift s in 1..31 and each two-op family

    xor:  P(w) = w XOR (w >> s)
    add:  P(w) = w + k*(w >> s)      (covers MAD(t,k,w), MAD(w,k,t), w-t)

ask whether odd mu, m and any nu, c (and k) satisfy
    m * P(mu*a + nu) + c == g19(a) XOR C1   for every uint32 a.
UNSAT for a family/shift excludes it at full 32-bit width; SAT is checked on
200,000 random inputs before being reported.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import random
import sys
import time

import z3

C1 = 0xC761C23C
MASK = 0xFFFFFFFF


def g19(v):
    return v ^ (v >> 19)


def program(fam, s, w, k):
    t = z3.LShR(w, s)
    return w ^ t if fam == 'xor' else w + k * t


def concrete(fam, s, w, k):
    t = w >> s
    return (w ^ t) if fam == 'xor' else (w + k * t) & MASK


TARGETS = {
    'g19c1': lambda a: g19(a) ^ C1,
    'g19top': lambda a: g19(a) ^ g19(1 << 31),   # positive control for xor
    'add3': lambda a: (a + 3 * (a >> 19)) & MASK,  # positive control for add
}


def query(args):
    fam, s, seconds, tname = args
    target = TARGETS[tname]
    mu, nu, m, c, k = z3.BitVecs('mu nu m c k', 32)
    sol = z3.Then('simplify', 'solve-eqs', 'bit-blast', 'sat').solver()
    sol.add(z3.Extract(0, 0, mu) == 1, z3.Extract(0, 0, m) == 1)
    rng = random.Random(s * 7 + (fam == 'add'))
    pts = [0, 1, 2, 1 << 12, 1 << 19, 1 << 31, MASK] + [rng.getrandbits(32) for _ in range(9)]

    def add(a):
        w = mu * z3.BitVecVal(a, 32) + nu
        sol.add(m * program(fam, s, w, k) + c == z3.BitVecVal(target(a), 32))

    for a in pts:
        add(a)
    t0 = time.monotonic()
    while True:
        left = seconds - (time.monotonic() - t0)
        if left <= 0:
            return dict(fam=fam, s=s, target=tname, result='timeout', samples=len(pts))
        sol.set(timeout=max(1, int(left * 1000)))
        r = sol.check()
        if r != z3.sat:
            return dict(fam=fam, s=s, target=tname, result=str(r), samples=len(pts),
                        seconds=round(time.monotonic() - t0, 2))
        md = sol.model()
        vals = [md.eval(v, model_completion=True).as_long() for v in (mu, nu, m, c, k)]
        bad = None
        chk = random.Random(99)
        for _ in range(200000):
            a = chk.getrandbits(32)
            if (vals[2] * concrete(fam, s, (vals[0] * a + vals[1]) & MASK, vals[4]) + vals[3]) & MASK != target(a):
                bad = a
                break
        if bad is None:
            return dict(fam=fam, s=s, target=tname, result='sat', model=vals)
        pts.append(bad)
        add(bad)


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 300
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    jobs = [(fam, s, seconds, 'g19c1') for fam in ('xor', 'add') for s in range(1, 32)]
    controls = [('xor', 19, seconds, 'g19top'), ('add', 19, seconds, 'add3')]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(query, controls + jobs):
            rows.append(row)
            print(json.dumps(row), flush=True)
    print(json.dumps(dict(summary={r: sum(1 for x in rows[2:] if x['result'] == r)
                                   for r in ('unsat', 'sat', 'timeout', 'unknown')},
                          controls=rows[:2])))


if __name__ == '__main__':
    main()
