"""Stage-one forms that combine the shifted product with the raw input x.

Extends affine_reencode_family863.py. With a = 4097*x + C0, MAD0 may emit any
odd affine image w = mu*a + nu, and the middle hash absorbs any odd affine
image m*y + c of its input. For each shift s in 1..31 ask whether

    xxor: P = x XOR (w >> s)
    xadd: P = x + k*(w >> s)        (MAD(t,k,x), x - t)
    kx:   P = k*x + (w >> s)        (MAD(x,k,t), any k)
    wk:   P = k*w + (w >> s)        (MAD(w,k,t), any k, including even)

can satisfy m*P + c == g19(a) XOR C1 for every uint32 x. UNSAT on a sample
set excludes the form at full 32-bit width; SAT is re-checked on 200,000
random inputs before being reported.

With --normalized the equivalent form P == m'*T + c' is solved instead
(m' = m^-1 is odd, c' = -m^-1*c); the unknown multiplier then multiplies a
per-sample constant, which is much cheaper to bit-blast.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import random
import sys
import time

import z3

C0, C1 = 0x7ED55D16, 0xC761C23C
MASK = 0xFFFFFFFF


def g19(v):
    return v ^ (v >> 19)


def a_of(x):
    return (4097 * x + C0) & MASK


def sym(fam, s, x, w, k):
    t = z3.LShR(w, s)
    xv = z3.BitVecVal(x, 32)
    if fam == 'xxor':
        return xv ^ t
    if fam == 'xadd':
        return xv + k * t
    if fam == 'kx':
        return k * xv + t
    return k * w + t


def conc(fam, s, x, w, k):
    t = w >> s
    if fam == 'xxor':
        return x ^ t
    if fam == 'xadd':
        return (x + k * t) & MASK
    if fam == 'kx':
        return (k * x + t) & MASK
    return (k * w + t) & MASK


TARGETS = {
    'g19c1': lambda x: g19(a_of(x)) ^ C1,
    'ctl_xxor': lambda x: x ^ (a_of(x) >> 19),
    'ctl_xadd': lambda x: (x + 3 * (a_of(x) >> 19)) & MASK,
    'ctl_kx': lambda x: (5 * x + (a_of(x) >> 19)) & MASK,
    'ctl_wk': lambda x: (6 * a_of(x) + (a_of(x) >> 19)) & MASK,
}


def query(args):
    fam, s, seconds, tname, normalized = args
    target = TARGETS[tname]
    mu, nu, m, c, k = z3.BitVecs('mu nu m c k', 32)
    sol = z3.Then('simplify', 'solve-eqs', 'bit-blast', 'sat').solver()
    sol.add(z3.Extract(0, 0, mu) == 1, z3.Extract(0, 0, m) == 1)
    rng = random.Random(1000 * s + len(fam))
    pts = [0, 1, 2, 1 << 12, 1 << 19, 1 << 31, MASK] + [rng.getrandbits(32) for _ in range(9)]

    def add(x):
        w = mu * z3.BitVecVal(a_of(x), 32) + nu
        if normalized:
            sol.add(sym(fam, s, x, w, k) == m * z3.BitVecVal(target(x), 32) + c)
        else:
            sol.add(m * sym(fam, s, x, w, k) + c == z3.BitVecVal(target(x), 32))

    for x in pts:
        add(x)
    t0 = time.monotonic()
    while True:
        left = seconds - (time.monotonic() - t0)
        if left <= 0:
            return dict(fam=fam, s=s, target=tname, normalized=normalized, result='timeout', samples=len(pts))
        sol.set(timeout=max(1, int(left * 1000)))
        r = sol.check()
        if r != z3.sat:
            return dict(fam=fam, s=s, target=tname, normalized=normalized, result=str(r), samples=len(pts),
                        seconds=round(time.monotonic() - t0, 2))
        md = sol.model()
        mu_, nu_, m_, c_, k_ = (md.eval(v, model_completion=True).as_long() for v in (mu, nu, m, c, k))
        chk = random.Random(99)
        bad = None
        for _ in range(200000):
            x = chk.getrandbits(32)
            w = (mu_ * a_of(x) + nu_) & MASK
            lhs, rhs = conc(fam, s, x, w, k_), target(x)
            if (lhs != (m_ * rhs + c_) & MASK) if normalized else ((m_ * lhs + c_) & MASK != rhs):
                bad = x
                break
        if bad is None:
            return dict(fam=fam, s=s, target=tname, normalized=normalized, result='sat', model=[mu_, nu_, m_, c_, k_])
        pts.append(bad)
        add(bad)


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 300
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    normalized = '--normalized' in sys.argv
    only = [a for a in sys.argv[3:] if a.startswith('--only=')]
    fams = ('xxor', 'xadd', 'kx', 'wk')
    controls = [(f, 19, seconds, 'ctl_' + f, normalized) for f in fams]
    jobs = [(f, s, seconds, 'g19c1', normalized) for f in fams for s in range(1, 32)]
    if only:
        keep = {tuple(item.split(':')) for item in only[0][7:].split(',')}
        jobs = [j for j in jobs if (j[0], str(j[1])) in keep]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(query, controls + jobs):
            rows.append(row)
            print(json.dumps(row), flush=True)
    body = rows[len(controls):]
    print(json.dumps(dict(summary={r: sum(1 for x in body if x['result'] == r)
                                   for r in ('unsat', 'sat', 'timeout', 'unknown')},
                          controls=rows[:len(controls)])))


if __name__ == '__main__':
    main()
