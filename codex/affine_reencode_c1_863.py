"""Can affine re-encodings on both sides of g19 absorb the C1 XOR?

MAD0 may emit any affine image of a = 4097*x + C0, and the two MADs that start
the middle hash absorb any odd affine image of their input. If odd mu, m and
any nu, c satisfied

    m * g19(mu*a + nu) + c == g19(a) XOR C1      for every uint32 a,

stage one would need only its shift and XOR, saving one op per element-round.
This CEGIS solves for (mu, nu, m, c) with full 32-bit semantics.
"""
from __future__ import annotations

import json
import random
import sys
import time

import z3

C1 = 0xC761C23C
MASK = 0xFFFFFFFF


def g19(v):
    return v ^ (v >> 19)


def check(mu, nu, m, c, samples=200000, seed=7):
    rng = random.Random(seed)
    for _ in range(samples):
        a = rng.getrandbits(32)
        if (m * g19((mu * a + nu) & MASK) + c) & MASK != g19(a) ^ C1:
            return a
    return None


def solve(seconds=600):
    mu, nu, m, c = z3.BitVecs('mu nu m c', 32)
    s = z3.Then('simplify', 'solve-eqs', 'bit-blast', 'sat').solver()
    s.add(z3.Extract(0, 0, mu) == 1, z3.Extract(0, 0, m) == 1)
    rng = random.Random(1)
    pts = [0, 1, 2, 3, 1 << 12, 1 << 19, 1 << 31, MASK] + [rng.getrandbits(32) for _ in range(8)]

    def add(a):
        w = mu * z3.BitVecVal(a, 32) + nu
        s.add(m * (w ^ z3.LShR(w, 19)) + c == z3.BitVecVal(g19(a) ^ C1, 32))

    for a in pts:
        add(a)
    t0 = time.monotonic()
    rounds = 0
    while True:
        s.set(timeout=max(1, int((seconds - (time.monotonic() - t0)) * 1000)))
        r = s.check()
        rounds += 1
        if r != z3.sat:
            return dict(result=str(r), rounds=rounds, samples=len(pts),
                        seconds=round(time.monotonic() - t0, 2))
        md = s.model()
        vals = [md.eval(v, model_completion=True).as_long() for v in (mu, nu, m, c)]
        bad = check(*vals)
        if bad is None:
            return dict(result='sat', model=vals, rounds=rounds)
        pts.append(bad)
        add(bad)


def control():
    """Positive control: with C1 replaced by g19(2**31), mu=1, nu=2**31 works."""
    global C1
    saved, C1 = C1, g19(1 << 31)
    try:
        return solve(120)
    finally:
        C1 = saved


if __name__ == '__main__':
    out = dict(control=control(), target=solve(float(sys.argv[1]) if len(sys.argv) > 1 else 600))
    print(json.dumps(out))
