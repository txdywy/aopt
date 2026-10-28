"""Necessary 11-bit equations from full32 slices with u_low16 fixed.

For t=K*X+Y (Y may be constant), X/Y in z,u,v,b have fixed low16 on
u=(h<<16)|low. Thus t_high16 = A*X_high16+Y_high16+B, with B constant
on the slice; t_low16 is also constant. Relax both B and t_low16 to
independent unknowns. An affine output normalization yields five unknown
coefficients at width11 and no unknown-times-unknown product. UNSAT is a
necessary full32 exclusion, not a truncation before a right shift.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import z3

from codex.search_joint_live_boundary863 import inputs, wanted, NODE, MASK, C0


def data(hs, low=0):
    u = (hs << 16) | low
    b = ((u-0xE9F8CC1D)*pow(33, -1, 1 << 32)) & MASK
    with np.errstate(over='ignore'):
        values = inputs(b)
    return b, values


def solve(source, addend=None, width=11, seconds=10, control=False):
    assert 1 <= width <= 16 and source in (0, 1)
    assert addend is None or addend in range(4)
    mask = (1 << width)-1
    rng = random.Random(8642842)
    hs = list(dict.fromkeys([*range(64), 511, 512, 513, 1023, 1024, 2047,
                            *(rng.randrange(1 << width) for _ in range(32))]))
    hs = [h for h in hs if h < 1 << width]
    _, vv = data(np.array(hs, dtype=np.uint64))
    aa, bb, kk, rr, ss = z3.BitVecs('slice_A slice_B slice_K slice_R slice_S', width)
    solver = z3.SolverFor('QF_BV'); solver.add(rr & 1 == 1)
    if control:
        assert source == 0 and addend is None
        v0 = inputs((-0xE9F8CC1D*pow(33, -1, 1 << 32)) & MASK)
        e = (9*v0[0]+0xFD7046C5) & MASK
        constants = [9, ((e >> 16)-9*(v0[0] >> 16)) & mask,
                     ((e & 65535) ^ NODE) & mask,
                     pow(4097, -1, 1 << width),
                     (-pow(4097, -1, 1 << width)*C0) & mask]
        solver.add(*(v == n for v, n in zip((aa, bb, kk, rr, ss), constants)))
    with np.errstate(over='ignore'):
        targets = wanted(vv, NODE)
    for i, h in enumerate(hs):
        x = int(vv[source][i] >> 16)
        y = int(vv[addend][i] >> 16) if addend is not None else 0
        solver.add(((aa*x+y+bb) ^ kk) == rr*int(targets[i])+ss)
    start = time.monotonic(); solver.set(timeout=int(seconds*1000)); status = solver.check()
    result = dict(source=source, addend=addend, width=width, fixed_u_low16=0,
                  samples=hs, status=str(status), seconds=round(time.monotonic()-start, 4),
                  full32_sources_shifted_before_reduction=True, unknown_coefficients=5,
                  relaxed_independent_carry_and_low16_mask=True, control=control, new_executable=False)
    if status == z3.sat:
        m = solver.model()
        coeffs = [m.eval(v, model_completion=True).as_long() for v in (aa, bb, kk, rr, ss)]
        h_all = np.arange(1 << width, dtype=np.uint64); _, values = data(h_all)
        a, b, k, r, s = coeffs
        with np.errstate(over='ignore'):
            left = (a*(values[source] >> 16)+(values[addend] >> 16 if addend is not None else 0)+b) & mask
            left ^= k
            right = (r*wanted(values, NODE)+s) & mask
        bad = np.flatnonzero(left != right)
        result.update(coefficients=coeffs, exhaustive_slice_inputs=1 << width,
                      exhaustive_slice_passed=not len(bad),
                      status='relaxed_slice_survivor' if not len(bad) else 'sampled_survivor_only')
    elif status == z3.unknown:
        result['reason'] = solver.reason_unknown()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    control = solve(0, None, control=True)
    assert control['status'] == 'relaxed_slice_survivor' and control['exhaustive_slice_passed']
    rows = [solve(1, None), solve(1, 2), solve(0, 1)]
    assert all(r['status'] == 'unsat' for r in rows), rows
    result = dict(node=NODE, results=rows, control=control, best_executable_cycles=864,
                  new_executable=False,
                  scope='Necessary full32 u_low16=0 slices for t affine in u, t=K*u+v, t=K*z+u, followed by g16 and an arbitrary odd affine tail.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print([(r['source'], r['addend'], r['status']) for r in rows], flush=True)


if __name__ == '__main__':
    main()
