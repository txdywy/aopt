"""Search G*(P1(x) XOR Q2(x))+H with a free input XOR encoding.

This is a necessary relaxation of several three-MAD/one-XOR circuits. Both
polynomials are closed under x -> ~x and x -> x+2**(w-1), so the free target
key's lowest and highest bits may be cleared. This normalization applies
to this polynomial family, not to arbitrary concrete SSA wirings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import z3

from codex.cegis_c1_absorb863 import middle
from codex.search_cross_encoding863 import PRE_C5, necessary_inputs
from codex.search_head_polynomial863 import poly, canonical_poly, interpolate_falling


def evaluate(x, coefficients, mask):
    p, q, g, h = coefficients
    return (g*((poly(x, p) & mask) ^ (poly(x, q) & mask))+h) & mask


def normalize(coefficients, key, width):
    mask, high = (1 << width)-1, 1 << (width-1)
    p, q, g, h = coefficients
    delta = mask if key & 1 else 0
    if (key ^ delta) & high:
        delta ^= high
    transformed = []
    for coeff in (p, q):
        values = [poly(x ^ delta, coeff) & mask for x in range(len(coeff))]
        out, valid = interpolate_falling(values, width)
        assert valid
        transformed.append(out)
    p, q = transformed
    if p[0] & 1:
        p, q = [-v for v in p], [-v for v in q]
        p[0] -= 1
        q[0] -= 1
    if p[0] & high:
        p[0] += high
        q[0] += high
    return [canonical_poly(p, width), canonical_poly(q, width), g & mask, h & mask], key ^ delta


def check_normalization():
    rng = random.Random(86428)
    width, mask = 6, 63
    for _ in range(80):
        c = [[rng.randrange(64) for _ in range(2)],
             [rng.randrange(64) for _ in range(3)], rng.randrange(64), rng.randrange(64)]
        key = rng.randrange(64)
        n, nk = normalize(c, key, width)
        assert nk & 33 == 0 and n[0][0] & 33 == 0
        for x in range(64):
            assert evaluate(x ^ key, c, mask) == evaluate(x ^ nk, n, mask)
    return dict(programs=80, inputs_per_program=64, exposed_function_preserved=True)


def solve(width=11, seconds=60, control=False):
    mask, high = (1 << width)-1, 1 << (width-1)
    s = z3.SolverFor('QF_BV')
    p = z3.BitVecs('sp0 sp1', width)
    q = z3.BitVecs('sq0 sq1 sq2', width)
    g, h, key = z3.BitVecs('sg sh sk', width)
    s.add(p[0] & (high | 1) == 0, key & (high | 1) == 0,
          g & 1 == 1, (p[1]+q[1]) & 1 == 1, q[2] & 1 == 0,
          z3.ULT(q[2], high))
    witness = [[3, 2], [7, 5, 2], 9, 29]
    if control:
        normalized, ik = normalize(witness, 0, width)
        for variables, values in ((p, normalized[0]), (q, normalized[1])):
            s.add(*(a == b for a, b in zip(variables, values)))
        s.add(g == normalized[2], h == normalized[3])

    def reference(x):
        return evaluate(x, witness, mask) ^ (PRE_C5 & mask) if control else middle(x, mask)

    def target(x):
        return reference(x ^ key) ^ (PRE_C5 & mask)

    def add(x):
        s.add(evaluate(x, [p, q, g, h], mask) == target(z3.BitVecVal(x, width)))

    points = list(dict.fromkeys(list(range(32)) + necessary_inputs(width)))
    for x in points:
        add(x)
    xs = np.arange(1 << width, dtype=np.uint64)
    start = time.monotonic()
    row = dict(width=width, family='G*(P1 XOR Q2)+H', output_key=PRE_C5,
               arbitrary_input_key=True, normalized_input_key_bits=[0, width-1],
               control=control, pinned_control=control, status='unknown', events=[],
               new_executable=False)
    while time.monotonic()-start < seconds:
        s.set(timeout=max(1, int((seconds-time.monotonic()+start)*1000)))
        status = s.check()
        event = dict(samples=len(points), status=str(status))
        row['events'].append(event)
        if status != z3.sat:
            row['status'] = str(status)
            if status == z3.unknown:
                row['reason'] = s.reason_unknown()
            break
        model = s.model()
        c = [[model.eval(v, model_completion=True).as_long() for v in vv] for vv in (p, q)]
        c += [model.eval(v, model_completion=True).as_long() for v in (g, h)]
        ik = model.eval(key, model_completion=True).as_long()
        got = evaluate(xs, c, mask)
        want = reference(xs ^ np.uint64(ik)) ^ np.uint64(PRE_C5 & mask)
        bad = np.flatnonzero(got != want)
        if not len(bad):
            row.update(status='relaxed_reduced_sat', coefficients=c, input_key=ik,
                       verified_inputs=1 << width)
            break
        for i in bad[:8]:
            x = int(xs[i])
            assert x not in points
            points.append(x)
            add(x)
    row['seconds'] = round(time.monotonic()-start, 3)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--control', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    result = dict(normalization_control=check_normalization(), query=solve(11, args.seconds, args.control))
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
