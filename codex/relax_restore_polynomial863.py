"""Broaden final-XOR restore candidates to independent polynomial arms.

All three-MAD/one-final-XOR programs with an affine first MAD are contained
in P4(x) XOR Q2(x); square-first programs are contained in P8(x) XOR Q4(x).
Coefficients and the target input XOR key are arbitrary. UNSAT excludes the
covered circuits. A SAT relaxed polynomial is not a four-instruction macro.
"""
from __future__ import annotations

import argparse
from math import factorial
import json
from pathlib import Path
import time

import numpy as np
import z3

from codex.cegis_c1_absorb863 import middle
from codex.search_cross_encoding863 import PRE_C5, necessary_inputs
from codex.search_head_polynomial863 import poly, canonical_poly, interpolate_falling


def solve(degrees=(4, 2), width=11, seconds=60, control=False, family='xor', bitop='^', pin_control=False,
          interpolate=False):
    mask, high = (1 << width)-1, 1 << (width-1)
    solver = z3.SolverFor('QF_BV')
    cs = {}
    specifications = list(zip(('p', 'q'), degrees))
    if family == 'product':
        specifications += [('g', 2), ('r', 2)]
    elif family == 'bypass':
        specifications += [('r', 4)]
    elif family == 'square':
        specifications += [('r', 2)]
    for name, degree in specifications:
        cs[name] = z3.BitVecs(' '.join(f'{name}{j}' for j in range(degree+1)), width)
        for j, c in enumerate(cs[name]):
            f = factorial(j)
            v = (f & -f).bit_length()-1
            if v:
                solver.add(z3.ULT(c, 1 << (width-v)))
    if bitop == '^':
        solver.add(cs['p'][0] & (high | 1) == 0)
    if pin_control:
        assert control
        witness = {}
        for name, variables in cs.items():
            base = {'p': [23, 17], 'q': [7, 41], 'g': [3, 1], 'r': [5, 1, 1]}[name]
            witness[name] = base + [0]*(len(variables)-len(base))
        if bitop == '^':
            for name in ('p', 'q'):
                witness[name] = [-c for c in witness[name]]
                witness[name][0] -= 1
                witness[name][0] += high
        for name, variables in cs.items():
            vals = canonical_poly(witness[name], width)
            solver.add(*(v == n for v, n in zip(variables, vals)))
    ik = z3.BitVec('restore_polynomial_key', width)

    def reference(x):
        if control:
            a, b = 17*x+23, 41*x+7
            value = a ^ b if bitop == '^' else a & b if bitop == '&' else a | b
            if family == 'product':
                value = (x+3)*value + x*x+5
            elif family == 'bypass':
                value = value + x*x+5
            elif family == 'square':
                value = value*value + x*x+5
            return value ^ (PRE_C5 & mask)
        return middle(x, mask)

    def value(x, coeff):
        a, b = poly(x, coeff['p']), poly(x, coeff['q'])
        a, b = a & mask, b & mask
        v = a ^ b if bitop == '^' else a & b if bitop == '&' else a | b
        if family == 'product':
            v = poly(x, coeff['g'])*v + poly(x, coeff['r'])
        elif family == 'bypass':
            v = v + poly(x, coeff['r'])
        elif family == 'square':
            v = v*v + poly(x, coeff['r'])
        return v & mask

    def target(x):
        return reference(x ^ ik) ^ (PRE_C5 & mask)

    if interpolate:
        assert family in ('bypass', 'product', 'square')
        residuals = []
        for p in range(len(cs['r'])):
            x = z3.BitVecVal(p, width)
            a, b = poly(x, cs['p']), poly(x, cs['q'])
            q = a ^ b if bitop == '^' else a & b if bitop == '&' else a | b
            term = q if family == 'bypass' else poly(x, cs['g'])*q if family == 'product' else q*q
            residuals.append(target(x)-term)
        rr, valid = interpolate_falling(residuals, width)
        solver.add(valid, *(a == b for a, b in zip(cs['r'], rr)))

    def add(p):
        solver.add(value(p, cs) == target(z3.BitVecVal(p, width)))

    points = list(dict.fromkeys(list(range(32)) + necessary_inputs(width)))
    for p in points:
        add(p)
    xs = np.arange(1 << width, dtype=np.uint64)
    started = time.monotonic()
    row = dict(degrees=list(degrees), width=width, input_key='arbitrary', output_key=PRE_C5,
               control=control, family=family, bitop=bitop,
               pinned_control=pin_control,
               interpolated_residual=interpolate,
               status='unknown', events=[], new_executable=False)
    while time.monotonic()-started < seconds:
        solver.set(timeout=max(1, int((seconds-time.monotonic()+started)*1000)))
        status = solver.check()
        event = dict(samples=len(points), status=str(status))
        row['events'].append(event)
        if status != z3.sat:
            row['status'] = str(status)
            if status == z3.unknown:
                row['reason'] = solver.reason_unknown()
            break
        m = solver.model()
        cv = {name: [m.eval(c, model_completion=True).as_long() for c in coeff]
              for name, coeff in cs.items()}
        iv = m.eval(ik, model_completion=True).as_long()
        got = value(xs, cv)
        y = xs ^ np.uint64(iv)
        want = reference(y) ^ np.uint64(PRE_C5 & mask)
        want &= np.uint64(mask)
        bad = np.flatnonzero(got != want)
        if not len(bad):
            row.update(status='relaxed_reduced_sat', constants=cv, input_key=iv,
                       verified_inputs=1 << width)
            break
        event['counterexamples'] = [int(xs[i]) for i in bad[:8]]
        for p in event['counterexamples']:
            assert p not in points
            points.append(p)
            add(p)
    row['seconds'] = round(time.monotonic()-started, 3)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--degrees', default='4,2')
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--control', action='store_true')
    ap.add_argument('--family', choices=('xor', 'product', 'bypass', 'square'), default='xor')
    ap.add_argument('--bitop', choices=('^', '&', '|'), default='^')
    ap.add_argument('--pin-control', action='store_true')
    ap.add_argument('--interpolate', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    row = solve(tuple(map(int, args.degrees.split(','))), 11, args.seconds, args.control,
                args.family, args.bitop, args.pin_control, args.interpolate)
    args.output.write_text(json.dumps(row, indent=2)+'\n')
    print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
