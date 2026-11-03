"""Exact normalized families for the two final restore-search wirings.

xor_affine: u XOR (C*u+D), u=(A*x+B) XOR K.
sum: C*x+((A*(x XOR K)+B) XOR L), up to input-key normalization.
For sum, normalizing the input introduces an offset H. Its exact orbit is
H in {0,C,C*high,C*(high+1)}, not an unrestricted extra operation.
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


def body(x, c, family, mask):
    if family == 'xor_affine':
        u = ((c['A']*x+c['B']) & mask) ^ c['K']
        return u ^ ((c['C']*u+c['D']) & mask)
    v = ((c['A']*(x ^ c['K'])+c['B']) & mask) ^ c['L']
    return (c['C']*x+v+c.get('H', 0)) & mask


def input_transform(c, key, family, width, complement=False, high_flip=False):
    c = dict(c)
    mask, high = (1 << width)-1, 1 << (width-1)
    if complement:
        c['B'] -= c['A']
        c['A'] = -c['A']
        if family == 'sum':
            c['H'] -= c['C']
            c['C'] = -c['C']
        key ^= mask
    if high_flip:
        c['B'] += c['A']*high
        if family == 'sum':
            c['H'] += c['C']*high
        key ^= high
    return {k: v & mask for k, v in c.items()}, key


def normalize(c, key, family, width):
    mask, high = (1 << width)-1, 1 << (width-1)
    c = dict(c)
    c.setdefault('H', 0)
    c, key = input_transform(c, key, family, width, complement=bool(key & 1))
    c, key = input_transform(c, key, family, width, high_flip=bool(key & high))
    if c['K'] & 1:
        c['K'] ^= mask
        if family == 'xor_affine':
            c['D'] = c['C']-c['D']-1
        else:
            c['B'] -= c['A']
            c['A'] = -c['A']
    if c['K'] & high:
        c['K'] ^= high
        if family == 'xor_affine':
            c['D'] += high
        else:
            c['B'] -= c['A']*high
    if family == 'xor_affine':
        if c['B'] & 1:
            c['A'], c['B'] = -c['A'], -c['B']-1
            c['D'] = c['C']-c['D']-1
        if c['B'] & high:
            c['B'] += high
            c['D'] += high
    else:
        if c['L'] & 1:
            c['L'] ^= mask
            c['A'], c['B'] = -c['A'], -c['B']-1
        if c['L'] & high:
            c['L'] ^= high
            c['B'] += high
    return {k: v & mask for k, v in c.items()}, key & mask


def controls():
    rng = random.Random(8642804)
    orbit_checks = 0
    for family in ('xor_affine', 'sum'):
        for _ in range(80):
            c = {k: rng.randrange(64) for k in ('A', 'B', 'C', 'D', 'K', 'L')}
            c['H'] = 0
            if family == 'xor_affine':
                c['A'] |= 1
                c['C'] &= ~1
            else:
                c['C'] = (c['C'] & ~1) | ((c['A'] ^ 1) & 1)
            key = rng.randrange(64)
            n, nk = normalize(c, key, family, 6)
            assert not (n['K'] | nk) & 33
            if family == 'xor_affine':
                assert n['A'] & 1 and not n['C'] & 1 and not n['B'] & 33
            else:
                assert not n['L'] & 33
                assert n['H'] in {(n['C']*m)&63 for m in (0, 1, 32, 33)}
                for e in (False, True):
                    for h in (False, True):
                        if n['H'] != (n['C']*(int(e)+32*int(h))) & 63:
                            continue
                        denorm, dk = input_transform(n, nk, family, 6, e, h)
                        assert denorm['H'] == 0
                        assert all(body(x ^ dk, denorm, family, 63) == body(x ^ key, c, family, 63) for x in range(64))
                        orbit_checks += 1
            assert all(body(x ^ nk, n, family, 63) == body(x ^ key, c, family, 63) for x in range(64))
    return dict(families=2, programs_per_family=80, inputs_per_program=64,
                exposed_function_preserved=True, sum_offset_orbit_controls=orbit_checks)


def solve(family, width, seconds, control=False):
    mask, high = (1 << width)-1, 1 << (width-1)
    names = ('A', 'B', 'C', 'K', 'I') if family == 'xor_affine' else ('A', 'B', 'C', 'K', 'L', 'I')
    params = dict(zip(names, z3.BitVecs(' '.join(names), width)))
    a, b, c, k, ik = [params[n] for n in ('A', 'B', 'C', 'K', 'I')]
    s = z3.SolverFor('QF_BV')
    s.add(k & (high | 1) == 0, ik & (high | 1) == 0)
    if family == 'xor_affine':
        s.add(a & 1 == 1, c & 1 == 0, b & (high | 1) == 0)
    else:
        s.add((a+c) & 1 == 1, params['L'] & (high | 1) == 0)
    witness = dict(A=3, B=24, C=40, D=7, K=42, L=6, H=0)

    def reference(x):
        return body(x, witness, family, mask) ^ (PRE_C5 & mask) if control else middle(x, mask)

    target = lambda x: reference(x ^ ik) ^ (PRE_C5 & mask)
    zero = target(z3.BitVecVal(0, width))
    if family == 'xor_affine':
        u = b ^ k
        params['D'] = (u ^ zero)-c*u
    else:
        params['H'] = zero-((a*k+b) ^ params['L'])
        s.add(z3.Or(*(params['H'] == c*m for m in (0, 1, high, high+1))))
    if control:
        normalized, _ = normalize(witness, 0, family, width)
        pins = ('A','B','C','K') if family == 'xor_affine' else ('A','B','C','K','L')
        s.add(*(params[n] == normalized[n] for n in pins))

    points = list(dict.fromkeys(list(range(32))+necessary_inputs(width)))
    def add(x):
        xv = z3.BitVecVal(x, width)
        s.add(body(xv, params, family, mask) == target(xv))
    for p in points:
        add(p)
    xs = np.arange(1 << width, dtype=np.uint64)
    start = time.monotonic()
    row = dict(family=family, width=width, control=control, status='unknown', events=[],
               target_output_key=PRE_C5, normalized_input_key=True,
               sum_offset='exact four-case orbit' if family == 'sum' else None, new_executable=False)
    while time.monotonic()-start < seconds:
        s.set(timeout=max(1, int((seconds-time.monotonic()+start)*1000)))
        status = s.check()
        row['events'].append(dict(samples=len(points), status=str(status)))
        if status != z3.sat:
            row['status'] = str(status)
            if status == z3.unknown:
                row['reason'] = s.reason_unknown()
            break
        m = s.model()
        cv = {n: m.eval(v, model_completion=True).as_long() for n, v in params.items()}
        got = body(xs, cv, family, mask)
        want = reference(xs ^ np.uint64(cv['I'])) ^ np.uint64(PRE_C5 & mask)
        bad = np.flatnonzero(got != want)
        if not len(bad):
            row.update(status='reduced_sat', parameters=cv, verified_inputs=len(xs))
            break
        for i in bad[:8]:
            p = int(xs[i])
            assert p not in points
            points.append(p)
            add(p)
    row['seconds'] = round(time.monotonic()-start, 3)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--family', choices=('xor_affine','sum'), required=True)
    ap.add_argument('--seconds', type=float, default=180)
    ap.add_argument('--control', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    result = dict(normalization_controls=controls(), query=solve(args.family, 11, args.seconds, args.control))
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
