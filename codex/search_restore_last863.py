"""Normalized exact families for four of the final restore-search wirings.

mixed: (A*x+B) XOR (C*(x XOR K)+D)
quad:  C*((A*x+B) XOR K)**2 + D*((A*x+B) XOR K)+E
The target is F(x XOR I) XOR PRE_C5. These are actual four-instruction
families, not relaxed polynomials. Low-word UNSAT excludes full32; reduced
SAT still requires lifting and a full32 proof before integration.
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
    a = (c['A']*x+c['B']) & mask
    if family == 'mixed':
        return a ^ ((c['C']*(x ^ c['K'])+c['D']) & mask)
    u = a ^ c['K']
    return (c['C']*u*u+c['D']*u+c['E']) & mask


def normalize(c, key, family, width):
    c = dict(c)
    mask, high = (1 << width)-1, 1 << (width-1)
    if family == 'mixed' and not c['A'] & 1:
        assert c['C'] & 1
        c['A'], c['C'] = c['C'], c['A']
        c['B'], c['D'] = c['D'], c['B']
        key ^= c['K']
    if key & 1:
        c['B'] -= c['A']
        c['A'] = -c['A']
        if family == 'mixed':
            c['D'] -= c['C']
            c['C'] = -c['C']
        key ^= mask
    if key & high:
        c['B'] += c['A']*high
        if family == 'mixed':
            c['D'] += c['C']*high
        key ^= high
    if c['K'] & 1:
        c['K'] ^= mask
        if family == 'mixed':
            c['D'] -= c['C']
            c['C'] = -c['C']
        else:
            c['E'] += c['C']-c['D']
            c['D'] = 2*c['C']-c['D']
    if c['K'] & high:
        c['K'] ^= high
        if family == 'mixed':
            c['D'] -= c['C']*high
        else:
            c['E'] += c['D']*high
    if c['B'] & 1:
        c['A'], c['B'] = -c['A'], -c['B']-1
        if family == 'mixed':
            c['C'], c['D'] = -c['C'], -c['D']-1
        else:
            c['E'] += c['C']-c['D']
            c['D'] = 2*c['C']-c['D']
    if c['B'] & high:
        c['B'] += high
        if family == 'mixed':
            c['D'] += high
        else:
            c['E'] += c['D']*high
    if family == 'quad' and c['C'] & high:
        c['C'] ^= high
        c['D'] ^= high
    return {k: v & mask for k, v in c.items()}, key & mask


def controls():
    rng = random.Random(8642802)
    for family in ('mixed', 'quad'):
        for _ in range(80):
            c = {k: rng.randrange(64) for k in ('A', 'B', 'C', 'D', 'E', 'K')}
            if family == 'mixed':
                c['C'] = (c['C'] & ~1) | ((c['A'] ^ 1) & 1)
            else:
                c['A'] |= 1
                c['C'] &= ~1
                c['D'] |= 1
            key = rng.randrange(64)
            n, nk = normalize(c, key, family, 6)
            assert n['A'] & 1 and not n['C'] & 1
            assert not (n['B'] | n['K'] | nk) & 33
            if family == 'quad':
                assert n['D'] & 1 and n['C'] < 32
            for x in range(64):
                assert body(x ^ key, c, family, 63) == body(x ^ nk, n, family, 63)
    return dict(families=2, programs_per_family=80, inputs_per_program=64,
                exposed_function_preserved=True)


def solve(family, width, seconds, control=False):
    mask, high = (1 << width)-1, 1 << (width-1)
    vs = dict(zip(('A', 'B', 'C', 'D', 'K', 'I'), z3.BitVecs('A B C D K I', width)))
    a, b, c, d, k, ik = [vs[n] for n in ('A', 'B', 'C', 'D', 'K', 'I')]
    solver = z3.SolverFor('QF_BV')
    solver.add(a & 1 == 1, c & 1 == 0,
               b & (high | 1) == 0, k & (high | 1) == 0, ik & (high | 1) == 0)
    if family == 'quad':
        solver.add(d & 1 == 1, z3.ULT(c, high))
    witness = dict(A=17, B=24, C=40, D=7, K=42, E=0) if family == 'mixed' else dict(A=3, B=24, C=2, D=5, E=7, K=42)

    def reference(x):
        return body(x, witness, family, mask) ^ (PRE_C5 & mask) if control else middle(x, mask)

    target = lambda x: reference(x ^ ik) ^ (PRE_C5 & mask)
    params = dict(vs)
    if family == 'mixed':
        params['D'] = (b ^ target(z3.BitVecVal(0, width)))-c*k
    else:
        u = b ^ k
        params['E'] = target(z3.BitVecVal(0, width))-c*u*u-d*u
    if control:
        normalized, _ = normalize(witness, 0, family, width)
        names = ('A', 'B', 'C', 'K') if family == 'mixed' else ('A', 'B', 'C', 'D', 'K')
        solver.add(*(vs[n] == normalized[n] for n in names))

    points = list(dict.fromkeys(list(range(32)) + necessary_inputs(min(width, 16))))
    points += [] if width < 32 else [1 << i for i in range(32)]
    def add(x):
        solver.add(body(z3.BitVecVal(x, width), params, family, mask) == target(z3.BitVecVal(x, width)))
    for x in points:
        add(x)
    xs = np.arange(1 << width, dtype=np.uint64) if width <= 16 else np.array(points+[random.Random(863+i).getrandbits(32) for i in range(2048)], dtype=np.uint64)
    start = time.monotonic()
    row = dict(family=family, width=width, control=control, status='unknown', events=[], new_executable=False)
    while time.monotonic()-start < seconds:
        solver.set(timeout=max(1, int((seconds-time.monotonic()+start)*1000)))
        status = solver.check()
        event = dict(samples=len(points), status=str(status))
        row['events'].append(event)
        if status != z3.sat:
            row['status'] = str(status)
            if status == z3.unknown:
                row['reason'] = solver.reason_unknown()
            break
        model = solver.model()
        cv = {n: model.eval(v, model_completion=True).as_long() for n, v in params.items()}
        got = body(xs, cv, family, mask)
        want = reference(xs ^ np.uint64(cv['I'])) ^ np.uint64(PRE_C5 & mask)
        bad = np.flatnonzero(got != want)
        if not len(bad):
            row.update(status='reduced_sat', parameters=cv, verified_inputs=len(xs))
            if width == 32:
                x = z3.BitVec('full32_x', 32)
                check = z3.SolverFor('QF_BV')
                check.set(timeout=max(1, int((seconds-time.monotonic()+start)*1000)))
                check.add(body(x, cv, family, mask) != (reference(x ^ cv['I']) ^ (PRE_C5 & mask)))
                checked = check.check()
                if checked == z3.unsat:
                    row['status'] = 'proved_full32'
                elif checked == z3.sat:
                    p = check.model().eval(x).as_long()
                    points.append(p)
                    add(p)
                    continue
                else:
                    row['status'] = 'verification_unknown'
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
    ap.add_argument('--family', choices=('mixed', 'quad'), required=True)
    ap.add_argument('--width', type=int, choices=(11, 12, 16, 32), default=11)
    ap.add_argument('--seconds', type=float, default=120)
    ap.add_argument('--control', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    row = dict(normalization_controls=controls(), query=solve(args.family, args.width, args.seconds, args.control))
    args.output.write_text(json.dumps(row, indent=2)+'\n')
    print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
