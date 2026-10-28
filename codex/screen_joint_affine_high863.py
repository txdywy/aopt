"""Necessary high-half condition for an affine tail after XOR with SHR16.

For out=G*(h XOR q)+H, G odd and q<2^16, the inverse affine output
A*out+B has high16 exactly h_high16. This ignores the prefix producing q
entirely. UNSAT therefore excludes every such prefix for a stated h.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import z3

from codex.search_joint_live_boundary863 import sample_data, inputs, wanted, NAMES, NODE, MASK, C0


def solve(source, seconds, node=NODE, control=False):
    start = time.monotonic()
    a, b = z3.BitVecs('high_inverse_a high_inverse_b', 32)
    s = z3.SolverFor('QF_BV'); s.add(a & 1 == 1)
    if control:
        assert source == len(NAMES) and node in (0, 1 << 31)
        ai = pow(4097, -1, 1 << 32)
        s.add(a == ai, b == (-ai*(C0+node)) & MASK)
    def h(v):
        return v[source] if source < len(NAMES) else (9*v[0]+0xFD7046C5) & MASK
    initial, tests, vv, values = sample_data(); points = list(initial)
    def add(v):
        s.add(z3.Extract(31, 16, a*wanted(v, node)+b) == h(v) >> 16)
    for v in vv:
        add(v)
    with np.errstate(over='ignore'):
        target = wanted(values, np.uint64(node)); expected = h(values) >> 16
    result = dict(source=source, source_name=NAMES[source] if source < len(NAMES) else 'e_positive_control_only',
                  node=node, status='unknown', control=control, events=[], new_executable=False)
    while time.monotonic()-start < seconds:
        s.set(timeout=max(1, int((seconds-time.monotonic()+start)*1000)))
        status = s.check(); result['events'].append(dict(status=str(status), samples=len(points)))
        if status != z3.sat:
            result['status'] = str(status)
            if status == z3.unknown:
                result['reason'] = s.reason_unknown()
            break
        model = s.model(); aa, bb = [model.eval(v, model_completion=True).as_long() for v in (a, b)]
        with np.errstate(over='ignore'):
            got = ((aa*target+bb) & MASK) >> 16
        bad = np.flatnonzero(got != expected)
        if len(bad):
            take = [tests[i] for i in bad[:8]]
        else:
            primary = z3.BitVec('high_primary', 32); v = inputs(primary)
            check = z3.SolverFor('QF_BV'); check.set(timeout=max(1, int((seconds-time.monotonic()+start)*1000)))
            check.add(z3.Extract(31, 16, aa*wanted(v, node)+bb) != z3.Extract(31, 16, h(v)))
            verified = check.check()
            if verified == z3.unsat:
                result.update(status='necessary_survivor_proved', inverse_a=aa, inverse_b=bb)
                break
            if verified == z3.unknown:
                result['status'] = 'verification_unknown'; break
            take = [check.model().eval(primary, model_completion=True).as_long()]
        for p in take:
            assert p not in points
            points.append(p); add(inputs(p))
    result.update(seconds=round(time.monotonic()-start, 4), samples=len(points))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds', type=float, default=30)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    controls = [solve(len(NAMES), 30, node, True) for node in (0, 1 << 31)]
    assert all(r['status'] == 'necessary_survivor_proved' for r in controls), controls
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(solve, i, args.seconds) for i in range(len(NAMES))]
        rows = [f.result() for f in as_completed(futures)]
    result = dict(node=NODE, input_names=NAMES, controls=controls, results=rows,
                  full32_semantics=True, ignores_entire_shift_prefix=True,
                  best_executable_cycles=864, new_executable=False)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print([(r['source_name'], r['status']) for r in rows], flush=True)


if __name__ == '__main__':
    main()
