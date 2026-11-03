"""Prefix-independent necessary conditions for affine quotient bypasses.

For out=G*q+W with a single downward-transfer quotient/remainder q,
G must be odd. Then q=A*(target-W), A odd. Range and high-output facts
can exclude this form without synthesizing the first operation at all.
No actual shorter kernel is claimed by a necessary-condition survivor.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import random
import time

import z3

from codex.audit_computed_shift_amount863 import zero_points
from codex.audit_division_first_t863 import CLASSES
from codex.search_early_boundary863 import MASK, NAMES, NODE, early_inputs, wanted, legal_point


def pool():
    rng = random.Random(8642861)
    return list(dict.fromkeys([*range(128), *CLASSES, *(p ^ 65536 for p in CLASSES),
                              *zero_points().values(), *(1 << j for j in range(32)),
                              *(rng.getrandbits(32) for _ in range(256))]))


def check(source, family, seconds=15, control=False):
    start = time.monotonic(); points = pool()
    vv = [early_inputs(b) for b in points]
    if control:
        assert family in ('bounded', 'one_high')
        qs = [(v[4] % 65536 if family == 'bounded' else MASK//(v[4] | 1)) for v in vv]
        targets = [(17*q+v[source]) & MASK for q, v in zip(qs, vv)]
    else:
        targets = [wanted(v) for v in vv]
    residuals = list(dict.fromkeys((y-v[source]) & MASK for v, y in zip(vv, targets)))
    a = z3.BitVec('middle_division_inverse_a', 32)
    solver = z3.SolverFor('QF_BV'); solver.set(timeout=int(seconds*1000)); solver.add(a & 1 == 1)
    if control:
        solver.add(a == pow(17, -1, 1 << 32))
    outputs = [a*z3.BitVecVal(r, 32) for r in residuals]
    if family == 'bounded':
        solver.add(*(z3.ULE(q, 1 << 31) for q in outputs))
    elif family == 'one_high':
        # Each distinct residual gives a distinct q because A is odd.
        # All q>2^31 must equal the same prepared dividend, for any of
        # floor/ceil/remainder, hence at most one distinct high q.
        solver.add(z3.PbLe([(z3.UGT(q, 1 << 31), 1) for q in outputs], 1))
    else:
        raise ValueError(family)
    status = solver.check()
    result = dict(source=source, source_name=NAMES[source], family=family, status=str(status),
                  points=points, distinct_residuals=len(residuals), positive_control=control,
                  prefix_completely_ignored=True, seconds=round(time.monotonic()-start, 6))
    if status == z3.unknown:
        result['reason'] = solver.reason_unknown()
    elif status == z3.sat:
        result['inverse_a'] = solver.model().eval(a, model_completion=True).as_long()
        result['necessary_survivor_only'] = True
    return result


def finite_anchors():
    zeros = zero_points(); rows = []
    for numerator, name in enumerate(NAMES):
        b = zeros[name]; v = early_inputs(b); y = wanted(v)
        assert v[numerator] == 0
        equal = [j for j, word in enumerate(v) if word == y]
        rows.append(dict(numerator=numerator, numerator_name=name, zero_witness=legal_point(b),
                         possible_initial_addends=equal,
                         proof='Every valid DIV/REM/CDIV with numerator0 returns0. A final q*G+W therefore equals W at this point, independent of the prefix/denominator/coefficient.'))
    factors = []
    for source, name in enumerate(NAMES):
        b = next(p for p in CLASSES if early_inputs(p)[source] & 1 == 0)
        v, w = early_inputs(b), early_inputs(b ^ 65536)
        assert [x & 1 for x in v] == [x & 1 for x in w] and (wanted(v) ^ wanted(w)) & 1
        factors.append(dict(source=source, source_name=name, primary_pair=[b, b ^ 65536],
                            legal_points=[legal_point(b), legal_point(b ^ 65536)],
                            proof='An initial multiplicative factor is even at both points. Any quotient times that factor plus an initial/T-only addend has equal output parity, but target parity differs.'))
    return dict(zero_dividend_addend_cases=rows, initial_multiplier_cases=factors)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds', type=float, default=15)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); rows = []
    controls = [check(0, family, 15, True) for family in ('bounded', 'one_high')]
    assert all(r['status'] == 'sat' for r in controls), controls
    anchors = finite_anchors()
    result = dict(node=NODE, input_names=NAMES, controls=controls, finite_anchors=anchors,
                  results=rows, complete=False, best_executable_cycles=864, new_executable=False)
    with ProcessPoolExecutor(args.workers) as executor:
        futures = [executor.submit(check, source, family, args.seconds)
                   for source in range(9) for family in ('bounded', 'one_high')]
        for future in as_completed(futures):
            row = future.result(); rows.append(row)
            print(row['source_name'], row['family'], row['status'], row['seconds'], flush=True)
            args.output.write_text(json.dumps(result, indent=2)+'\n')
    result.update(complete=True, excluded=sum(r['status'] == 'unsat' for r in rows),
                  scope='Single quotient/remainder q and final constant-G MAD G*q+W, W an initial word, on nine live words and root-only constants. Bounded q applies to arbitrary uint32 numerator divided by prepared D>=2 using floor/ceil. One-high applies to a prepared dividend and arbitrary nonzero runtime denominator for floor/ceil/REM. Zero-dividend anchors cover initial numerator with arbitrary denominator and initial final addend. No computed final addend or computed multiplicative factor is assumed excluded.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
