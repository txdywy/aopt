"""Necessary quotient-range sketches for unresolved prepared-division rows.

Each concrete input receives an independent arbitrary q in a sound interval.
This deliberately drops the division equation and all cross-input consistency
of q. UNSAT excludes the original row; SAT is only a finite relaxed model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import time

import z3

from codex.audit_one_high_tail863 import information_fibre
from codex.scout_joint_wirings863 import write_report
from codex.search_focused_joint_boundary863 import edge_points, key
from codex.search_joint_live_boundary863 import inputs, wanted, sample_data, NODE, MASK
from codex.search_live_boundary863 import expression


DIVS = ('//', '%', 'cdiv')


def quotient_maximum(op, divisor=None):
    if op not in DIVS:
        raise ValueError(op)
    if divisor is not None and not 2 <= divisor <= MASK:
        raise ValueError('This sketch uses the independently excluded divisor1 case.')
    if op == '%':
        # Even REM is T-only in bit0; the478-state fibre bounds odd D.
        cap = information_fibre()['prepared_remainder_divisor_max']
        return min(cap, divisor if divisor is not None else cap)-1
    d = 2 if divisor is None else divisor
    return MASK//d if op == '//' else (MASK+d-1)//d


def target_quotient_maximum(op, divisor=None):
    maximum = quotient_maximum(op, divisor)
    if op in ('//','cdiv'):
        # Every initial low16 is invariant under b^65536. Any T prefix
        # preserves this, and floor(P/2) or ceil(P/2) bit0 depends only
        # on P low2. A T suffix still cannot transmit b16 to target bit0.
        # Thus D2, as well as D1, is impossible for this actual target.
        maximum = min(maximum, MASK//3 if op=='//' else (MASK+2)//3)
    return maximum


def sketch(row, constants, initial, quotient):
    values = list(initial)
    for op, w, cc in zip(row['order'], row['wiring'], constants):
        result = quotient if op in DIVS else expression(
            dict(order=[op], wiring=[w]), [cc], values)
        values.append(result)
    return values[-1]


def validate_row(row):
    if len(row['order']) != len(row['wiring']):
        raise ValueError('Mismatched instruction and wiring lists')
    special = [i for i, op in enumerate(row['order']) if op in DIVS]
    if len(special) != 1 or any(op not in ('mad', '^', '&', '|', *DIVS) for op in row['order']):
        raise ValueError('Exactly one prepared DIV and otherwise T instructions are required')
    for i, (op, w) in enumerate(zip(row['order'], row['wiring'])):
        size = 3 if op == 'mad' else 1 if op in DIVS else 2
        if len(w) != size or any(not 0 <= j < 8+i+(0 if op in DIVS else 2) for j in w):
            raise ValueError('Invalid prepared-operand wiring')
    return special[0]


def solve(row, seconds=2, points=None, pins=()):
    special = validate_row(row)
    started = time.monotonic()
    op = row['order'][special]
    maximum = target_quotient_maximum(op, row.get('divisor_pin'))
    points = list(dict.fromkeys([*sample_data(True)[0], *edge_points()])) if points is None else list(points)
    constants = [z3.BitVecs(f'qr_k{i}_0 qr_k{i}_1', 32) for i in range(len(row['order']))]
    solver = z3.SolverFor('QF_BV')
    for i, j, v in pins:
        solver.add(constants[i][j] == (v & MASK))
    for rank, b in enumerate(points):
        q = z3.BitVec(f'qr_q{rank}', 32)
        solver.add(z3.ULE(q, maximum), sketch(row, constants, inputs(b), q) == wanted(inputs(b), NODE))
    solver.set(timeout=max(1, int(seconds*1000)))
    status = solver.check()
    result = dict(row, status='range_unsat' if status == z3.unsat else 'range_sat' if status == z3.sat else 'unknown',
                  quotient_maximum=maximum, samples=len(points),
                  divisor2_excluded_by_lowbit=op in ('//','cdiv'),
                  seconds=round(time.monotonic()-started, 4),
                  relaxed_quotients_independent=True, new_executable=False)
    if status == z3.unknown:
        result['reason'] = solver.reason_unknown()
    if status == z3.sat:
        model = solver.model()
        result['relaxed_constants'] = [[model.eval(k, model_completion=True).as_long() for k in pair]
                                       for pair in constants]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path('codex/joint_prepared_division863_open_20261029.json'))
    ap.add_argument('--seconds', type=float, default=2)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    raw = args.source.read_bytes()
    source = json.loads(gzip.decompress(raw) if args.source.suffix == '.gz' else raw)
    assert source['completed'] == len(source['manifest']) == len(source['results'])
    rows = [{k: row[k] for k in ('order', 'wiring', 'divisor_pin') if k in row}
            for row in source['results'] if row['status'] in ('unknown', 'range_sat')]
    assert len({(key(r), r.get('divisor_pin')) for r in rows}) == len(rows)
    points = list(dict.fromkeys([*sample_data(True)[0], *edge_points()]))
    report = dict(source=str(args.source), source_sha256=hashlib.sha256(raw).hexdigest(),
                  node=NODE, input_names=['z','u','v','b','prior','a','a_shr19','a_xor_c1'],
                  primary_points=points, scope='One prepared DIV/REM/CDIV, other operations T-only. '
                  'Divisors1/2 for floor/ceil and even REM use bit0 exclusions; REM uses the478-state bound. '
                  'Only UNSAT is a closure; a relaxed SAT or timeout is still open.',
                  arbitrary_q_per_sample=True, seconds_per_query=args.seconds,
                  manifest=rows, results=[], completed=0, counts={},
                  new_executable=False, best_executable_cycles=864)
    started = time.monotonic()
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(solve, row, args.seconds, points) for row in rows]
        for done, future in enumerate(as_completed(futures), 1):
            report['results'].append(future.result())
            if done % 10 == 0 or done == len(rows):
                report.update(completed=done, counts=dict(Counter(r['status'] for r in report['results'])),
                              seconds=round(time.monotonic()-started, 3))
                write_report(args.output, report)
                print(done, report['counts'], report['seconds'], flush=True)


if __name__ == '__main__':
    main()
