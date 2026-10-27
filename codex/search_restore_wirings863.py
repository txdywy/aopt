"""Search four-instruction middle programs that absorb the output C5 XOR.

Find P(x) = F(x XOR I) XOR PRE_C5, with unknown I. The preceding C1 XOR
can absorb I at unchanged instruction count. A full32 solution would let
the last round omit its output restore; it still needs real IR scheduling.

Unlike search_absorb_restore863, connections and nonlinear MAD operands are
enumerated explicitly. MAD/XOR/AND/OR commute with low-bit projection: a
low-width UNSAT is conclusive for its wiring, while SAT is exhaustively
checked on that reduced domain and lifted before any executable claim.
Timeouts remain UNKNOWN. First-MAD canonicalization covers affine, square
plus constant and square plus x; dead/constant-only prefixes reduce to the
previously excluded <=3-instruction free-endpoint-key grammar.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations, product
import json
from pathlib import Path
import random
import time

import numpy as np
import z3

from codex.cegis_c1_absorb863 import middle, run_numpy, wirings_general
from codex.search_cross_encoding863 import PRE_C5


def expression(order, wiring, constants, x):
    values = [x]
    for op, indices, kk in zip(order, wiring, constants):
        pool = values + list(kk)
        a, b = (pool[i] for i in indices[:2])
        if op == 'mad':
            value = a*b + pool[indices[2]]
        elif op == '^':
            value = a ^ b
        elif op == '&':
            value = a & b
        elif op == '|':
            value = a | b
        else:
            raise ValueError(op)
        values.append(value)
    return values[-1]


def solve(order, wiring, width, seconds, output_key=PRE_C5, pins=None, dense=False,
          fixed_input_key=None):
    start = time.monotonic()
    mask = (1 << width)-1
    solver = z3.Then('simplify', 'solve-eqs', 'bit-blast', 'sat').solver()
    constants = [z3.BitVecs(f'k{i}a k{i}b', width) for i in range(len(order))]
    ik = z3.BitVec('input_key', width)
    if fixed_input_key is not None:
        solver.add(ik == (fixed_input_key & mask))
    for i, j, value in pins or ():
        solver.add(constants[i][j] == (value & mask))

    def target(x):
        return middle(x ^ ik, mask) ^ (output_key & mask)

    def add_point(p):
        x = z3.BitVecVal(p, width)
        solver.add(expression(order, wiring, constants, x) == target(x))

    rng = random.Random(8631)
    points = list(dict.fromkeys((v & mask) for v in [
        0, 1, 2, 3, 4, 7, 8, 31, 32, 127, 128, 255, 256, 511, 512, 513,
        1023, 1024, mask, mask-1, *(rng.getrandbits(width) for _ in range(8))]))
    if dense:
        points = list(dict.fromkeys(list(range(min(64, 1 << width))) + points))
    for p in points:
        add_point(p)
    if width <= 16:
        xs = np.arange(1 << width, dtype=np.uint64)
    else:
        xs = np.array(points + [1 << k for k in range(32)] +
                      [rng.getrandbits(width) for _ in range(2048)], dtype=np.uint64)
    iterations = 0
    row = dict(width=width, status='unknown')
    if fixed_input_key is not None:
        row['fixed_input_key'] = fixed_input_key
    if dense:
        row['dense_initial_samples'] = True
    if pins:
        row['pinned_coefficients'] = pins
    while time.monotonic()-start < seconds:
        solver.set(timeout=max(1, int((seconds-time.monotonic()+start)*1000)))
        result = solver.check()
        iterations += 1
        if result != z3.sat:
            row.update(status=str(result))
            if result == z3.unknown:
                row['reason'] = solver.reason_unknown()
            break
        model = solver.model()
        cv = [[model.eval(k, model_completion=True).as_long() for k in kk]
              for kk in constants]
        iv = model.eval(ik, model_completion=True).as_long()
        got = run_numpy(order, wiring, cv, xs, mask)
        want = middle(xs ^ np.uint64(iv), np.uint64(mask)) ^ np.uint64(output_key & mask)
        bad = np.flatnonzero(got != want)
        if len(bad):
            # Distinct necessary counterexamples, retaining the unknown input key.
            take = [int(xs[i]) for i in bad[:8]]
            for p in take:
                assert p not in points
                points.append(p)
                add_point(p)
            continue
        if width < 32:
            row.update(status='reduced_sat', constants=cv, input_key=iv,
                       verified_inputs=1 << width)
            break
        x = z3.BitVec('restore_full32_x', 32)
        cc = [[z3.BitVecVal(k, 32) for k in kk] for kk in cv]
        checker = z3.SolverFor('QF_BV')
        checker.set(timeout=max(1, int((seconds-time.monotonic()+start)*1000)))
        checker.add(expression(order, wiring, cc, x) !=
                    (middle(x ^ iv, mask) ^ (output_key & mask)))
        checked = checker.check()
        if checked == z3.unsat:
            row.update(status='proved_full32', constants=cv, input_key=iv)
            break
        if checked == z3.unknown:
            row.update(status='verification_unknown', reason=checker.reason_unknown())
            break
        p = checker.model().eval(x, model_completion=True).as_long()
        assert p not in points
        points.append(p)
        add_point(p)
    row.update(iterations=iterations, samples=len(points),
               seconds=round(time.monotonic()-start, 4))
    return row


def filtered_wirings(order):
    for wiring in wirings_general(order):
        if order[0] == 'mad' and wiring[0] not in ((0, 1, 2), (0, 0, 1), (0, 0, 0)):
            continue
        if order[-1] != 'mad' and max(wiring[-1]) >= 4:
            # A final XOR constant is a <=3-op free-output-key program.
            # A final AND/OR constant must be neutral, as the target is a permutation.
            continue
        yield wiring


def orders_for(mads):
    for positions in combinations(range(4), mads):
        for bitops in product(('^', '&', '|'), repeat=4-mads):
            it = iter(bitops)
            yield tuple('mad' if i in positions else next(it) for i in range(4))


def job(args):
    order, wiring, seconds, top, *options = args
    start = options[0] if options else 11
    dense = options[1] if len(options) > 1 else False
    trace = []
    row = {}
    for width in (8, 10, 11, 12, 16, 32):
        if width < start:
            continue
        if width > top:
            break
        row = solve(order, wiring, width, seconds, dense=dense)
        trace.append({k: row[k] for k in ('width', 'status', 'seconds')})
        if row['status'] != 'reduced_sat':
            break
    row.update(order=list(order), wiring=[list(w) for w in wiring], ladder=trace)
    return row


def controls():
    order = ('mad', 'mad', '^', 'mad')
    wiring = ((0, 1, 2), (0, 2, 3), (1, 2), (3, 4, 5))
    rows = []
    for key in (0, 1 << 10):
        # Keep the original multipliers while discovering all three addends
        # and the input key. The search itself does not use these pins.
        row = solve(order, wiring, 11, 30, key, [(0, 0, 33), (1, 0, 16896), (3, 0, 9)])
        assert row['status'] == 'reduced_sat', row
        rows.append(row)
    # Independent, exact full32 reference and boundary transport check.
    x = z3.BitVec('restore_reference_x', 32)
    cc = [[33, 0xE9F8CC1D], [16896, 0xACCF6200], [0, 0], [9, 0xFD7046C5]]
    solver = z3.SolverFor('QF_BV')
    solver.add(expression(order, wiring, cc, x) != middle(x, 0xFFFFFFFF))
    assert solver.check() == z3.unsat
    e = middle(x, 0xFFFFFFFF)
    enc = e ^ PRE_C5
    solver = z3.SolverFor('QF_BV')
    solver.add((enc ^ z3.LShR(enc, 16)) !=
               ((e ^ z3.LShR(e, 16)) ^ 0xB55A4F09))
    assert solver.check() == z3.unsat
    return dict(discovered_reduced_controls=rows, full32_reference=True,
                full32_transport_identity=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--orders', default='')
    ap.add_argument('--mads', type=int, choices=(1, 2, 3), default=2)
    ap.add_argument('--seconds', type=float, default=1)
    ap.add_argument('--top', type=int, choices=(11, 12, 16, 32), default=32)
    ap.add_argument('--start', type=int, choices=(8, 10, 11), default=11)
    ap.add_argument('--dense', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--per-order-limit', type=int, default=0)
    ap.add_argument('--controls-only', action='store_true')
    ap.add_argument('--skip-controls', action='store_true')
    ap.add_argument('--refine-from', type=Path, nargs='+')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    report = dict(target='F(x XOR I) XOR PRE_C5', output_key=PRE_C5,
                  input_key='arbitrary', z3=z3.get_version_string(), numpy=np.__version__,
                  seconds_per_width=args.seconds, top=args.top, new_executable=False,
                  start=args.start, dense_initial_samples=args.dense,
                  counts={}, by_order={}, unresolved=[])
    if not args.skip_controls:
        report['controls'] = controls()
        print(json.dumps(report['controls']), flush=True)
    if args.controls_only:
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        return
    jobs = []
    if args.refine_from:
        seen = set()
        for path in args.refine_from:
            for row in json.loads(path.read_text())['unresolved']:
                order = tuple(row['order'])
                wiring = tuple(tuple(w) for w in row['wiring'])
                if (order, wiring) not in seen:
                    jobs.append((order, wiring, args.seconds, args.top, args.start, args.dense))
                    seen.add((order, wiring))
        orders = sorted({j[0] for j in jobs})
        report['source_reports'] = [str(p) for p in args.refine_from]
    else:
        orders = [tuple(s.split(',')) for s in args.orders.split(';')] if args.orders else list(orders_for(args.mads))
        for order in orders:
            ws = list(filtered_wirings(order))
            if args.per_order_limit:
                # Deterministic broad sampling for the scout, never called exhaustive.
                random.Random(8632).shuffle(ws)
                ws = ws[:args.per_order_limit]
            jobs.extend((order, w, args.seconds, args.top, args.start, args.dense) for w in ws)
    report['jobs'] = len(jobs)
    report['per_order_limit'] = args.per_order_limit
    report['orders'] = [list(o) for o in orders]
    counts = Counter()
    by_order = defaultdict(Counter)
    started = time.monotonic()
    print(json.dumps(dict(jobs=len(jobs), orders=len(orders))), flush=True)

    def checkpoint(done):
        report.update(completed=done, counts=dict(counts), seconds=round(time.monotonic()-started, 2),
                      by_order={o: dict(c) for o, c in by_order.items()})
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(done=done, counts=dict(counts), seconds=report['seconds'])), flush=True)

    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(job, j) for j in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            row = future.result()
            counts[row['status']] += 1
            by_order[','.join(row['order'])][row['status']] += 1
            if row['status'] != 'unsat':
                report['unresolved'].append(row)
                if row['status'] in ('reduced_sat', 'proved_full32'):
                    print(json.dumps(row), flush=True)
            if done % 100 == 0 or done == len(jobs):
                checkpoint(done)
    checkpoint(len(jobs))


if __name__ == '__main__':
    main()
