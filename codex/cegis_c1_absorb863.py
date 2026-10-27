"""Explicit-wiring CEGIS for the open C1-absorption templates.

Target (uint32 wrap):  P(x) XOR O == F(x XOR C1)
  F(x) = 9*((33*x + 0xe9f8cc1d) XOR (16896*x + 0xaccf6200)) + 0xfd7046c5
P has four instructions over the pool [x, earlier results, two private
constants]. MAD multiplies two pool values and adds a third; bitwise ops are
XOR/AND/OR. With no right shifts, the low w bits of every value depend only
on the low w bits of its inputs, so UNSAT at width w excludes every full
uint32 program with the same wiring. SAT at width w is only a candidate and
is lifted width by width before any claim.

The previous Z3 runs used free operand selectors and timed out; here each
canonical wiring is enumerated and solved separately.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations_with_replacement, product
import json
from pathlib import Path
import random
import time

import numpy as np
import z3

C1 = 0xC761C23C
K1, K2, C4 = 0xE9F8CC1D, 0xACCF6200, 0xFD7046C5


def middle(x, mask):
    return (9 * (((33 * x + K1) & mask) ^ ((16896 * x + K2) & mask)) + C4) & mask


def target(x, mask):
    return middle(x ^ (C1 & mask), mask)


def pool_size(i):
    return i + 3  # x, t0..t(i-1), k0_i, k1_i


def wirings(order, second):
    """Yield canonical non-degenerate wirings for an affine-first order."""
    assert order[0] == 'mad'
    first = (0, 1, 2)                     # t0 = x*k0 + k1
    a1, b1 = second                       # pool of instr1: [x, t0, k0, k1]
    for a2, b2 in combinations_with_replacement(range(5), 2):
        if a2 == b2:
            continue                      # x OP x is trivial
        if a2 >= 3 and b2 >= 3:
            continue                      # constant OP constant
        for a3, b3 in combinations_with_replacement(range(6), 2):
            if a3 >= 4 and b3 >= 4:
                pass                      # constant product allowed (then t3 = c + K)
            for c3 in range(6):
                used = {a1, b1, a2, b2, a3, b3, c3}
                # pools: instr1 index1=t0; instr2 index1=t0,2=t1; instr3 1=t0,2=t1,3=t2
                uses_t0 = (1 in (a1, b1)) or (1 in (a2, b2)) or (1 in (a3, b3, c3))
                uses_t1 = (2 in (a2, b2)) or (2 in (a3, b3, c3))
                uses_t2 = 3 in (a3, b3, c3)
                if not (uses_t0 and uses_t1 and uses_t2):
                    continue
                if a3 >= 4 and b3 >= 4 and c3 >= 4:
                    continue
                yield (first, (a1, b1), (a2, b2), (a3, b3, c3))


def _instr_options(i, op):
    """Canonical operand tuples for instruction i over pool [x, t0.., k0, k1]."""
    size = pool_size(i)
    ka, kb = i + 1, i + 2
    consts = {ka, kb}
    out = []
    if op == 'mad':
        for a in range(size):
            for b in range(a, size):
                for c in range(size):
                    used = [v for v in (a, b, c) if v in consts]
                    if len(used) == 3:
                        continue
                    # rename constants: first constant appearing must be k0
                    if used and used[0] != ka:
                        continue
                    if len(set(used)) == 2 and used[1] != kb:
                        continue
                    out.append((a, b, c))
    else:
        for a in range(size):
            for b in range(a + 1, size):
                if a in consts and b in consts:
                    continue
                if b == kb and a not in consts:
                    continue      # x OP k1 is x OP k0 after renaming
                out.append((a, b))
    return out


def wirings_general(order):
    """All canonical live wirings of a four-instruction order."""
    opts = [_instr_options(i, op) for i, op in enumerate(order)]
    n = len(order)
    for combo in product(*opts):
        live = True
        for j in range(n - 1):
            ref = j + 1           # t_j sits at pool index j+1 for later instructions
            if not any(ref in combo[i] and ref < pool_size(i) - 2 for i in range(j + 1, n)):
                live = False
                break
        if live:
            yield combo


def run_numpy(order, wiring, consts, xs, mask):
    vals = [xs]
    for i, op in enumerate(order):
        k0, k1 = consts[i]
        pool = vals + [np.full_like(xs, k0), np.full_like(xs, k1)]
        idx = wiring[i]
        if op == 'mad':
            a, b, c = (pool[j] for j in idx)
            v = (a * b + c) & mask
        else:
            a, b = (pool[j] for j in idx)
            v = a ^ b if op == '^' else a & b if op == '&' else a | b
        vals.append(v)
    return vals[-1]


def solve(order, wiring, width, seconds, seed=1):
    mask = (1 << width) - 1
    xs = np.arange(1 << width, dtype=np.uint64)
    want = np.array([target(int(x), mask) for x in range(1 << width)], dtype=np.uint64)
    solver = z3.Then('simplify', 'solve-eqs', 'bit-blast', 'sat').solver()
    ks = [z3.BitVecs(f'k{i}_0 k{i}_1', width) for i in range(len(order))]
    out_key = z3.BitVec('O', width)

    def symbolic(x):
        vals = [z3.BitVecVal(x, width)]
        for i, op in enumerate(order):
            pool = vals + list(ks[i])
            idx = wiring[i]
            if op == 'mad':
                a, b, c = (pool[j] for j in idx)
                v = a * b + c
            else:
                a, b = (pool[j] for j in idx)
                v = a ^ b if op == '^' else a & b if op == '&' else a | b
            vals.append(v)
        return vals[-1]

    rng = random.Random(seed)
    points = list(dict.fromkeys([0, 1, 2, 3, 4, 5, 7, 8, 16, mask, mask - 1,
                                 1 << (width - 1)] + [rng.getrandbits(width) for _ in range(12)]))
    for p in points:
        solver.add(symbolic(p) ^ out_key == target(p, mask))
    start = time.monotonic()
    iterations = 0
    while True:
        left = seconds - (time.monotonic() - start)
        if left <= 0:
            return dict(result='timeout', iterations=iterations, samples=len(points))
        solver.set(timeout=max(1, int(left * 1000)))
        status = solver.check()
        iterations += 1
        if status == z3.unsat:
            return dict(result='unsat', iterations=iterations, samples=len(points))
        if status == z3.unknown:
            return dict(result='timeout', iterations=iterations, samples=len(points))
        model = solver.model()
        consts = [(model.eval(a, model_completion=True).as_long(),
                   model.eval(b, model_completion=True).as_long()) for a, b in ks]
        o = model.eval(out_key, model_completion=True).as_long()
        got = run_numpy(order, wiring, consts, xs, mask) ^ np.uint64(o)
        bad = np.nonzero(got != want)[0]
        if len(bad) == 0:
            return dict(result='sat', consts=consts, out_key=o, iterations=iterations,
                        samples=len(points))
        p = int(bad[rng.randrange(len(bad))])
        points.append(p)
        solver.add(symbolic(p) ^ out_key == target(p, mask))


LADDER = (8, 10, 11, 12, 14, 16)


def job(args):
    """Climb a width ladder; the first UNSAT excludes every wider program."""
    order, wiring, top, seconds = args
    t = time.monotonic()
    trace = []
    row = dict(result='unsat')
    for width in [w for w in LADDER if w <= top]:
        row = solve(order, wiring, width, seconds)
        trace.append((width, row['result']))
        if row['result'] != 'sat':
            break
    row.update(order=list(order), wiring=[list(w) for w in wiring], ladder=trace,
               wall=round(time.monotonic() - t, 3))
    return row


CLASSES = [(('mad', '&', '^', 'mad'), (0, 2)), (('mad', '&', '^', 'mad'), (1, 2)),
           (('mad', '^', '&', 'mad'), (0, 2)), (('mad', '^', '&', 'mad'), (1, 2)),
           (('mad', '^', '|', 'mad'), (0, 2)), (('mad', '^', '|', 'mad'), (1, 2)),
           (('mad', '|', '^', 'mad'), (0, 2)), (('mad', '|', '^', 'mad'), (1, 2))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--width', type=int, default=16, help='Top of the width ladder')
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--classes', default='all')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--orders', default='', help='Semicolon list of comma orders, e.g. "&,mad,^,mad"')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    jobs = []
    if args.orders:
        for text in args.orders.split(';'):
            order = tuple(text.split(','))
            for w in wirings_general(order):
                jobs.append((order, w, args.width, args.seconds))
    else:
        chosen = CLASSES if args.classes == 'all' else [CLASSES[int(i)] for i in args.classes.split(',')]
        for order, second in chosen:
            for w in wirings(order, second):
                jobs.append((order, w, args.width, args.seconds))
    if args.limit:
        jobs = jobs[:args.limit]
    print(json.dumps(dict(jobs=len(jobs), width=args.width)), flush=True)
    rows = []
    counts = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, j) for j in jobs]
        for k, f in enumerate(as_completed(futures), 1):
            row = f.result()
            rows.append(row)
            counts[row['result']] += 1
            if row['result'] != 'unsat':
                print(json.dumps(row), flush=True)
            if k % 500 == 0 or k == len(jobs):
                print(json.dumps(dict(done=k, counts=dict(counts))), flush=True)
                args.output.write_text(json.dumps(dict(counts=dict(counts), rows=rows)) + '\n')
    args.output.write_text(json.dumps(dict(counts=dict(counts), rows=rows)) + '\n')


if __name__ == '__main__':
    main()
