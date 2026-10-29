"""Explicit-wiring CEGIS for folding C5 into the middle of the hash.

The hash ends with  e = F(y)  (four instructions: MAD, MAD, XOR, MAD) and then
L16(e) = e ^ (e >> 16), after which the 864 kernel spends 1,279 lanes on C5:
511 node pre-XORs (d0-d8), 512 recoveries (d9, d10) and 256 on the output.
L16 is XOR-linear, so  L16(e ^ D) = L16(e) ^ C5  with  D = L16(C5).  If four
instructions computed  F(y ^ K) ^ O  with

  K  any constant (stage 1 ends with the C1 XOR, which absorbs it), and
  O  in D ^ L16({0, 2^31, MASK, 2^31 - 1}) (keys the next MAD0 absorbs),

all C5 work would disappear. At width w <= 15 all four O are D mod 2^w.

Target (uint32 wrap):  P(z) == F(z XOR K) XOR D   with K symbolic.
P has four instructions over the pool [z, earlier results, two private
constants] with MAD (covers +, -, *), XOR, AND, OR. With no right shifts the
low w bits of every value depend only on the low w bits of its inputs, and the
target's low w bits depend only on K mod 2^w, so UNSAT at width w excludes
every uint32 program with the same wiring and every K.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import json
from pathlib import Path
import random
import time

import numpy as np
import z3

from codex.cegis_c1_absorb863 import K1, K2, C4, run_numpy, wirings_general

MASK = 0xFFFFFFFF
C5 = 0xB55A4F09
D = C5 ^ C5 >> 16                         # L16(C5) = 0xb55afa53
OKEYS = tuple(D ^ c ^ c >> 16 for c in (0, 1 << 31, MASK, (1 << 31) - 1))


def middle(y, mask):
    return (9 * (((33 * y + K1) & mask) ^ ((16896 * y + K2) & mask)) + C4) & mask


def target(z, key, mask, okey=None):
    return middle(z ^ key & mask, mask) ^ (D if okey is None else okey) & mask


def orders(mads):
    return [o for o in product(('mad', '^', '&', '|'), repeat=4) if o.count('mad') == mads]


def symbolic(order, wiring, ks, x, width):
    vals = [x]
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


def z3_middle(y):
    return 9 * ((33 * y + K1) ^ (16896 * y + K2)) + C4


def _symmetries(order, wiring):
    """The redundancies behind symmetry_pins: (first_xor, chains, products) where
    chains lists (j, i) with t_j merged into its only source t_i, and products lists
    the instructions MAD(k0, k1, v) = v + k0*k1."""
    n = len(order)
    variables = [[j for j in wiring[i] if j <= i] for i in range(n)]
    uses = Counter(j for vs in variables for j in vs)
    first_xor = order[0] == '^' and uses[0] == 1 and tuple(wiring[0]) == (0, 1)
    chains = []
    for j in range(1, n):
        if len(variables[j]) != 1 or variables[j][0] == 0 or uses[variables[j][0]] != 1:
            continue
        i = variables[j][0] - 1
        if len(variables[i]) != 1:
            continue
        if order[j] == order[i] and order[j] != 'mad':
            chains.append((j, i))
        elif order[j] == order[i] == 'mad' and tuple(wiring[i][1:]) == (i + 1, i + 2) \
                and wiring[i][0] <= i and tuple(wiring[j]) in ((i + 1, j + 1, j + 2),
                                                             (j + 1, j + 2, i + 1)):
            chains.append((j, i))
    products = [j for j in range(n) if order[j] == 'mad'
                and tuple(wiring[j][:2]) == (j + 1, j + 2) and wiring[j][2] <= j]
    return first_xor, chains, products


def symmetry_pins(order, wiring):
    """Constants that can be fixed without losing a solution. Returns {(i, slot): v}
    with v = -1 for all ones.

    first-xor: t0 = z ^ k and z is used nowhere else, so P depends on z ^ k and the
      free input key K absorbs k (pin k = 0).
    chain: t_j = t_i OP k, t_i = u OP k' (same bitwise OP, both one variable) and
      t_j is the only reader of t_i, so k merges into k' (pin k to the identity).
    affine chain: t_j = MAD(t_i, k, k') or MAD(k, k', t_i) with t_i = MAD(u, c, c'),
      t_j its only reader; the composite is a MAD of u, so t_j is WLOG identity.
    product: MAD(k0, k1, v) = v + k0*k1 depends only on the product (pin k1 = 1).
    """
    first_xor, chains, products = _symmetries(order, wiring)
    pins = {(0, 0): 0} if first_xor else {}
    for j, i in chains:
        if order[j] != 'mad':
            (slot,) = [p - j - 1 for p in wiring[j] if p > j]
            pins[(j, slot)] = -1 if order[j] == '&' else 0
        elif tuple(wiring[j]) == (i + 1, j + 1, j + 2):
            pins[(j, 0)], pins[(j, 1)] = 1, 0
        else:
            pins[(j, 0)] = 0
    for j in products:
        pins[(j, 1)] = 1
    return pins


def merge_constants(order, wiring, consts, mask):
    """Constructive form of symmetry_pins: constants obeying the pins and an input
    shift s with P_consts(z) == P_new(z ^ s) for every z."""
    first_xor, chains, products = _symmetries(order, wiring)
    new = [list(c) for c in consts]
    for j, i in sorted(chains, reverse=True):
        if order[j] != 'mad':
            (sj,) = [p - j - 1 for p in wiring[j] if p > j]
            (si,) = [p - i - 1 for p in wiring[i] if p > i]
            k = new[j][sj]
            new[i][si] = (new[i][si] ^ k if order[j] == '^' else
                          new[i][si] & k if order[j] == '&' else new[i][si] | k)
            new[j][sj] = mask if order[j] == '&' else 0
        else:
            c, c1 = new[i]
            k0, k1 = new[j]
            if tuple(wiring[j]) == (i + 1, j + 1, j + 2):
                new[i] = [k0 * c & mask, (k0 * c1 + k1) & mask]
                new[j] = [1, 0]
            else:
                new[i] = [c, (c1 + k0 * k1) & mask]
                new[j] = [0, k1]
    for j in products:
        new[j] = [new[j][0] * new[j][1] & mask, 1]
    shift = 0
    if first_xor:
        shift, new[0][0] = new[0][0], 0
    return [tuple(c) for c in new], shift


def solve(order, wiring, width, seconds, seed=1, pins=None):
    mask = (1 << width) - 1
    xs = np.arange(1 << width, dtype=np.uint64)
    solver = z3.Then('simplify', 'solve-eqs', 'bit-blast', 'sat').solver()
    ks = [z3.BitVecs(f'k{i}_0 k{i}_1', width) for i in range(len(order))]
    key = z3.BitVec('K', width)
    for (i, slot), v in (pins or {}).items():
        solver.add(ks[i][slot] == v & mask)

    def constrain(p):
        x = z3.BitVecVal(p, width)
        solver.add(symbolic(order, wiring, ks, x, width) == z3_middle(x ^ key) ^ (D & mask))

    rng = random.Random(seed)
    points = list(dict.fromkeys([0, 1, 2, 3, 4, 5, 7, 8, 16, mask, mask - 1,
                                 1 << (width - 1)] + [rng.getrandbits(width) for _ in range(12)]))
    for p in points:
        constrain(p)
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
        k = model.eval(key, model_completion=True).as_long()
        got = run_numpy(order, wiring, consts, xs, mask)
        want = np.array([target(int(x), k, mask) for x in range(1 << width)], dtype=np.uint64)
        bad = np.nonzero(got != want)[0]
        if len(bad) == 0:
            return dict(result='sat', consts=consts, key=k, iterations=iterations,
                        samples=len(points))
        p = int(bad[rng.randrange(len(bad))])
        points.append(p)
        constrain(p)


def lift(order, wiring, width=12, pins=None, cap=4_000_000, chunk=1 << 14, seed=7):
    """Exact bit-by-bit enumeration. Every value is a T-function, so a program meets
    the target mod 2^m exactly when its constants and K do mod 2^m. Level m extends
    every survivor of level m-1 by one bit of each unknown and keeps those that
    match on all 2^m inputs. Zero survivors at some level proves the wiring
    impossible for every constant and every K; no solver is involved."""
    used = sorted({(i, p - i - 1) for i in range(len(order)) for p in wiring[i] if p > i})
    pins = {k: v for k, v in (pins or {}).items() if k in used}
    names = used + ['K']
    free = [u for u, n in enumerate(names) if n not in pins]
    pats = np.array([[(e >> f) & 1 for f in range(len(free))] for e in range(1 << len(free))],
                    dtype=np.uint64)
    rng = np.random.default_rng(seed)
    states = np.zeros((1, len(names)), dtype=np.uint64)
    counts = []

    def evaluate(rows, ys, mask):
        col = {n: rows[:, u][:, None] for u, n in enumerate(names)}
        vals = [np.broadcast_to(ys[None, :], (len(rows), len(ys)))]
        for i, op in enumerate(order):
            pool = lambda p: vals[p] if p <= i else col[(i, p - i - 1)]
            if op == 'mad':
                a, b, c = (pool(p) for p in wiring[i])
                vals.append((a * b + c) & mask)
            else:
                a, b = (pool(p) for p in wiring[i])
                vals.append(a ^ b if op == '^' else a & b if op == '&' else a | b)
        y = ys[None, :] ^ col['K']
        want = (9 * (((33 * y + K1) & mask) ^ ((16896 * y + K2) & mask)) + C4 ^ D) & mask
        return np.all(vals[-1] == want, axis=1)

    for m in range(1, width + 1):
        mask, bit = np.uint64((1 << m) - 1), np.uint64(1 << (m - 1))
        ys_all = np.arange(1 << m, dtype=np.uint64)
        probe = rng.choice(ys_all, size=min(32, len(ys_all)), replace=False)
        fixed = np.zeros(len(names), dtype=np.uint64)
        for u, n in enumerate(names):
            if n in pins:
                fixed[u] = np.uint64((pins[n] >> (m - 1)) & 1) * bit
        ext = np.zeros((len(pats), len(names)), dtype=np.uint64)
        ext[:, free] = pats * bit
        ext |= fixed
        keep = []
        for s in range(0, len(states), max(1, chunk // len(pats))):
            rows = (states[s:s + max(1, chunk // len(pats))][:, None, :] | ext[None, :, :]).reshape(-1, len(names))
            rows = rows[evaluate(rows, probe, mask)]
            if len(rows):
                rows = rows[evaluate(rows, ys_all, mask)]
            keep.append(rows)
        states = np.concatenate(keep) if keep else np.zeros((0, len(names)), dtype=np.uint64)
        counts.append(int(len(states)))
        if not len(states):
            return dict(result='unsat', level=m, counts=counts)
        if len(states) > cap:
            return dict(result='too-many', level=m, counts=counts)
    return dict(result='survivors', level=width, counts=counts)


LADDER = (10, 11, 12, 14)


def job(args):
    """Climb a width ladder; the first UNSAT excludes every wider program."""
    order, wiring, ladder, seconds = args
    t = time.monotonic()
    trace = []
    row = dict(result='unsat')
    for width in ladder:
        row = solve(order, wiring, width, seconds)
        trace.append((width, row['result']))
        if row['result'] != 'sat':
            break
    row.update(order=list(order), wiring=[list(w) for w in wiring], ladder=trace,
               wall=round(time.monotonic() - t, 3))
    return row


def mixed_difference_witness(key, width=11):
    """A rectangle x, x+h, x+k, x+h+k (h*k = 2^width) whose mixed difference of the
    target is nonzero mod 2^width; every integer polynomial gives zero there."""
    mask = (1 << width) - 1
    for i in range(1, width):
        h, k = 1 << i, 1 << (width - i)
        for x in range(1 << width):
            d = (target(x + h + k, key, mask) - target(x + h, key, mask)
                 - target(x + k, key, mask) + target(x, key, mask)) & mask
            if d:
                return x, h, k, d
    return None


def kinds(order, wiring):
    """'local' (bit-local in z: bit i depends only on z_i), 'poly' (integer polynomial
    in z and bit-local values), 'const' or 'other', for z and each result."""
    kind = ['local']
    for i, op in enumerate(order):
        ks = [kind[j] if j <= i else 'const' for j in wiring[i]]
        if op == 'mad':
            kind.append('other' if 'other' in ks else 'const' if set(ks) == {'const'} else 'poly')
        else:
            kind.append('local' if set(ks) <= {'local', 'const'} else 'other')
    return kind


def carry_free_class(order, wiring):
    """'class' if the output is a polynomial in z and bit-local values of z;
    'class+key' if it is such a value combined with one constant by XOR/AND/OR."""
    kind = kinds(order, wiring)
    if kind[-1] in ('poly', 'local'):
        return 'class'
    n = len(order) - 1
    vs = [j for j in wiring[n] if j <= n]
    if order[n] != 'mad' and len(vs) == 1 and kind[vs[0]] in ('poly', 'local'):
        return 'class+key'
    return None


def carry_free_certificate(width=11):
    """For every K mod 2^width, rectangles {x, x|h, x|k, x|h|k} (x has 0 at bits
    i, j with i + j = width, h = 2^i, k = 2^j) such that for every O mod 2^width
    some rectangle has a nonzero mixed difference of F(z^K) ^ O.

    On such a rectangle every bit-local value of z moves by +-h, +-k or 0
    independently, so any integer polynomial Q in z and bit-local values is a
    polynomial in (alpha*h, beta*k) and its mixed difference is divisible by
    h*k = 2^width. Hence no program of carry_free_class computes F(z^K) ^ O for any
    K, O; with AND/OR as the last step, bijectivity forces the identity constant."""
    mask = (1 << width) - 1
    fv = np.array([middle(y, mask) for y in range(1 << width)], dtype=np.int64)
    os = np.arange(1 << width, dtype=np.int64)
    cert = {}
    for key in range(1 << width):
        left, rects = np.ones(1 << width, dtype=bool), []
        for x in range(1 << width):
            for i in range(1, width):
                j = width - i
                if x >> i & 1 or x >> j & 1:
                    continue
                h, k = 1 << i, 1 << j
                v = [fv[p ^ key] ^ os for p in (x, x | h, x | k, x | h | k)]
                hit = (v[3] - v[1] - v[2] + v[0]) & mask != 0
                if (hit & left).any():
                    rects.append((x, i))
                    left &= ~hit
            if not left.any():
                break
        if left.any():
            return None
        cert[key] = rects
    return cert


def bitlocal_witness(key):
    """Flipping input bit 0 changes target bits above bit 0 (z = K gives y = 0, 1)."""
    d = target(key, key, MASK) ^ target(key ^ 1, key, MASK)
    return d & ~1


LIFT_SRC = Path(__file__).with_name('cegis_middle_c5_lift863.c')
RESULTS = Path(__file__).with_name('cegis_middle_c5_863_results.jsonl.gz')
LIFTS = Path(__file__).with_name('cegis_middle_c5_lift863_results.jsonl.gz')
AUDIT = Path(__file__).with_name('cegis_middle_c5_863_audit.json')


def lift_binary(out=None):
    import subprocess
    import tempfile
    out = out or Path(tempfile.gettempdir()) / 'cegis_middle_c5_lift863'
    if not out.exists() or out.stat().st_mtime < LIFT_SRC.stat().st_mtime:
        subprocess.run(['cc', '-O2', '-o', str(out), str(LIFT_SRC)], check=True)
    return out


def lift_c(order, wiring, width=12, pins=None, d=None):
    """Run the C depth-first lifter; returns result, refuting level, nodes per level."""
    import subprocess
    spec = ';'.join(('m' if op == 'mad' else op) + ':' + ','.join(map(str, a))
                    for op, a in zip(order, wiring))
    extra = [f'{i}.{s}={v}' for (i, s), v in sorted((pins or {}).items())]
    if d is not None:
        extra.append(f'D={d}')
    out = subprocess.run([str(lift_binary()), str(width), spec] + extra,
                         capture_output=True, text=True, check=True).stdout.split()
    return dict(result=out[0], level=int(out[1].split('=')[1]),
                nodes=[int(v) for v in out[2].split('=')[1].split(',')])


def lift_job(args):
    order, wiring = args
    pins = symmetry_pins(order, wiring)
    row = lift_c(order, wiring, pins=pins)
    row.update(order=','.join(order), wiring=[list(w) for w in wiring],
               pins=sorted([i, s, v] for (i, s), v in pins.items()))
    return row


def lift_closes(order, wiring, row):
    """A lift record proves the wiring only if it is unsat and every pin it used is
    one of the constructive symmetry pins."""
    if not row or row['result'] != 'unsat':
        return False
    pins = symmetry_pins(order, wiring)
    return all(pins.get((i, s)) == v for i, s, v in row['pins'])


def audit():
    """Every canonical mixed wiring: closed by the CEGIS run, by the carry-free
    certificate, or by the exact lifter; anything else is open."""
    with gzip.open(RESULTS, 'rt') as f:
        rows = [json.loads(line) for line in f]
    with gzip.open(LIFTS, 'rt') as f:
        lifts = {(r['order'], json.dumps(r['wiring'])): r for r in map(json.loads, f)}
    how, opened = Counter(), []
    for r in rows:
        o, w = tuple(r['order'].split(',')), tuple(map(tuple, r['wiring']))
        if r['result'] == 'unsat':
            how['cegis'] += 1
        elif carry_free_class(o, w):
            how['carry-free-certificate'] += 1
        elif lift_closes(o, w, lifts.get((r['order'], json.dumps(r['wiring'])))):
            how['lift'] += 1
        else:
            opened.append([r['order'], r['wiring']])
    first = Counter(r['ladder'][-1][0] for r in rows if r['result'] == 'unsat')
    return dict(wirings=len(rows), closed=dict(how), cegis_first_unsat_width=dict(first),
                lift_levels=dict(Counter(r['level'] for r in lifts.values())),
                open=opened)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mads', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--ladder', default=','.join(map(str, LADDER)))
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--sample', type=int, default=0, help='Random sample of the jobs')
    ap.add_argument('--orders', default='', help='Semicolon list of comma orders')
    ap.add_argument('--lift', type=Path, help='Run the exact lifter on the open rows of this file')
    ap.add_argument('--audit', action='store_true')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if args.audit:
        AUDIT.write_text(json.dumps(audit(), indent=1) + '\n')
        print(AUDIT.read_text())
        return
    if args.lift:
        with gzip.open(args.lift, 'rt') as f:
            rows = [json.loads(line) for line in f]
        todo = [(tuple(r['order'].split(',')), tuple(map(tuple, r['wiring']))) for r in rows
                if r['result'] != 'unsat']
        todo = [t for t in todo if not carry_free_class(*t)]
        print(json.dumps(dict(lift=len(todo))), flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool, \
                gzip.open(args.output, 'wt') as out:
            for row in pool.map(lift_job, todo):
                out.write(json.dumps(row) + '\n')
                print(json.dumps(dict(order=row['order'], result=row['result'],
                                      level=row['level'])), flush=True)
        return
    ladder = tuple(int(w) for w in args.ladder.split(','))
    chosen = [tuple(t.split(',')) for t in args.orders.split(';')] if args.orders else \
        [o for m in args.mads for o in orders(m)]
    jobs = [(o, w, ladder, args.seconds) for o in chosen for w in wirings_general(o)]
    if args.sample:
        jobs = random.Random(863).sample(jobs, args.sample)
    if args.limit:
        jobs = jobs[:args.limit]
    print(json.dumps(dict(jobs=len(jobs), ladder=ladder)), flush=True)
    counts = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool, \
            gzip.open(args.output, 'wt') as out:
        futures = [pool.submit(job, j) for j in jobs]
        for k, f in enumerate(as_completed(futures), 1):
            row = f.result()
            counts[row['result']] += 1
            if row['result'] != 'unsat':
                print(json.dumps(row), flush=True)
            out.write(json.dumps(dict(order=','.join(row['order']), wiring=row['wiring'],
                                      ladder=row['ladder'], result=row['result'],
                                      **{x: row[x] for x in ('consts', 'key') if x in row})) + '\n')
            if k % 5000 == 0 or k == len(jobs):
                print(json.dumps(dict(done=k, counts=dict(counts))), flush=True)


if __name__ == '__main__':
    main()
