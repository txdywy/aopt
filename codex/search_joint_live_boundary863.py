"""Search four-op root-boundary programs without a free final-MAD result.

Inputs [z,u,v,b,prior,a,a>>19,a XOR C1] are real earlier words. The target
includes e=9*z+C4, its final fold, root XOR, and next MAD (five original
ops). Exactly one literal SHR16 and one to three MADs; remaining ops are
XOR/AND/OR. Manifests deliberately sample each order and are not complete.
Fixed-root coefficients are all free. A positive fixed-root formula is
not an executable kernel until generalized and integrated.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from itertools import permutations, product
import gzip
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import z3

from codex.cegis_c1_absorb863 import _instr_options
from codex.search_early_boundary863 import early_inputs, NODE, NAMES as EARLY_NAMES
from codex.search_live_boundary863 import expression, primary_path_avoiding
from codex.search_root_boundary863 import target, MASK, C0
from codex.scout_joint_wirings863 import write_report


NAMES = EARLY_NAMES[1:]
C4 = 0xFD7046C5


def inputs(b, node=None, runtime_node=False):
    values = early_inputs(b)[1:]
    return values+[node] if runtime_node else values


def wanted(values, node):
    return target((9*values[0]+C4) & MASK, node)


@lru_cache(maxsize=1)
def initial_truth_tables():
    points = [inputs(p) for p in (0, 1, 1 << 19, (1 << 19) | 1)]
    return [sum((v[j] & 1) << (2*i+q) for i, v in enumerate(points) for q in (0, 1))
            for j in range(len(NAMES))]


def parity_models(row):
    n = len(NAMES)
    used = [(i, j-n-i) for i, w in enumerate(row['wiring']) for j in set(w) if j >= n+i]
    valid = []
    for bits in product((0, 1), repeat=len(used)):
        constants = [[0, 0] for _ in row['order']]
        for (i, j), bit in zip(used, bits):
            constants[i][j] = 255*bit
        values = list(initial_truth_tables())
        for op, w, cc in zip(row['order'], row['wiring'], constants):
            pool = values+cc
            if op == 'shr16':
                v = 0xAA
            else:
                a, b = pool[w[0]], pool[w[1]]
                v = ((a & b) ^ pool[w[2]] if op == 'mad' else
                     a ^ b if op == '^' else a & b if op == '&' else a | b)
            values.append(v)
        if (values[-1] ^ (values[-1] >> 1)) & 0x55 == 0x55:
            valid.append([(i, j, bit) for (i, j), bit in zip(used, bits)])
    return valid


def make_manifest(per_order, seed=8642834):
    rng = random.Random(seed)
    rows, stats, seen = [], [], set()
    n = len(NAMES)
    for mads in (1, 2, 3):
        for bits in product(('^', '&', '|'), repeat=3-mads):
            for order in sorted(set(permutations(('shr16', *('mad',)*mads, *bits)))):
                if order in seen or order[-1] == 'shr16':
                    continue
                seen.add(order)
                options = [[(j,) for j in range(n+i)] if op == 'shr16' else _instr_options(n+i-1, op)
                           for i, op in enumerate(order)]
                selected = set(); attempts = 0
                while len(selected) < per_order and attempts < 100000:
                    attempts += 1
                    w = tuple(rng.choice(oo) for oo in options)
                    if all(any(n+j in w[i] for i in range(j+1, 4)) for j in range(3)):
                        selected.add(w)
                rows.extend(dict(order=list(order), wiring=[list(ww) for ww in w]) for w in sorted(selected))
                stats.append(dict(order=list(order), selected=len(selected), attempts=attempts))
    return rows, stats


def affine_tail_manifest():
    """Complete MAD,SHR16,XOR,affine-MAD subfamily, not all four-op graphs."""
    rows = []
    for first in _instr_options(7, 'mad'):
        # t8=MAD(initials/constants), q9=t8>>16, bit10=q9 XOR h.
        for h in (*range(9), 10):
            rows.append(dict(order=['mad', 'shr16', '^', 'mad'],
                             wiring=[list(first), [8], sorted([9, h]), [10, 11, 12]]))
    return rows, [dict(order=['mad', 'shr16', '^', 'mad'], cases=len(rows),
                       complete_only_this_affine_tail_family=True)]


@lru_cache(maxsize=2)
def sample_data(dense=False):
    rng = random.Random(8642835)
    ps = list(dict.fromkeys([*range(16), MASK, MASK-1, 65535, 65536, 0xffff0000,
                            *(1 << j for j in range(32)), *(rng.getrandbits(32) for _ in range(16))]))
    if dense:
        bases = [0, 1, 7, 31, 65535, 1 << 19, 0xfff8ffff,
                 *(rng.getrandbits(32) & ~(7 << 16) for _ in range(17))]
        ps = list(dict.fromkeys(ps+[base | (j << 16) for base in bases for j in range(8)]))
    tests = list(dict.fromkeys(ps+[rng.getrandbits(32) for _ in range(4096)]))
    xs = np.array(tests, dtype=np.uint64)
    with np.errstate(over='ignore'):
        values = inputs(xs)
    return ps, tests, [inputs(p) for p in ps], values


def solve(row, node, seconds, pins=(), dense=False, extra_points=()):
    start = time.monotonic(); n = len(NAMES)
    ks = [z3.BitVecs(f'joint_{i}a joint_{i}b', 32) for i in range(len(row['order']))]
    for i, op in enumerate(row['order']):
        if op == 'shr16' and not primary_path_avoiding(row, n, False, n+i):
            return dict(row, status='shift_wordcut_excluded', seconds=0,
                        certificate='SHR16 is a primary-data word cut; target is a permutation')
    low = parity_models(row)
    if not low:
        return dict(row, status='lsb_dependency_excluded', seconds=0,
                    certificate='Shifted bit0 cannot affect output in every b0/b19 class')
    s = z3.SolverFor('QF_BV')
    last = len(row['order'])-1
    inverse_tail = row['order'][-1] == 'mad' and row['wiring'][-1] == [n+last-1, n+last, n+last+1]
    inverse_a, inverse_b = z3.BitVecs('joint_inverse_tail_A joint_inverse_tail_B', 32)
    if inverse_tail:
        # Exact reparameterization: G*v+H=T, G odd, iff v=A*T+B,
        # A=G^-1 and B=-A*H. T is concrete on synthesis samples, removing
        # the expensive product of two independent unknown expressions.
        s.add(inverse_a & 1 == 1)
    s.add(z3.Or(*(z3.And(*(ks[i][j] & 1 == v for i, j, v in pattern
                          if not inverse_tail or i != last)) for pattern in low)))
    for i, j, value in pins:
        if inverse_tail and i == last:
            if j == 0:
                assert value & 1
                s.add(inverse_a == pow(value, -1, 1 << 32))
            else:
                s.add(inverse_b+inverse_a*value == 0)
        else:
            s.add(ks[i][j] == (value & MASK))
    prefix = dict(order=row['order'][:-1], wiring=row['wiring'][:-1])
    first_t_wordcut = (row['order'][0] == 'mad' and
                      all(j >= n or j < 4 for j in row['wiring'][0]) and
                      not primary_path_avoiding(row, n, False, n))
    if first_t_wordcut:
        # Only z/u/v/b are T functions of b. This condition is deliberately
        # withheld whenever prior/a/s19/mix or any genuine input bypass is read.
        first = dict(order=row['order'][:1], wiring=row['wiring'][:1])
        low_images = [expression(first, ks[:1], inputs(p)) & 3 for p in range(4)]
        # A MAD reading only existing words has no symbolic operand on a
        # concrete synthesis sample. Distinct still needs typed BV values.
        s.add(z3.Distinct(*(v if z3.is_expr(v) else z3.BitVecVal(v, 32) for v in low_images)))
    def add(v):
        if inverse_tail:
            s.add(expression(prefix, ks[:-1], v) == inverse_a*wanted(v, node)+inverse_b)
        else:
            s.add(expression(row, ks, v) == wanted(v, node))
    initial, tests, vv, values = sample_data(dense)
    if extra_points:
        assert all(isinstance(p, int) and 0 <= p <= MASK for p in extra_points)
        initial = list(dict.fromkeys([*initial, *extra_points]))
        vv = [inputs(p) for p in initial]
    points = list(initial)
    for v in vv:
        add(v)
    with np.errstate(over='ignore'):
        expected = wanted(values, np.uint64(node))
    report = dict(row, node=node, status='unknown', events=[], pinned_constants=[list(p) for p in pins],
                  full32_semantics=True, free_e=False, first_prefix_mod4_guard_used=first_t_wordcut,
                  inverse_affine_tail_normalization=inverse_tail,
                  dense_fibres=dense,
                  extra_primary_points=list(extra_points),
                  new_executable=False)
    while time.monotonic()-start < seconds:
        s.set(timeout=max(1, int((seconds-time.monotonic()+start)*1000)))
        status = s.check(); report['events'].append(dict(samples=len(points), status=str(status)))
        if status != z3.sat:
            report['status'] = str(status)
            if status == z3.unknown:
                report['reason'] = s.reason_unknown()
            break
        model = s.model()
        cc = [[model.eval(k, model_completion=True).as_long() for k in pair] for pair in ks]
        if inverse_tail:
            ai = model.eval(inverse_a, model_completion=True).as_long()
            bi = model.eval(inverse_b, model_completion=True).as_long()
            g = pow(ai, -1, 1 << 32)
            cc[-1] = [g, (-g*bi) & MASK]
        with np.errstate(over='ignore'):
            bad = np.flatnonzero(expression(row, cc, values) != expected)
        if len(bad):
            take = [tests[i] for i in bad[:8]]
        else:
            b = z3.BitVec('joint_counterexample_b', 32); v = inputs(b)
            ck = [[z3.BitVecVal(k, 32) for k in pair] for pair in cc]
            check = z3.SolverFor('QF_BV')
            check.set(timeout=max(1, int((seconds-time.monotonic()+start)*1000)))
            check.add(expression(row, ck, v) != wanted(v, node))
            verified = check.check()
            if verified == z3.unsat:
                report.update(status='proved_equivalent', constants=cc,
                              macro=dict(row, constants=cc, live='joint_early', runtime_node=False))
                break
            if verified == z3.unknown:
                report.update(status='verification_unknown', reason=check.reason_unknown())
                break
            take = [check.model().eval(b, model_completion=True).as_long()]
        for p in take:
            assert p not in points
            points.append(p); v = inputs(p)
            add(v)
    report.update(seconds=round(time.monotonic()-start, 4), samples=len(points))
    return report


def controls():
    row = dict(order=['mad', 'shr16', '^', 'mad'], wiring=[[0, 8, 9], [8], [8, 9], [10, 11, 12]])
    results = [solve(row, node, 30, pins=[(0, 0, 9), (0, 1, C4), (3, 0, 4097)])
               for node in (0, 1 << 31)]
    assert all(r['status'] == 'proved_equivalent' for r in results), results
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--per-order', type=int, default=128)
    ap.add_argument('--family', choices=('sample', 'affine-tail'), default='sample')
    ap.add_argument('--seed', type=int, default=8642834)
    ap.add_argument('--seconds', type=float, default=1)
    ap.add_argument('--dense-fibres', action='store_true')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--refine-from', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); source = {}
    if args.refine_from:
        encoded = args.refine_from.read_bytes()
        old = json.loads(gzip.decompress(encoded) if args.refine_from.suffix == '.gz' else encoded)
        assert old['completed'] == len(old['manifest']) == len(old['results'])
        args.per_order, args.seed = old['per_order'], old['seed']
        args.family = old.get('family', 'sample')
        args.dense_fibres = args.dense_fibres or old.get('dense_fibres', False)
        rows = [{k:r[k] for k in ('order', 'wiring')} for r in old['results']
                if r['status'] not in ('unsat', 'shift_wordcut_excluded', 'lsb_dependency_excluded', 'proved_equivalent')]
        stats = old['orders']
        source = dict(source=str(args.refine_from), source_sha256=hashlib.sha256(encoded).hexdigest())
    else:
        rows, stats = affine_tail_manifest() if args.family == 'affine-tail' else make_manifest(args.per_order, args.seed)
    report = dict(node=NODE, input_names=NAMES, free_e=False, sampled_not_exhaustive=args.family == 'sample',
                  family=args.family, no_claim_of_complete_four_op_grammar=True,
                  dense_fibres=args.dense_fibres,
                  per_order=args.per_order, seed=args.seed, manifest=rows, orders=stats,
                  controls=controls(), results=[], completed=0, counts={},
                  seconds_per_case=args.seconds, full32_semantics=True,
                  best_executable_cycles=864, new_executable=False, **source)
    write_report(args.output, report); print('cases', len(rows), 'family', args.family, flush=True)
    started = time.monotonic(); counts = Counter()
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(solve, row, NODE, args.seconds, (), args.dense_fibres) for row in rows]
        for done, f in enumerate(as_completed(futures), 1):
            r = f.result(); report['results'].append(r); counts[r['status']] += 1
            if r['status'] == 'proved_equivalent':
                print(r, flush=True)
            if done % 1000 == 0 or done == len(rows):
                report.update(completed=done, counts=dict(counts), seconds=round(time.monotonic()-started, 3))
                write_report(args.output, report); print(done, dict(counts), report['seconds'], flush=True)


if __name__ == '__main__':
    main()
