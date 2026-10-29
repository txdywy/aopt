"""Sample four-op C5 absorption with real earlier inputs and a free C1 key.

Unlike the one-input restore grammar, inputs are a, prior and the changed
preceding-XOR result. All target values retain the original a relation.
Low-width UNSAT is necessary for full32; reduced SAT is never a kernel.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
from itertools import combinations_with_replacement, product
import json
from pathlib import Path
import random
import time

import z3

from codex.cegis_c1_absorb863 import _instr_options
from codex.scout_joint_wirings863 import write_report
from codex.search_cross_encoding863 import PRE_C5, necessary_inputs
from codex.search_keyed_live_middle863 import C1, live_values, reference, original_program
from codex.scan_keyed_live_nonlinear863 import first_forms


OPS = ('mad', '^', '&', '|')


def evaluate(row, constants, a, key, width):
    mask = (1 << width)-1
    values = live_values(a, key, width)
    for op, w, cc in zip(row['order'], row['wiring'], constants):
        pool = values+list(cc)
        x, y = pool[w[0]], pool[w[1]]
        out = x*y+pool[w[2]] if op == 'mad' else x^y if op == '^' else x&y if op == '&' else x|y
        values.append(out & mask)
    return values[-1]


def validate(row):
    if len(row['order']) != len(row['wiring']) or not 1 <= len(row['order']) <= 4:
        raise ValueError('One to four matching T instructions required')
    for i, (op, w) in enumerate(zip(row['order'], row['wiring'])):
        if op not in OPS or len(w) != (3 if op == 'mad' else 2):
            raise ValueError('Unsupported instruction')
        if any(not 0 <= j < 5+i for j in w):
            raise ValueError('Invalid wire')


def solve(row, width=11, seconds=2, output_key=PRE_C5, pins=(), key_pin=None, control=None):
    validate(row)
    if width not in (*range(2, 17), 32) or seconds <= 0:
        raise ValueError('Width2..16/32 and a positive time limit required')
    started = time.monotonic()
    mask = (1 << width)-1
    constants = [z3.BitVecs(f'klr_k{i}a klr_k{i}b', width) for i in range(len(row['order']))]
    key = z3.BitVec('klr_input_key', width)
    solver = z3.SolverFor('QF_BV')
    for i, j, v in pins:
        solver.add(constants[i][j] == (v & mask))
    if key_pin is not None:
        solver.add(key == (key_pin & mask))

    def target(a):
        base = reference(a, width) if control is None else evaluate(
            control['row'], control['constants'], a, control['key'], width)
        return (base ^ (output_key & mask)) & mask

    points = necessary_inputs(width if width < 32 else 16)
    points = list(dict.fromkeys([*(v & mask for v in points), *([1 << i for i in range(32)] if width == 32 else [])]))
    def constrain(a):
        solver.add(evaluate(row, constants, z3.BitVecVal(a, width), key, width) == target(a))
    for a in points:
        constrain(a)
    rng = random.Random(8642941)
    checks = range(1 << width) if width < 32 else list(dict.fromkeys(
        [*points, 0xffffffff, 0x80000, 0x80001, *(rng.getrandbits(32) for _ in range(4096))]))
    result = dict(row, width=width, output_key=output_key, status='unknown', events=[],
                  input_key_arbitrary=key_pin is None, key_pin=key_pin,
                  pinned_constants=[list(p) for p in pins], control=control,
                  full32_low_a_subdomain=width < 32, new_executable=False)
    while time.monotonic()-started < seconds:
        solver.set(timeout=max(1, int(1000*(seconds-time.monotonic()+started))))
        status = solver.check()
        event = dict(samples=len(points), status=str(status))
        result['events'].append(event)
        if status != z3.sat:
            result['status'] = str(status)
            if status == z3.unknown:
                result['reason'] = solver.reason_unknown()
            break
        model = solver.model()
        cv = [[model.eval(k, model_completion=True).as_long() for k in pair] for pair in constants]
        kv = model.eval(key, model_completion=True).as_long()
        bad = []
        for a in checks:
            if evaluate(row, cv, a, kv, width) != target(a):
                bad.append(a)
                if len(bad) == 8:
                    break
        if not bad and width < 32:
            result.update(status='reduced_sat', constants=cv, input_key=kv, verified_inputs=1 << width)
            break
        if not bad:
            a = z3.BitVec('klr_verify_a', 32)
            cc = [[z3.BitVecVal(v, 32) for v in pair] for pair in cv]
            check = z3.SolverFor('QF_BV')
            check.set(timeout=max(1, int(1000*(seconds-time.monotonic()+started))))
            check.add(evaluate(row, cc, a, kv, 32) != target(a))
            checked = check.check()
            event['full32_check'] = str(checked)
            if checked == z3.unsat:
                result.update(status='proved_full32', constants=cv, input_key=kv)
                break
            if checked == z3.unknown:
                result.update(status='verification_unknown', reason=check.reason_unknown())
                break
            bad = [check.model().eval(a, model_completion=True).as_long()]
        event['counterexamples'] = bad
        for a in bad:
            assert a not in points
            points.append(a)
            constrain(a)
    result.update(seconds=round(time.monotonic()-started, 4), samples=len(points))
    return result


def manifest(per_order=8, seed=8642942):
    rng = random.Random(seed)
    rows, stats = [], []
    for rest in product(OPS, repeat=3):
        order = ('mad', *rest)
        if order.count('mad') == 4:
            continue
        options = [_instr_options(2+i, op) for i, op in enumerate(order)]
        chosen = set()
        for attempt in range(100000):
            if len(chosen) == per_order:
                break
            wiring = tuple(rng.choice(v) for v in options)
            if not all(any(3+j in wiring[i] for i in range(j+1, 4)) for j in range(3)):
                continue
            # The prior input is precisely what distinguishes this search
            # from the already closed one-input/four-op restore grammar.
            if not any(1 in w for w in wiring):
                continue
            chosen.add(wiring)
        rows.extend(dict(order=list(order), wiring=[list(w) for w in wiring]) for wiring in sorted(chosen))
        stats.append(dict(order=list(order), selected=len(chosen), attempts=attempt+1))
    return rows, stats


def parallel_manifest():
    """Two independent live MADs, a bitop, then an arbitrary affine MAD.

    The30 first shapes are a useful finite family, not a canonicalization
    proof for the complete full32 live grammar. In particular, do not
    infer full32 affine-prior equivalence from its low11 reduction.
    """
    forms = [(0,3,4),(2,3,4),*(role for _,role in first_forms())]
    assert len(forms) == len(set(forms)) == 30
    rows, stats = [], []
    for op in ('^','&','|'):
        order = ['mad','mad',op,'mad']
        for first,second in combinations_with_replacement(forms,2):
            w1 = [v+1 if v>=3 else v for v in second]
            rows.append(dict(order=order[:],wiring=[list(first),w1,[3,4],[5,6,7]]))
        stats.append(dict(order=order[:],selected=465))
    return rows, stats


def control():
    program = original_program()
    row = dict(order=[p['op'] for p in program], wiring=[p['operands'][:3 if p['op']=='mad' else 2] for p in program])
    pins = [(i,j,v) for i,p in enumerate(program) for j,v in enumerate(p['constants'])]
    full = solve(row, 32, 15, 0, pins, C1)
    assert full['status'] == 'proved_full32', full
    return full


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--per-order', type=int, default=8)
    ap.add_argument('--seed', type=int, default=8642942)
    ap.add_argument('--width', type=int, default=11)
    ap.add_argument('--seconds', type=float, default=1)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--family', choices=('sample','parallel'), default='sample')
    ap.add_argument('--source', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    provenance = {}
    if args.source:
        raw = args.source.read_bytes()
        old = json.loads(gzip.decompress(raw) if args.source.suffix == '.gz' else raw)
        assert old['completed'] == len(old['manifest']) == len(old['results'])
        rows = [{k:r[k] for k in ('order','wiring')} for r in old['results'] if r['status'] != 'unsat']
        stats = old['orders']
        args.seed, args.per_order = old['seed'], old['per_order']
        args.family = old.get('family','sample')
        provenance = dict(source=str(args.source), source_sha256=hashlib.sha256(raw).hexdigest())
    else:
        rows, stats = parallel_manifest() if args.family=='parallel' else manifest(args.per_order, args.seed)
    report = dict(width=args.width, output_key=PRE_C5, per_order=args.per_order, seed=args.seed, family=args.family,
                  prior_free_but_correlated=True, input_key_arbitrary=True,
                  target='F(a XOR (a>>19) XOR C1) XOR PRE_C5',
                  inputs=['a','prior=(a-C0)/4097','a XOR (a>>19) XOR K'],
                  sampled_not_exhaustive=True, positive_control=control(), orders=stats,
                  manifest=rows, results=[], completed=0, counts={}, new_executable=False,
                  best_executable_cycles=864, **provenance)
    started = time.monotonic()
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(solve, row, args.width, args.seconds) for row in rows]
        for done, future in enumerate(as_completed(futures), 1):
            report['results'].append(future.result())
            if report['results'][-1]['status'] in ('reduced_sat','proved_full32'):
                print(report['results'][-1], flush=True)
            if done % 25 == 0 or done == len(rows):
                report.update(completed=done, counts=dict(Counter(r['status'] for r in report['results'])),
                              seconds=round(time.monotonic()-started, 3))
                write_report(args.output, report)
                print(done, report['counts'], report['seconds'], flush=True)


if __name__ == '__main__':
    main()
