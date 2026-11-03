"""Full32 root-boundary synthesis with four earlier, non-T live words.

The new initial words are prior, a, a>>19 and a XOR C1. In terms of b,
they include information from higher bits, so the old first-prefix mod4
and five-word low-bit guards must not be reused without proof.

The default complete first-early-SHR grammar has one literal SHR16 of an
earlier word, one MAD and one XOR/AND/OR. --all-orders also includes a MAD
or bitwise first instruction and shifts of new results, requiring a real
earlier-word reference. Fixed-node constants may be prepared freely.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from itertools import combinations_with_replacement, permutations, product
import hashlib
import gzip
import json
from pathlib import Path
import random
import time

import numpy as np
import z3

from codex.cegis_c1_absorb863 import _instr_options
from codex.scout_joint_wirings863 import write_report
from codex.search_live_boundary863 import inputs, expression, target, MASK, C0, C5
from codex.search_full_live_shift863 import prior_for_b
from codex.screen_xor_mad_polynomial import consistent
from tests.frozen_problem import Tree, Input, reference_kernel, myhash, myhash_traced


NAMES = ('e', 'z', 'u', 'v', 'b', 'prior', 'a', 'a_shr19', 'a_xor_c1')
NODE = 0x12345678 ^ C5
INV4097 = pow(4097, -1, 1 << 32)
C1 = 0xC761C23C


def shr(value, count):
    return z3.LShR(value, count) if z3.is_bv(value) else value >> count


def early_inputs(b):
    vv = inputs(b, NODE, 'arms', False)
    folded = b ^ C1
    a = folded ^ shr(folded, 19)
    prior = ((a - C0) * INV4097) & MASK
    return vv + [prior, a, shr(a, 19), a ^ C1]


def wanted(values, control=False):
    if control:
        return (33 * (shr(values[5], 16) ^ values[6]) + 17) & MASK
    return target(values[0], NODE)


@lru_cache(maxsize=1)
def sample_data():
    rng = random.Random(8642822)
    points = list(dict.fromkeys([*range(16), MASK, MASK-1, 0xffff, 0x10000,
                                0xffff0000, *(1 << j for j in range(32)),
                                *(rng.getrandbits(32) for _ in range(16))]))
    scouts = list(dict.fromkeys(points + [rng.getrandbits(32) for _ in range(2048)]))
    return points, scouts, [early_inputs(b) for b in points], np.array(scouts, dtype=np.uint64)


def lsb_constraints(row, ks):
    """All initial bit0 values ignore b16; the target bit0 flips on b16.

    The nine initial LSBs depend only on b0,b19. Check all four classes,
    retaining the real early inputs rather than treating them as T words.
    """
    conditions = []
    kk = [[z3.Extract(0, 0, k) for k in pair] for pair in ks]
    for p in (0, 1, 1 << 19, (1 << 19) | 1):
        vv = [z3.BitVecVal(v & 1, 1) for v in early_inputs(p)]
        outs = []
        for qbit in (0, 1):
            vals = list(vv)
            for op, w, cc in zip(row['order'], row['wiring'], kk):
                pool = vals + cc
                if op == 'shr16':
                    value = z3.BitVecVal(qbit, 1)
                else:
                    a, b = pool[w[0]], pool[w[1]]
                    value = (a * b + pool[w[2]] if op == 'mad' else
                             a ^ b if op == '^' else a & b if op == '&' else a | b)
                vals.append(value)
            outs.append(vals[-1])
        conditions.append(outs[0] != outs[1])
    return conditions


def make_manifest(all_orders=False):
    rows, stats = [], []
    n = len(NAMES)
    for bit in ('^', '&', '|'):
        orders = (sorted(set(permutations(('mad', 'shr16', bit)))) if all_orders else
                  [('shr16', 'mad', bit), ('shr16', bit, 'mad')])
        for order in orders:
            if order[-1] == 'shr16':
                continue
            order = list(order)
            options = [[(j,) for j in range(n+i)] if op == 'shr16' else _instr_options(n+i-1, op)
                       for i, op in enumerate(order)]
            if not all_orders:
                options[0] = [(j,) for j in range(5, n)]
            count = 0
            for w in product(*options):
                if not all(any(n+j in w[i] for i in range(j+1, 3)) for j in range(2)):
                    continue
                if not any(j in range(5, n) for ww in w for j in ww):
                    continue
                rows.append(dict(order=order, wiring=[list(ww) for ww in w]))
                count += 1
            stats.append(dict(order=order, cases=count))
    return rows, stats


def solve(row, seconds, control=False):
    started = time.monotonic()
    ks = [z3.BitVecs(f'early_{i}a early_{i}b', 32) for i in range(len(row['order']))]
    solver = z3.SolverFor('QF_BV')
    if not control:
        needed = lsb_constraints(row, ks)
        low = z3.SolverFor('QF_BV'); low.add(*needed)
        if low.check() == z3.unsat:
            return dict(row, status='lsb_dependency_excluded', seconds=0,
                        certificate='Shifted bit cannot reach bit0 for all four b0/b19 classes')
        solver.add(*needed)
    initial, scouts, vv, xs = sample_data()
    points = list(initial)
    for v in vv:
        solver.add(expression(row, ks, v) == wanted(v, control))
    with np.errstate(over='ignore'):
        values = early_inputs(xs); want = wanted(values, control)
    report = dict(row, status='unknown', control=control, events=[],
                  full32_semantics=True, private_constants_arbitrary=True,
                  prefix_mod4_guard_used=False, new_executable=False)
    while time.monotonic()-started < seconds:
        solver.set(timeout=max(1, int((seconds-time.monotonic()+started)*1000)))
        status = solver.check()
        report['events'].append(dict(samples=len(points), status=str(status)))
        if status != z3.sat:
            report['status'] = str(status)
            if status == z3.unknown:
                report['reason'] = solver.reason_unknown()
            break
        model = solver.model()
        cc = [[model.eval(k, model_completion=True).as_long() for k in pair] for pair in ks]
        with np.errstate(over='ignore'):
            bad = np.flatnonzero(expression(row, cc, values) != want)
        if len(bad):
            take = [scouts[i] for i in bad[:8]]
        else:
            b = z3.BitVec('early_primary', 32); v = early_inputs(b)
            ck = [[z3.BitVecVal(k, 32) for k in pair] for pair in cc]
            check = z3.SolverFor('QF_BV')
            check.set(timeout=max(1, int((seconds-time.monotonic()+started)*1000)))
            check.add(expression(row, ck, v) != wanted(v, control))
            checked = check.check()
            if checked == z3.unsat:
                report.update(status='proved_equivalent', constants=cc,
                              macro=dict(row, constants=cc, live='early', runtime_node=False))
                break
            if checked == z3.unknown:
                report.update(status='verification_unknown', reason=check.reason_unknown())
                break
            take = [check.model().eval(b, model_completion=True).as_long()]
        for p in take:
            assert p not in points
            points.append(p); v = early_inputs(p)
            solver.add(expression(row, ks, v) == wanted(v, control))
    report.update(samples=len(points), seconds=round(time.monotonic()-started, 4))
    return report


@lru_cache(maxsize=1)
def quarter_witnesses():
    raw = NODE ^ C5
    witnesses = {}
    for initial in range(64):
        value, index = initial, 0
        for _ in range(10):
            value = myhash(value ^ (raw if index == 0 else 0))
            index = 2*index + (1 if value % 2 == 0 else 2)
        witnesses.setdefault(value >> 30, (initial, value, index))
    assert set(witnesses) == {0, 1, 2, 3}
    return witnesses


@lru_cache(maxsize=None)
def legal_point(b):
    """Construct a valid 30-bit tree/initial value with the requested b."""
    p = prior_for_b(b)
    assert early_inputs(b)[5] == p
    raw = NODE ^ C5
    witnesses = quarter_witnesses()
    initial, before, index = witnesses[p >> 30]
    leaf = p ^ before
    assert leaf < 1 << 30
    nodes = [0]*2047; nodes[0] = raw; nodes[index] = leaf
    inp = Input([0], [initial], 11); reference_kernel(Tree(10, nodes), inp)
    trace = {}; out = myhash_traced(p, trace, 10, 0)
    vv = early_inputs(b)
    assert trace[(10, 0, 'hash_stage', 0)] == vv[6]
    assert trace[(10, 0, 'hash_stage', 1)] == b
    assert trace[(10, 0, 'hash_stage', 3)] == vv[1]
    assert trace[(10, 0, 'hash_stage', 4)] == vv[0]
    assert inp.values[0] == out and inp.indices[0] == 0
    trace = {}; myhash_traced(out ^ raw, trace, 11, 0)
    assert trace[(11, 0, 'hash_stage', 0)] == wanted(vv)
    return dict(primary_b=b, prior=p, values=vv, target=wanted(vv),
                legal_tree=dict(height=10, root=raw, initial=initial,
                                leaf_index=index, leaf_value=leaf, other_nodes=0),
                frozen_reference_checked=True)


def dependency_proofs():
    # Same nine highest bits, different target bit31. Every fixed right
    # shift has top bit zero (or preserves it for shift0), so any finite
    # expression over fixed right shifts and bitwise Boolean operations
    # preserves this indistinguishability. No constant or node preparation
    # can depend on b in the scoped macro.
    first, second = [legal_point(b) for b in (659915343, 7650447)]
    signature = [v >> 31 for v in first['values']]
    assert signature == [v >> 31 for v in second['values']]
    assert (first['target'] ^ second['target']) >> 31 == 1
    b = z3.BitVec('early_dependency_b', 32)
    vv, toggled = early_inputs(b), early_inputs(b ^ (1 << 16))
    facts = []
    for i, name in enumerate(NAMES):
        s = z3.SolverFor('QF_BV'); s.set(timeout=10000)
        s.add(z3.Extract(0, 0, vv[i]) != z3.Extract(0, 0, toggled[i]))
        assert s.check() == z3.unsat, name
        facts.append(dict(name=name+'_lsb_ignores_b16', status='unsat'))
        sample = early_inputs(b & ((1 << 19) | 1))[i]
        s = z3.SolverFor('QF_BV'); s.set(timeout=10000)
        s.add(z3.Extract(0, 0, vv[i]) != z3.Extract(0, 0, sample))
        assert s.check() == z3.unsat, name
        facts.append(dict(name=name+'_lsb_depends_only_on_b0_b19', status='unsat'))
    s = z3.SolverFor('QF_BV'); s.set(timeout=10000)
    s.add(z3.Extract(0, 0, wanted(vv)) == z3.Extract(0, 0, wanted(toggled)))
    assert s.check() == z3.unsat
    facts.append(dict(name='target_lsb_flips_on_b16', status='unsat'))
    return dict(no_mad_fixed_right_shift=dict(first=first, second=second, top_signature=signature,
                excluded='Any finite expression of fixed right shifts and bitwise Boolean operations over these nine words',
                excludes_variable_shifts=False, excludes_left_shifts=False),
                no_new_downward_bit_transfer=dict(full32_facts=facts,
                excluded='Any finite expression of MAD/ADD/SUB/MUL/bitwise and fixed left shifts over these words; output bit0 cannot depend on b16'),
                lsb_guard_checked_for_all_primary_inputs=True)


def polynomial_proofs():
    """Two MADs after a shift of an initial word: degree-four relaxation.

    All nine initial words are affine in b,z,a,a>>19,a XOR C1. Append q,
    the already computed (full32 source)>>16. This gives 210 independent
    integer monomials, including every possible two-MAD expression.
    """
    rng = random.Random(8642821)
    bs = list(dict.fromkeys([*range(64), MASK, MASK-1, *(1 << j for j in range(32)),
                            *(rng.getrandbits(32) for _ in range(512))]))
    vv = [early_inputs(b) for b in bs]
    monomials = [()] + [m for d in range(1, 5) for m in combinations_with_replacement(range(6), d)]
    rhs = [wanted(v) for v in vv]
    rows = []
    for source, name in enumerate(NAMES):
        start = time.monotonic(); matrix = []
        for b, v in zip(bs, vv):
            fields = [b, v[1], v[6], v[7], v[8], v[source] >> 16]
            row = []
            for monomial in monomials:
                value = 1
                for j in monomial:
                    value = (value * fields[j]) & MASK
                row.append(value)
            matrix.append(row)
        compatible = consistent(matrix, rhs, 32)
        # Actual first-SHR, MAD, MAD program with non-T a as a bypass.
        controls = [(((b*(v[source] >> 16) + v[1]) & MASK)**2 + v[6]) & MASK
                    for b, v in zip(bs, vv)]
        assert consistent(matrix, controls, 32)
        rows.append(dict(source_index=source, source_name=name,
                         status='relaxed_survivor' if compatible else 'inconsistent_full32',
                         two_mad_positive_control=True,
                         seconds=round(time.monotonic()-start, 4)))
    return dict(node=NODE, width=32, samples=bs, variables=['b', 'z', 'a', 'a_shr19', 'a_xor_c1', 'q'],
                monomial_indices=monomials, arbitrary_independent_coefficients=len(monomials),
                results=rows, all_sources_excluded=all(r['status']=='inconsistent_full32' for r in rows),
                scope='One literal SHR16 of an initial word, two other MADs. No shift of a new MAD result.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--proofs-only', action='store_true')
    ap.add_argument('--all-orders', action='store_true')
    ap.add_argument('--reuse-first-shift', type=Path)
    ap.add_argument('--seconds', type=float, default=1)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--refine-from', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.proofs_only:
        result = dict(dependencies=dependency_proofs(), polynomial=polynomial_proofs(),
                      node=NODE, input_names=NAMES, best_executable_cycles=864, new_executable=False)
        write_report(args.output, result)
        print('Dependency certificates valid; polynomial sources:', result['polynomial']['results'], flush=True)
        return
    source = {}
    if args.refine_from:
        encoded = args.refine_from.read_bytes()
        old = json.loads(gzip.decompress(encoded) if args.refine_from.suffix == '.gz' else encoded)
        assert old['completed'] == len(old['manifest']) == len(old['results'])
        rows = [{k:r[k] for k in ('order', 'wiring')} for r in old['results']
                if r['status'] not in ('unsat', 'lsb_dependency_excluded', 'prior_excluded', 'proved_equivalent')]
        stats = old['orders']
        args.all_orders = old.get('all_orders', False)
        source = dict(source=str(args.refine_from), source_sha256=hashlib.sha256(encoded).hexdigest())
    else:
        rows, stats = make_manifest(args.all_orders)
    control = solve(dict(order=['shr16', '^', 'mad'], wiring=[[5], [6, 9], [10, 11, 12]]), 30, True)
    assert control['status'] == 'proved_equivalent', control
    reused = []
    if args.reuse_first_shift:
        assert args.all_orders and not args.refine_from
        encoded = args.reuse_first_shift.read_bytes()
        old = json.loads(gzip.decompress(encoded) if args.reuse_first_shift.suffix == '.gz' else encoded)
        assert old['completed'] == len(old['manifest']) == len(old['results']) == 74100
        assert old['node'] == NODE and old['input_names'] == list(NAMES)
        assert not old['sampled_not_exhaustive'] and old['full32_semantics']
        assert all(r['status'] in ('unsat', 'lsb_dependency_excluded') for r in old['results'])
        expected, _ = make_manifest()
        key = lambda r: (tuple(r['order']), tuple(tuple(w) for w in r['wiring']))
        assert {key(r) for r in old['manifest']} == {key(r) for r in old['results']} == {key(r) for r in expected}
        reused = [dict(r, status='prior_excluded', prior_status=r['status']) for r in old['results']]
        source.update(reused_first_shift=str(args.reuse_first_shift),
                      reused_first_shift_sha256=hashlib.sha256(encoded).hexdigest())
    report = dict(node=NODE, input_names=NAMES, manifest=rows, orders=stats,
                  results=list(reused), controls=[control], completed=len(reused), counts={},
                  full32_semantics=True, all_orders=args.all_orders,
                  first_shift_sources=list(NAMES[5:]) if not args.all_orders else None,
                  sampled_not_exhaustive=False, seconds_per_case=args.seconds,
                  best_executable_cycles=864, new_executable=False, **source)
    write_report(args.output, report); print('cases', len(rows), flush=True)
    start = time.monotonic(); counts = Counter(r['status'] for r in reused)
    key = lambda r: (tuple(r['order']), tuple(tuple(w) for w in r['wiring']))
    skip = {key(r) for r in reused}
    pending = [r for r in rows if key(r) not in skip]
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(solve, r, args.seconds) for r in pending]
        for done, f in enumerate(as_completed(futures), len(reused)+1):
            r = f.result(); report['results'].append(r); counts[r['status']] += 1
            if r['status'] == 'proved_equivalent':
                print(r, flush=True)
            if done % 5000 == 0 or done == len(rows):
                report.update(completed=done, counts=dict(counts), seconds=round(time.monotonic()-start, 3))
                write_report(args.output, report); print(done, dict(counts), report['seconds'], flush=True)


if __name__ == '__main__':
    main()
