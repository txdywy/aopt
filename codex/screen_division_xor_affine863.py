"""Exact divisor screen for a three-operation live-word boundary family.

The family is q=V/D, V%D or cdiv(V,D); f=q XOR W; out=G*f+H,
with nine real live words V/W and arbitrary root-only uint32 D/G/H.
Only this topology is covered. Division after a computed prefix, extra
operations and data-dependent nonzero denominators remain outside it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from math import gcd
from pathlib import Path
import random
import time

import numpy as np
import z3

from codex.search_early_boundary863 import NAMES, NODE, MASK, early_inputs, wanted, legal_point


OPS = ('//', '%', 'cdiv')


def quotient(value, divisor, op):
    """Both NumPy uint64 and Python integers preserve the ceil carry."""
    if op == '//':
        return value // divisor
    if op == '%':
        return value % divisor
    if op == 'cdiv':
        return (value + divisor - 1) // divisor
    raise ValueError(op)


def first_divisors(n0, n1, xor16, op, direct_limit=1 << 20, divisor_limit=MASK):
    """Enumerate *all* divisors satisfying a necessary first-pair mask.

    For REM, quotient-constant intervals plus a low-bit congruence cover
    the full divisor domain. Equal-quotient intervals are skipped only
    when an exhaustive 16-bit residue calculation proves them impossible.
    """
    assert op in OPS and xor16 and 0 < divisor_limit <= MASK
    started = time.monotonic()
    high_bit = 1 << (xor16.bit_length()-1)
    stats = dict(op=op, numerator_pair=[n0, n1], required_xor16=xor16,
                 divisor_domain=[1, divisor_limit], quotient_intervals=0,
                 equal_quotient_intervals_skipped=0, divisors_evaluated=0)
    pieces = []

    def keep(ds):
        if not len(ds):
            return
        stats['divisors_evaluated'] += len(ds)
        a = quotient(np.uint64(n0), ds, op)
        b = quotient(np.uint64(n1), ds, op)
        survivors = ds[((a ^ b) & 65535) == xor16]
        if len(survivors):
            pieces.append(survivors)

    maximum = max(n0, n1)
    if n0 == n1 or maximum < high_bit:
        stats['bound_reason'] = 'equal_numerators_or_both_results_below_required_bit'
    elif op in ('//', 'cdiv'):
        bound = maximum // high_bit if op == '//' else (maximum-1) // (high_bit-1)
        bound = min(bound, divisor_limit)
        stats['complete_divisor_upper_bound'] = bound
        keep(np.arange(1, bound+1, dtype=np.uint64))
    else:
        limit = min(maximum, divisor_limit)
        # For D>max(n0,n1), both remainders are the unchanged numerators.
        assert divisor_limit <= maximum or ((n0 ^ n1) & 65535) != xor16
        stats['unchanged_remainder_tail_excluded'] = divisor_limit > maximum
        stop = min(direct_limit, limit)
        keep(np.arange(1, stop+1, dtype=np.uint64))
        delta = (n1-n0) & 65535
        residues = np.arange(65536, dtype=np.uint64)
        equal_possible = bool(np.any((residues ^ ((residues+delta) & 65535)) == xor16))
        stats['equal_quotient_low16_possible'] = equal_possible
        common_bits = (xor16 & -xor16).bit_length()-1
        modulus = 1 << common_bits
        difference = n0-n1
        current = stop+1
        while current <= limit:
            qa, qb = n0//current, n1//current
            end = min(n0//qa if qa else divisor_limit,
                      n1//qb if qb else divisor_limit, limit)
            assert end >= current
            stats['quotient_intervals'] += 1
            if qa == qb and not equal_possible:
                stats['equal_quotient_intervals_skipped'] += 1
            else:
                common = gcd(qa-qb, modulus)
                if difference % common == 0:
                    step = modulus//common
                    residue = ((difference//common)*pow((qa-qb)//common, -1, step)) % step
                    first = current+(residue-current) % step
                    keep(np.arange(first, end+1, step, dtype=np.uint64))
            current = end+1
    ds = np.unique(np.concatenate(pieces)) if pieces else np.array([], dtype=np.uint64)
    stats.update(survivors=len(ds), divisor_sha256=hashlib.sha256(ds.astype('<u4').tobytes()).hexdigest(),
                 seconds=round(time.monotonic()-started, 6))
    return ds, stats


def affine_fit(ds, values, targets, source, partner, op):
    """Exact full-word filtering; an odd target difference fixes A/B.

    out=G*f+H, G odd is equivalent to f=A*out+B with A odd. All
    divisors get their own A/B; no unjustified shared-constant constraint.
    """
    points = list(range(len(targets)))
    other = next(i for i in points[1:] if (int(targets[i])-int(targets[0])) & 1)
    inv_delta = pow((int(targets[other])-int(targets[0])) & MASK, -1, 1 << 32)
    with np.errstate(over='ignore'):
        f0 = quotient(np.uint64(values[0][source]), ds, op) ^ np.uint64(values[0][partner])
        f1 = quotient(np.uint64(values[other][source]), ds, op) ^ np.uint64(values[other][partner])
        aa = ((f1-f0)*np.uint64(inv_delta)) & MASK
        bb = (f0-aa*np.uint64(targets[0])) & MASK
        keep = (aa & 1) == 1
        ds, aa, bb = ds[keep], aa[keep], bb[keep]
        history = [dict(after='odd_inverse_multiplier', remaining=len(ds))]
        checked = []
        for i in points:
            if not len(ds):
                break
            f = quotient(np.uint64(values[i][source]), ds, op) ^ np.uint64(values[i][partner])
            keep = f == ((aa*np.uint64(targets[i])+bb) & MASK)
            ds, aa, bb = ds[keep], aa[keep], bb[keep]
            checked.append(i)
            history.append(dict(after_sample=i, remaining=len(ds)))
    return ds, aa, bb, dict(anchor_indices=[0, other], samples_checked=checked, history=history)


def verify_candidate(source, partner, op, divisor, aa, bb, seconds=30):
    b = z3.BitVec('division_xor_primary', 32); vv = early_inputs(b)
    n, d = vv[source], z3.BitVecVal(divisor, 32)
    if op == '//':
        q = z3.UDiv(n, d)
    elif op == '%':
        q = z3.URem(n, d)
    else:
        q = z3.Extract(31, 0, z3.UDiv(z3.ZeroExt(1, n)+z3.ZeroExt(1, d)-1, z3.ZeroExt(1, d)))
    solver = z3.SolverFor('QF_BV'); solver.set(timeout=int(seconds*1000))
    solver.add((q ^ vv[partner]) != aa*wanted(vv)+bb)
    status = solver.check()
    row = dict(divisor=divisor, inverse_a=aa, inverse_b=bb, status=str(status))
    if status == z3.sat:
        row['counterexample'] = legal_point(solver.model().eval(b, model_completion=True).as_long())
    elif status == z3.unknown:
        row['reason'] = solver.reason_unknown()
    else:
        g = pow(aa, -1, 1 << 32)
        row.update(multiplier=g, offset=(-g*bb) & MASK, fixed_root_only=True)
    return row


def derivative_facts():
    b = z3.BitVec('division_derivative_b', 32)
    vv, ww = early_inputs(b), early_inputs(b ^ (1 << 31))
    rows = []
    s = z3.SolverFor('QF_BV'); s.set(timeout=30000)
    # An odd affine map preserves a difference of valuation15 modulo2^16.
    s.add(z3.Extract(15, 0, wanted(vv) ^ wanted(ww)) != 0x8000)
    assert s.check() == z3.unsat
    for i, name in enumerate(NAMES):
        if i == 5:  # prior can have carries; its mask is computed per sample.
            continue
        expected = 0 if i < 5 else 0x1000
        s = z3.SolverFor('QF_BV'); s.set(timeout=30000)
        s.add(z3.Extract(15, 0, vv[i] ^ ww[i]) != expected)
        assert s.check() == z3.unsat, name
        rows.append(dict(source=i, name=name, xor16=expected, status='unsat'))
    first = [early_inputs(x) for x in (0, 1 << 31)]
    return dict(target_xor16=0x8000, full32_status='unsat', source_facts=rows,
                first_pair=[legal_point(x) for x in (0, 1 << 31)],
                first_required_xor16=[0x8000 ^ ((first[0][i] ^ first[1][i]) & 65535) for i in range(9)])


def run(output):
    start = time.monotonic(); facts = derivative_facts()
    rng = random.Random(8642850)
    bs = list(dict.fromkeys([*range(64), MASK, *(1 << j for j in range(32)),
                            *(rng.getrandbits(32) for _ in range(4096))]))
    values = [early_inputs(b) for b in bs]; targets = [wanted(v) for v in values]
    pair0, pair1 = early_inputs(0), early_inputs(1 << 31)
    caches = {}; rows = []
    for op in OPS:
        for source in range(9):
            for partner in range(9):
                required = facts['first_required_xor16'][partner]
                key = op, source, required
                if key not in caches:
                    ds, stats = first_divisors(pair0[source], pair1[source], required, op)
                    caches[key] = ds, stats
                    print(op, NAMES[source], hex(required), stats['survivors'], 'first-pair divisors', flush=True)
                ds, stats = caches[key]; ds = ds.copy()
                pair_history = []
                for b in bs[1:65]:
                    if not len(ds):
                        break
                    vv, ww = early_inputs(b), early_inputs(b ^ (1 << 31))
                    required_pair = 0x8000 ^ ((vv[partner] ^ ww[partner]) & 65535)
                    x = quotient(np.uint64(vv[source]), ds, op)
                    y = quotient(np.uint64(ww[source]), ds, op)
                    ds = ds[((x ^ y) & 65535) == required_pair]
                    pair_history.append(dict(primary_b=b, remaining=len(ds)))
                after_pairs = len(ds)
                ds, aa, bb, affine = affine_fit(ds, values, targets, source, partner, op)
                candidates = [verify_candidate(source, partner, op, int(d), int(a), int(c))
                              for d, a, c in zip(ds, aa, bb)]
                row = dict(op=op, source=source, source_name=NAMES[source], partner=partner,
                           partner_name=NAMES[partner], first_pair_key=list(key),
                           first_pair_divisors=stats['survivors'], pair_history=pair_history,
                           after_derivative_pairs=after_pairs, affine_fit=affine, candidates=candidates,
                           status='excluded' if not len(ds) else 'candidate_needs_review')
                rows.append(row)
            partial = dict(node=NODE, input_names=NAMES, derivative=facts, results=rows,
                           first_pair_screens=[v[1] for v in caches.values()], complete=False)
            output.write_text(json.dumps(partial, indent=2)+'\n')
    result = dict(node=NODE, input_names=NAMES, derivative=facts, results=rows,
                  first_pair_screens=[v[1] for v in caches.values()], complete=True,
                  family_cases=len(rows), excluded=sum(r['status'] == 'excluded' for r in rows),
                  sample_points=bs, sample_points_sha256=hashlib.sha256(np.array(bs, dtype='<u4').tobytes()).hexdigest(),
                  seconds=round(time.monotonic()-start, 4), best_executable_cycles=864, new_executable=False,
                  scope='Exactly division/remainder/ceil of one of nine live words by an arbitrary nonzero root-prepared uint32 divisor, XOR with a live word, and arbitrary affine tail. Only fixed legal root tested. No division of computed prefixes, memory/FLOW or other three-op topologies.')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print('COMPLETE', result['excluded'], '/', result['family_cases'], 'excluded', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args(); run(args.output)


if __name__ == '__main__':
    main()
