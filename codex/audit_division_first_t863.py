"""Exact necessary parity intervals for division followed by two T ops.

T is MAD/XOR/AND/OR. On an initial-word parity class the final bit is a
Boolean function of the sole new quotient bit. Since the target varies
within every class, that function must be q XOR a class constant.
The complete divisor domain is retained as exact quotient intervals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import z3

from codex.audit_comparison_core863 import fibre
from codex.audit_computed_shift_amount863 import zero_points
from codex.screen_live_division_tail863 import divisor_boundaries
from codex.screen_division_xor_affine863 import quotient
from codex.search_early_boundary863 import C1, C0, MASK, NAMES, NODE, early_inputs, wanted, legal_point


CLASSES = (0, 1, 1 << 19, (1 << 19) | 1)


def merge(intervals):
    merged = []
    for lo, hi in intervals:
        if merged and merged[-1][1]+1 == lo:
            merged[-1][1] = hi
        else:
            merged.append([lo, hi])
    return merged


def first_intervals(nums, targets):
    edges = divisor_boundaries(nums, MASK)
    starts, ends = edges[:-1], edges[1:]-1
    total = len(starts)
    for n in nums:
        assert np.array_equal(n//starts, n//ends)
    for i in range(len(nums)):
        for j in range(i):
            delta = (targets[i] ^ targets[j]) & 7
            bit = delta & -delta; assert bit
            keep = (((nums[i]//starts) ^ (nums[j]//starts)) & (2*bit-1)) == bit
            starts, ends = starts[keep], ends[keep]
    return merge(zip(starts.tolist(), ends.tolist())), dict(
        quotient_intervals=total, covered_divisors=MASK, endpoint_constancy_checked=True,
        boundary_sha256=hashlib.sha256(edges.astype('<i8').tobytes()).hexdigest())


def refine(intervals, n, anchor, desired_parity):
    out = []
    for lo, hi in intervals:
        current = lo
        while current <= hi:
            qa, qb = n//current, anchor//current
            end = min(n//qa if qa else MASK, anchor//qb if qb else MASK, hi)
            assert end >= current
            if (qa ^ qb) & 1 == desired_parity:
                out.append([current, end])
            current = end+1
    return merge(out)


def source_inverse(source, value):
    """A real primary b for requested source value, or None if unreachable."""
    if not 0 <= value <= MASK:
        return None
    if source in (0, 1):
        target = ((value-0xFD7046C5)*pow(9, -1, 1 << 32)) & MASK if source == 0 else value
        b = 0
        for bit in range(32):
            if ((early_inputs(b)[1] ^ target) >> bit) & 1:
                b ^= 1 << bit
    elif source == 2:
        b = ((value-0xE9F8CC1D)*pow(33, -1, 1 << 32)) & MASK
    elif source == 3:
        if value & 511:
            return None
        b = ((value//512-0xACCF6200//512)*pow(33, -1, 1 << 23)) & ((1 << 23)-1)
    elif source == 4:
        b = value
    elif source in (5, 6, 8):
        a = ((value*4097+C0) & MASK) if source == 5 else (value ^ C1) if source == 8 else value
        b = a ^ (a >> 19) ^ C1
    elif source == 7:
        if value >= 8192:
            return None
        a = value << 19; b = a ^ (a >> 19) ^ C1
    else:
        raise ValueError(source)
    assert early_inputs(b)[source] == value
    return b


def odd_intervals(intervals):
    # Retain intervals rather than expanding; count only the odd members.
    return sum((hi+1)//2-lo//2 for lo, hi in intervals)


def screen(source, ceil=False, rounds=12):
    started = time.monotonic(); ff = fibre(); vv, yy = ff['initial_values'], ff['target']
    nums = [v[source]-int(ceil) for v in vv]  # All fibre numerators are positive.
    intervals, first = first_intervals(nums, yy)
    anchors = {p:(early_inputs(p)[source], wanted(early_inputs(p))) for p in CLASSES}
    history = []; used = []

    def add(b):
        nonlocal intervals
        v = early_inputs(b); y = wanted(v); cls = b & ((1 << 19) | 1)
        n0, target0 = anchors[cls]; n = v[source]
        offset = int(ceil and n > 0); offset0 = int(ceil and n0 > 0)
        before = sum(hi-lo+1 for lo, hi in intervals)
        intervals = refine(intervals, max(n-1, 0) if ceil else n,
                           max(n0-1, 0) if ceil else n0, (y ^ target0 ^ offset ^ offset0) & 1)
        after = sum(hi-lo+1 for lo, hi in intervals)
        if after != before:
            history.append(dict(primary_b=b, before=before, after=after, intervals=len(intervals)))
        used.append(b)

    rng = random.Random(8642856)
    pool = list(dict.fromkeys([*CLASSES, *(p ^ 65536 for p in CLASSES), *range(128),
                              *(1 << j for j in range(32)),
                              *(rng.getrandbits(32) for _ in range(256))]))
    for b in pool:
        if not intervals:
            break
        add(b)
    # Probe exact quotient boundaries. Each found point refines every
    # surviving interval, not just the divisor used to construct it.
    for iteration in range(rounds):
        before = [i[:] for i in intervals]
        divisors = sorted({d for lo, hi in intervals for d in (lo, (lo+hi)//2, hi)})
        if len(divisors) > 64:
            divisors = [divisors[i*(len(divisors)-1)//63] for i in range(64)]
        for divisor in divisors:
            if not any(lo <= divisor <= hi for lo, hi in intervals):
                continue
            found = None
            for multiple in (0, 1, 2, 3, 7, 31, 127, 1023, MASK//divisor//2, MASK//divisor):
                for residual in (-1, 0, 1, divisor//2, divisor-1):
                    n = multiple*divisor+residual
                    if source == 3:
                        n &= ~511
                    b = source_inverse(source, n)
                    if b is None:
                        continue
                    v = early_inputs(b); y = wanted(v); cls = b & ((1 << 19) | 1)
                    n0, t0 = anchors[cls]
                    op = 'cdiv' if ceil else '//'
                    if (quotient(v[source], divisor, op) ^ quotient(n0, divisor, op)) & 1 != (y ^ t0) & 1:
                        found = b; break
                if found is not None:
                    break
            if found is not None:
                add(found)
        if intervals == before:
            break
    return dict(source=source, source_name=NAMES[source], op='cdiv' if ceil else '//',
                first_interval_partition=first, first_fibre=ff['primary_b'],
                refinement_history=history, tested_points=used, survivor_intervals=intervals,
                remaining_divisors=sum(hi-lo+1 for lo, hi in intervals),
                remaining_odd_divisors_for_rem=odd_intervals(intervals) if not ceil else None,
                seconds=round(time.monotonic()-started, 6),
                necessary_condition_only=True, all_domain_partition_exact=True)


def full32_class_facts():
    b = z3.BitVec('first_division_class_b', 32)
    vv = early_inputs(b); cc = early_inputs(b & ((1 << 19) | 1))
    rows = []
    for source, name in enumerate(NAMES):
        s = z3.SolverFor('QF_BV'); s.set(timeout=10000)
        s.add(z3.Extract(0, 0, vv[source]) != z3.Extract(0, 0, cc[source]))
        assert s.check() == z3.unsat, name
        rows.append(dict(source=source, source_name=name, status='unsat'))
    s = z3.SolverFor('QF_BV'); s.set(timeout=10000)
    changed = wanted(vv) ^ wanted(early_inputs(b ^ 65536))
    s.add(z3.Extract(0, 0, changed) == 0)
    assert s.check() == z3.unsat
    return dict(initial_lsb_depends_only_on_b0_b19=rows, target_b16_flip_full32='unsat')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rounds', type=int, default=12)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); rows = []; facts = full32_class_facts()
    for source in range(9):
        for ceil in (False, True):
            row = screen(source, ceil, args.rounds); rows.append(row)
            print(row['source_name'], row['op'], row['remaining_divisors'], row['survivor_intervals'][:12], flush=True)
            args.output.write_text(json.dumps(dict(node=NODE, results=rows, complete=False), indent=2)+'\n')
    prior_path = Path('codex/bidirectional_shift_core863_audit.json')
    prior_bytes = prior_path.read_bytes(); prior = json.loads(prior_bytes)
    assert prior['node'] == NODE and prior['remaining_in_stated_three_op_core'] == 0
    surviving = [(r['source'], r['op'], r['survivor_intervals']) for r in rows if r['remaining_divisors']]
    closed = surviving == [(0, '//', [[65536, 65536]])]
    assert all(r['remaining_odd_divisors_for_rem'] == 0 for r in rows if r['op'] == '//')
    zeros = zero_points()
    result = dict(node=NODE, input_names=NAMES, results=rows, complete=True,
                  full32_class_facts=facts,
                  initial_denominator_zero_witnesses={name:legal_point(zeros[name]) for name in NAMES},
                  constant_numerator='A root-only constant numerator and divisor produce a constant; two T-only operations on the initial words cannot transfer b16 to output bit0.',
                  only_floor_survivor=surviving, final_family_status='excluded' if closed else 'open',
                  floor_survivor_reduction='e//65536 is exactly e>>16; two T suffixes are inside the prior closed three-op shift core.',
                  prior_shift_core=dict(path=str(prior_path), sha256=hashlib.sha256(prior_bytes).hexdigest()),
                  target_parity_classes='All nine initial LSBs depend only on b0,b19; b16 flips target parity without changing them. A T-only suffix must map the quotient bit to target parity by XOR with a class constant.',
                  rem_reduction='For even D, remainder parity equals initial numerator parity and cannot depend on b16. For odd D, remainder parity is numerator parity XOR floor-quotient parity, and numerator parity is constant within each initial class. Thus floor intervals plus odd-D restriction apply.',
                  all_survivors_require_further_analysis=not closed, new_executable=False, best_executable_cycles=864,
                  scope='A first DIV/REM/CDIV on initial words/root constants, followed by at most two MAD/XOR/AND/OR operations. Initial-word denominators hit0; nonzero constant denominators have exact necessary interval screens. The sole floor survivor is reduced to the previously closed shift grammar. No computed division prefix, second shift/comparison/division, or other input encoding.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
