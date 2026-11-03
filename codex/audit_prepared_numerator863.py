"""Prepared numerator, computed denominator, and q-plus-product bypasses.

High quotients give prefix-independent certificates. Small initial
factors also permit exact inverse equations for every denominator word.
Floor/ceil inverses use integer quadratic intervals, without SMT division
or a denominator bound. REM uses complete quotient-constant intervals.
"""
from __future__ import annotations

import argparse
from itertools import combinations_with_replacement
import json
from math import isqrt
from pathlib import Path

import numpy as np

from codex.audit_denominator_products863 import linear_class
from codex.branch_computed_denominator863 import data
from codex.screen_middle_division_residual863 import pool
from codex.screen_division_xor_affine863 import quotient
from codex.search_early_boundary863 import C0, C1, MASK, NAMES, NODE, early_inputs, wanted


def quadratic_interval(a, b, c):
    """Exact integer interval where a*x*x-b*x+c<=0, for a>0."""
    assert a > 0
    disc = b*b-4*a*c
    if disc < 0: return None
    root = isqrt(disc); d = 2*a
    lo, hi = -((root-b)//d), (b+root)//d
    return (lo, hi) if lo <= hi else None


def subtract_interval(outer, inner):
    if outer is None: return []
    lo, hi = outer
    if inner is None or inner[1] < lo or inner[0] > hi: return [(lo, hi)]
    out = []
    if lo < inner[0]: out.append((lo, min(hi, inner[0]-1)))
    if inner[1] < hi: out.append((max(lo, inner[1]+1), hi))
    return out


def denominator_intervals(n, c, y, op, width=32):
    """All P in1..MASK with n*P+op(c,P)==y (mod2**width).

    For floor,0<=nP^2-SP+C<P. For ceil,-P<nP^2-SP+C<=0.
    S=Y+t*2**width with0<=t<=n covers every possible wrap count.
    Returned intervals contain only solutions, not just candidates.
    """
    assert n > 0 and op in ('//', 'cdiv')
    modulus = 1 << width; mask = modulus-1; out = []
    for winding in range(n+1):
        s = y+winding*modulus
        if op == '//':
            outer = quadratic_interval(n, s+1, c+1)  # F-P<0
            inner = quadratic_interval(n, s, c+1)  # F<0
        else:
            outer = quadratic_interval(n, s, c)  # F<=0
            inner = quadratic_interval(n, s-1, c)  # F+P<=0
        for lo, hi in subtract_interval(outer, inner):
            lo, hi = max(lo, 1), min(hi, mask)
            if lo <= hi:
                assert all((n*p+quotient(c, p, op)) & mask == y for p in (lo, hi))
                out.append((lo, hi))
    return sorted(out)


def remainder_progressions(n, c, y, width=32):
    """Exact denominator progressions for n*P+(C%P)==Y modulo2**width."""
    mask = (1 << width)-1
    lo = 1; out = []
    while lo <= mask:
        k = c//lo; hi = c//k if k else mask
        # C%P=C-k*P throughout this quotient-constant interval.
        cls = linear_class(n-k, y-c, width)
        if cls is not None:
            base, step = cls; first = lo+(base-lo) % step
            if first <= hi: out.append((first, hi, step))
        lo = hi+1
    return out


def numerator_candidates(y0, op, width=32):
    """All prepared C compatible with op(C,P0)=y0 for some nonzero P0."""
    assert y0 > 1 and op in ('//', 'cdiv')
    mask = (1 << width)-1
    bound = mask//y0 if op == '//' else (mask-1)//(y0-1)
    result = set()
    for p in range(1, bound+1):
        lo, hi = (y0*p, (y0+1)*p-1) if op == '//' else ((y0-1)*p+1, y0*p)
        result.update(range(max(lo, 0), min(hi, mask)+1))
    return sorted(result)


def vector_inverse(source, xx):
    """Batch inverse for the seven permutation-valued initial sources."""
    xx = np.asarray(xx, dtype=np.uint64)
    if source in (0, 1):
        target = ((xx-np.uint64(0xFD7046C5))*np.uint64(pow(9, -1, 1 << 32))) & MASK if source == 0 else xx
        b = np.zeros(len(xx), dtype=np.uint64)
        for bit in range(32):
            z = (((33*b+0xE9F8CC1D) & MASK) ^ ((16896*b+0xACCF6200) & MASK))
            b ^= (((z ^ target) >> bit) & 1) << bit
    elif source == 2:
        b = ((xx-np.uint64(0xE9F8CC1D))*np.uint64(pow(33, -1, 1 << 32))) & MASK
    elif source == 4:
        b = xx
    elif source in (5, 6, 8):
        a = ((4097*xx+C0) & MASK) if source == 5 else xx ^ C1 if source == 8 else xx
        b = a ^ (a >> 19) ^ C1
    else:
        raise ValueError(source)
    assert np.array_equal(early_inputs(b)[source], xx)
    return b


def distinct_tag_slice(coefficients, targets, tags=8):
    """Select targets with distinct low tags; coefficient low tags must agree."""
    mask = (1 << tags)-1; seen = set(); selected = []
    assert len({int(x) & mask for x in coefficients}) == 1
    for index, y in enumerate(targets):
        tag = int(y) & mask
        if tag not in seen: selected.append(index); seen.add(tag)
    return selected


def low_scalar_filter(coefficients, targets, width=32, split=16):
    """Exhaust all low scalar bits, using the necessary one-high condition.

    Coefficients are multiples of2**split and width=2*split. The omitted
    scalar high bits vanish modulo2**width. Distinct target low split
    bits imply distinct q for every scalar, so two high q contradict a
    single prepared numerator for DIV/REM/CDIV.
    """
    assert width == 2*split
    mask = (1 << width)-1; half = 1 << (width-1)
    assert all(int(x) & ((1 << split)-1) == 0 for x in coefficients)
    assert len({int(y) & ((1 << split)-1) for y in targets}) == len(targets)
    candidates = np.arange(1 << split, dtype=np.uint64)
    high = np.zeros(len(candidates), dtype=np.uint8); history = []
    for index, (x, y) in enumerate(zip(coefficients, targets)):
        with np.errstate(over='ignore'):
            q = (np.uint64(y)-candidates*np.uint64(x)) & np.uint64(mask)
        high += q > half
        keep = high <= 1; candidates = candidates[keep]; high = high[keep]
        history.append(dict(index=index, low_scalar_survivors=len(candidates)))
        if not len(candidates): break
    return [int(g) for g in candidates], history


def audit():
    bs = pool(); vv = [early_inputs(b) for b in bs]; yy = [wanted(v) for v in vv]
    fixed = []
    for left, right in combinations_with_replacement(range(9), 2):
        high = {}
        for b, v, y in zip(bs, vv, yy):
            q = (y-v[left]*v[right]) & MASK
            if q > 1 << 31: high.setdefault(q, b)
            if len(high) == 2: break
        fixed.append(dict(factors=[left, right], status='excluded' if len(high) >= 2 else 'open',
                          high_outputs=[dict(q=q, primary_b=b) for q, b in high.items()]))
    scalars = []; products = []
    for source in range(9):
        raw_b, values, targets, nn, _ = data(source)
        if source in (3, 7):
            zeros = raw_b if source == 3 else [C1 ^ a for a in range(1024)] + [C1 ^ 65536 ^ a for a in range(1024)]
            high = {}
            for b in zeros:
                v = early_inputs(b); assert v[source] == 0
                y = wanted(v)
                if y > 1 << 31: high.setdefault(y, b)
                if len(high) == 2: break
            status = 'excluded' if len(high) == 2 else 'open'
            certificate = dict(source=source, source_name=NAMES[source], status=status,
                               certificate='two_distinct_high_targets_at_zero_factor',
                               high_outputs=[dict(q=q, primary_b=b) for q, b in high.items()])
            scalars.append(certificate); products.append(dict(**certificate, all_ops=True))
            continue
        # Complete low16 scalar enumeration represents all2^32 G; no
        # restriction to odd G. Only512 legal points are needed here.
        xx = np.arange(512, dtype=np.uint64) << 16
        primary = vector_inverse(source, xx); target = wanted(early_inputs(primary))
        selected = distinct_tag_slice(xx, target)
        indices = selected[:128]
        lows, history = low_scalar_filter(xx[indices], target[indices])
        scalars.append(dict(source=source, source_name=NAMES[source], status='excluded' if not lows else 'open',
                            represented_word_scalars=1 << 32, enumerated_low16_scalars=1 << 16,
                            primary_b=[int(primary[i]) for i in indices],
                            initial_factors=[int(xx[i]) for i in indices], targets=[int(target[i]) for i in indices],
                            history=history, low16_survivors=lows,
                            omitted_high16_bits_annihilated=True))
        y0 = targets[0]; cases = []
        for op in ('//', 'cdiv'):
            candidates = numerator_candidates(y0, op); live = candidates[:]; dead = []; history = []
            for index in range(1, len(nn)):
                survivors = []
                for c in live:
                    if denominator_intervals(nn[index], c, targets[index], op): survivors.append(c)
                    else: dead.append(dict(numerator=c, impossible_factor=nn[index], witness_index=index))
                live = survivors; history.append(dict(index=index, surviving_numerators=len(live)))
                if not live: break
            cases.append(dict(op=op, status='excluded' if not live else 'open',
                              initial_numerator_candidates=candidates, history=history,
                              empty_denominator_certificates=dead, surviving_numerators=live))
        if y0 > 1 << 31:
            # A high quotient/remainder equals its prepared numerator.
            c = y0; impossible = None; counts = []
            for index in range(1, min(len(nn), 33)):
                roots = remainder_progressions(nn[index], c, targets[index])
                counts.append(dict(index=index, progression_count=len(roots),
                                   denominator_count=sum((hi-lo)//step+1 for lo, hi, step in roots)))
                if not roots: impossible = index; break
            cases.append(dict(op='%', status='excluded' if impossible is not None else 'open',
                              pinned_numerator=c, high_zero_factor_anchor=True,
                              impossible_index=impossible, denominator_counts=counts))
        else:
            cases.append(dict(op='%', status='open',
                              reason='The zero-factor target is not high; no single prepared numerator is pinned.'))
        products.append(dict(source=source, source_name=NAMES[source],
                             primary_b=raw_b, initial_factors=nn, targets=targets, cases=cases))
        print(NAMES[source], 'scalar', scalars[-1]['status'],
              'prefix products', [(r['op'], r['status']) for r in cases], flush=True)
    return dict(node=NODE, input_names=NAMES, fixed_initial_products=fixed,
                scalar_initial_products=scalars, prefix_initial_products=products,
                fixed_product_scope='Y=I*J+q, I/J initial; q=DIV/REM/CDIV(C_prepared,P) with any nonzero denominator P.',
                scalar_product_scope='Y=G_prepared*I+q, arbitrary32-bit G/C; same arbitrary denominator. Every high q>2^31 must equal C. Two distinct high residuals exclude the case.',
                prefix_product_scope='Y=I*P+q, I initial, q=DIV/REM/CDIV(C_prepared,P). Excluded cases permit an arbitrary nonzero denominator per input, more general than a single prefix operation. Floor/ceil numerator candidates are complete, not a bounded search.',
                best_executable_cycles=864, new_executable=False, no_global_optimality_claim=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Fixed products:', sum(r['status'] == 'excluded' for r in result['fixed_initial_products']), flush=True)
    print('Scalar products:', sum(r['status'] == 'excluded' for r in result['scalar_initial_products']), flush=True)


if __name__ == '__main__':
    main()
