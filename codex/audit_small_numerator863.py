"""Force computed odd denominators from legal small-numerator inputs.

Family: P=one MAD/bitop, q=DIV/REM/CDIV(V,P), out=P*q+H.
The initial numerator V is allowed to range over all32-bit words where
bijective. At V=0, H is pinned. Small positive V then either contradict
floor bounds or force P, allowing independent polynomial/bitop checks.
"""
from __future__ import annotations

import argparse
from itertools import combinations_with_replacement
import json
from pathlib import Path

from codex.audit_division_first_t863 import source_inverse
from codex.audit_computed_shift_amount863 import zero_points
from codex.audit_runtime_shift_core863 import base_fields, monomials
from codex.screen_xor_mad_polynomial import consistent
from codex.search_early_boundary863 import MASK, NAMES, NODE, early_inputs, wanted, legal_point


def bit_fits(values, target):
    """Exact one-bitop interpolation with arbitrary prepared masks."""
    assert values and len(values) == len(target)
    fits = []
    if len(set(target)) == 1:
        fits.append(dict(op='constant', value=target[0]))
    for i, j in combinations_with_replacement(range(9), 2):
        for op in ('^', '&', '|'):
            out = [v[i]^v[j] if op == '^' else v[i]&v[j] if op == '&' else v[i]|v[j] for v in values]
            if out == target:
                fits.append(dict(op=op, operands=[i, j]))
    for i in range(9):
        xx = [v[i] for v in values]
        keys = {x^y for x, y in zip(xx, target)}
        if len(keys) == 1:
            fits.append(dict(op='^K', source=i, key=keys.pop()))
        on = off = 0; valid = True
        for x, y in zip(xx, target):
            valid &= y & ~x == 0
            on |= y; off |= x & ~y & MASK
        if valid and on & off == 0:
            fits.append(dict(op='&K', source=i, key=on))
        on = off = 0; valid = True
        for x, y in zip(xx, target):
            valid &= x & ~y == 0
            on |= y & ~x & MASK; off |= ~y & MASK
        if valid and on & off == 0:
            fits.append(dict(op='|K', source=i, key=on))
    return fits


def forced_denominators(numerators, targets, offset, op):
    rows = []
    for n, y in zip(numerators, targets):
        if n == 0 or not n & 1:
            continue
        residual = (y-offset) & MASK
        threshold = 2*n-1 if op == 'cdiv' else n*n
        if residual > threshold:
            p = residual if op == 'cdiv' else (residual*pow(n, -1, 1 << 32)) & MASK
            rows.append(dict(numerator=n, target=y, residual=residual, forced_prefix=p,
                             prefix_must_exceed_numerator=True,
                             parity_compatible=bool(p & 1), exceeds_numerator=p > n))
    return rows


def audit():
    rows = []; zeros = zero_points()
    for source, name in enumerate(NAMES):
        if source in (3, 7):
            b = zeros[name]; c = b ^ (1 << 23 if source == 3 else 65536)
            v, w = early_inputs(b), early_inputs(c)
            assert v[source] == w[source] == 0 and wanted(v) != wanted(w)
            rows.append(dict(source=source, source_name=name, all_ops='excluded',
                             zero_pair=[legal_point(b), legal_point(c)],
                             proof='Both numerators0 force the same output offset H, but targets differ.'))
            continue
        bs = [source_inverse(source, n) for n in range(129)]
        vv = [early_inputs(b) for b in bs]; yy = [wanted(v) for v in vv]; h = yy[0]
        assert (yy[1]-h) & MASK > 1
        results = [dict(op='//', status='excluded', numerator1_point=legal_point(bs[1]),
                        residual=(yy[1]-h) & MASK,
                        proof='P*floor(1/P) is either0 or1; the target minus pinned H exceeds1.')]
        for op in ('cdiv', '%'):
            forced = forced_denominators(list(range(129)), yy, h, op)
            bad = next((r for r in forced if not r['parity_compatible'] or not r['exceeds_numerator']), None)
            if bad:
                report = dict(op=op, status='excluded', certificate='forced_prefix_invalid_or_even', witness=bad,
                              legal_point=legal_point(bs[bad['numerator']]))
            else:
                values = [vv[r['numerator']] for r in forced]
                target = [r['forced_prefix'] for r in forced]
                matrix = [monomials(base_fields(bs[r['numerator']], vv[r['numerator']])) for r in forced]
                polynomial = consistent(matrix, target, 32)
                bit = bit_fits(values, target)
                assert not polynomial and not bit, (source, op, polynomial, bit)
                control = [(v[0]*v[2]+v[5]) & MASK for v in values]
                assert consistent(matrix, control, 32)
                report = dict(op=op, status='excluded', certificate='all_one_MAD_or_bitop_prefixes_incompatible',
                              forced=forced, primary_b=[bs[r['numerator']] for r in forced],
                              degree2_base_fields=5, degree2_coefficients=21,
                              polynomial_consistent=polynomial, bitop_fits=bit,
                              positive_polynomial_control=True)
            results.append(report)
        rows.append(dict(source=source, source_name=name, pinned_offset=h,
                         zero_point=legal_point(bs[0]), results=results))
    return dict(node=NODE, input_names=NAMES, results=rows,
                even_prefix_exclusion='Target parity flips on b16 while all initial and one-T prefix parities stay fixed. The coefficient P in P*q+H must therefore be odd everywhere.',
                ceil_forcing='If n>0 and P<=n then n<=P*ceil(n/P)<=2*n-1, without wrap for small n. A residual above that bound forces P>n and quotient1, so P equals the residual.',
                rem_forcing='If P<=n then P*(n%P)<n^2, without wrap for small n. A larger residual forces P>n and remainder n. Odd n is invertible modulo2^32 and pins P exactly.',
                all_stated_cases_excluded=True, best_executable_cycles=864, new_executable=False,
                scope='One MAD/XOR/AND/OR computed denominator P, initial-word numerator, DIV/REM/CDIV, and final P*q+prepared H. Arbitrary root-only prefix constants. No other final tail or additional prefix operations.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('All nine initial-numerator, computed-odd-denominator product/offset families excluded', flush=True)


if __name__ == '__main__':
    main()
