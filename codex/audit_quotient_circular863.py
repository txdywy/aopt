"""Circular low-word certificates for a prepared middle divisor.

The eight legal states have identical initial low16. Every one-T prefix
and T-only tail coefficient/addend therefore also has identical low16.
For REM with an affine-in-q final MAD, an odd unit maps target residues
to a translate of small integer quotients k=floor(P/D).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from codex.audit_comparison_core863 import fibre
from codex.audit_computed_shift_amount863 import zero_points
from codex.close_computed_denominator863 import prepared_bitop_certificates
from codex.search_early_boundary863 import MASK, NAMES, NODE, early_inputs, wanted, legal_point


def circular_span(values, width=16):
    modulus = 1 << width
    vv = sorted(set(int(x) & (modulus-1) for x in values))
    assert vv
    largest_gap = max([b-a for a, b in zip(vv, vv[1:])] + [vv[0]+modulus-vv[-1]])
    return modulus-largest_gap


def unit_span_scan(targets, width=16, word_bits=32):
    modulus = 1 << width
    units = np.arange(1, modulus, 2, dtype=np.uint64)
    yy = np.array(targets, dtype=np.uint64) & (modulus-1)
    values = np.sort((units[:, None]*yy[None, :]) & (modulus-1), axis=1)
    gap = np.maximum(np.diff(values, axis=1).max(axis=1), values[:, 0]+modulus-values[:, -1]) if len(yy) > 1 else np.full(len(units), modulus)
    spans = modulus-gap; minimum = int(spans.min())
    return dict(width=width, enumerated_odd_units=len(units), complete_unit_domain=True,
                minimum_span=minimum, minimizing_inverse_units=[int(x) for x in units[spans == minimum]],
                span_array_sha256=hashlib.sha256(spans.astype('<u4').tobytes()).hexdigest(),
                necessary_max_divisor=((1 << word_bits)-1)//minimum if minimum else (1 << word_bits)-1,
                word_bits=word_bits)


def audit():
    ff = fibre(); yy = ff['target']; vv = ff['initial_values']
    assert len({y & 7 for y in yy}) == 8
    assert all(len({v[j] & 65535 for v in vv}) == 1 for j in range(9))
    rem = unit_span_scan(yy); assert rem['minimum_span'] == 14190
    assert rem['necessary_max_divisor'] == 302675
    zeros = zero_points(); rows = []; legal = set(ff['primary_b'])
    for source, name in enumerate(NAMES):
        b = zeros[name]; legal.add(b); values = early_inputs(b); y0 = wanted(values)
        assert values[source] == 0 and y0 > rem['necessary_max_divisor']
        cases = []
        for op in ('//', 'cdiv'):
            maximum = MASK//y0 if op == '//' else (MASK-1)//(y0-1)
            checks = [dict(divisor=d, span=circular_span([d*y for y in yy]),
                           maximum_allowed_span=d-1) for d in range(1, maximum+1)]
            survivors = [r['divisor'] for r in checks if r['span'] <= r['maximum_allowed_span']]
            assert not survivors
            cases.append(dict(op=op, exact_necessary_max_divisor=maximum,
                              divisor_checks=checks, remaining_divisors=survivors, status='excluded'))
        cases.append(dict(op='%', status='excluded',
                          required_divisor_minimum=y0+1, global_divisor_maximum=rem['necessary_max_divisor']))
        rows.append(dict(source=source, name=name, zero_factor_primary_b=b, target=y0, cases=cases))
    bit = prepared_bitop_certificates()
    # Its XOR witnesses were originally used for a prepared numerator.
    # With a computed numerator and prepared D, one high required q
    # already pins floor/ceil to D=1. For REM it forces D>2^31.
    xor = []
    for row in bit['xor_initial']:
        source = row['source']; witness = row['distinct_high_quotients'][0]
        v = early_inputs(witness['primary_b']); assert wanted(v) ^ v[source] == witness['q'] > 1 << 31
        legal.add(witness['primary_b'])
        xor.append(dict(source=source, witness=witness, all_ops_excluded=True))
    return dict(node=NODE, input_names=NAMES, eight_state_fibre=ff,
                rem_affine_quotient_bound=rem,
                rem_even_divisor_proof='With even D, P%D has P parity. All T-only input/prefix parities agree on b16 partners, while target parity flips.',
                rem_unit_span_proof='For an affine-in-q last MAD Y=A+G*q, A/G have common low16 on the fibre. Target parity variation forces G odd there. With odd D, q=P-D*k gives Y=L-G*D*k modulo2^16. An odd inverse unit sends the targets to a translate of k=floor(P/D), where0<=k<=floor(MASK/D). The shortest circular span cannot exceed that range. Complete enumeration of all odd inverse units gives the recorded global bound, without limiting D to16 bits.',
                floor_ceil_circle_proof='For Y=R+q, R/P have common low16. Floor has P=D*q+r and ceil has P=D*q-s, with0<=r,s<D. Thus D*Y residues lie in a translate of an interval of length D-1. This does not assume D odd.',
                zero_factor_families='Y=I*X+q, I any initial word and X any initial word, computed one-T P or root-prepared constant. At I=0 the quotient equals the target, giving a complete small upper bound on floor/ceil D, or a lower bound on REM D.',
                zero_factor_results=rows, bitop_initial_xor=xor,
                bitop_initial_and_or=dict(and_witness=bit['and_initial_mask_witness'],
                                         or_witness=bit['or_initial_zero_witness']),
                high_xor_proof='Floor/ceil with D>=2 cannot exceed2^31; D=1 reduces to two T ops. A high REM requires D>2^31, so floor(P/D) is0 or1. P and the bypass have common low16 on the eight-state fibre, giving at most two output low16 values, fewer than the eight target values.',
                legal_points=[legal_point(b) for b in sorted(legal)],
                all_zero_factor_and_bitop_initial_cases_excluded=True,
                best_executable_cycles=864, new_executable=False, no_global_optimality_claim=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('REM bound:', result['rem_affine_quotient_bound']['necessary_max_divisor'], flush=True)
    print('Zero-factor and initial bitwise tails: excluded', flush=True)


if __name__ == '__main__':
    main()
