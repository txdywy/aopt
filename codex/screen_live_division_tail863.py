"""Exhaust constant divisors for a two-instruction live tail replacement.

On eight real inputs all existing live words agree modulo 2**16, while the
target covers all residues modulo 8. A final MAD using the division result
must therefore be affine in it; its square cases cannot cover all residues
modulo 4. Floor/ceil quotient vectors are constant on finitely many divisor
intervals. Every nonzero uint32 divisor is covered, without a solver timeout.
Remainder followed by an affine map reduces to the same quotient condition.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from codex.audit_full_live_tail863 import congruence_certificate
from codex.search_full_live_shift863 import MASK, LIVE_NAMES, zero_witnesses


@lru_cache(maxsize=16)
def inverse_table(bits):
    modulus = 1 << bits
    table = np.zeros(modulus, dtype=np.int64)
    for value in range(1, modulus, 2):
        table[value] = pow(value, -1, modulus)
    return table


def divisor_boundaries(numerators, limit):
    pieces = [np.array([1, limit+1], dtype=np.int64)]
    for numerator in numerators:
        n = int(numerator)
        root = math.isqrt(n)
        pieces.append(np.arange(1, min(root+1, limit)+1, dtype=np.int64))
        if root:
            edges = n//np.arange(1, root+1, dtype=np.int64)+1
            pieces.append(edges[edges <= limit+1])
    edges = np.unique(np.concatenate(pieces))
    assert edges[0] == 1 and edges[-1] == limit+1
    return edges


def affine_intervals(numerators, outputs, *, ceil=False, bits=16, divisor_limit=MASK):
    if not 1 <= bits <= 16 or not 1 <= divisor_limit <= MASK:
        raise ValueError('bits must be 1..16 and divisor limit must be a positive uint32')
    if len(numerators) != len(outputs) or len(outputs) < 2:
        raise ValueError('matching numerator/output arrays of length at least two required')
    if any(not isinstance(n, int) or not 0 <= n <= MASK for n in numerators):
        raise ValueError('numerators must be uint32 integers')
    mask = (1 << bits)-1
    nums = np.array(numerators, dtype=np.int64)
    offsets = (nums > 0).astype(np.int64) if ceil else np.zeros(len(nums), dtype=np.int64)
    if ceil:
        nums = np.maximum(0, nums-1)
    ys = np.array([y & mask for y in outputs], dtype=np.int64)
    differences = (ys-ys[0])&mask
    odd = next((i for i, d in enumerate(differences) if d & 1), None)
    if odd is None and np.any(differences):
        raise ValueError('screen requires an odd output difference or a constant target')
    edges = divisor_boundaries(nums, divisor_limit)
    starts, ends = edges[:-1], edges[1:]-1
    count = len(starts)
    # Quotients are monotone. Equal endpoint values prove constancy on each
    # whole interval; the adjacent intervals partition [1, divisor_limit].
    for n in nums:
        assert np.array_equal(n//starts, n//ends)
    covered = int(np.sum(ends-starts+1))
    assert covered == divisor_limit
    digest = hashlib.sha256(edges.astype('<i8').tobytes()).hexdigest()
    if odd is not None:
        q0 = nums[0]//starts+offsets[0]
        delta = (nums[odd]//starts+offsets[odd]-q0)&mask
        keep = (delta & 1) != 0
        starts, ends, q0, delta = starts[keep], ends[keep], q0[keep], delta[keep]
        scale = (differences[odd]*inverse_table(bits)[delta])&mask
        for i in range(1, len(nums)):
            delta = (nums[i]//starts+offsets[i]-q0)&mask
            keep = ((scale*delta-differences[i])&mask) == 0
            starts, ends, q0, scale = starts[keep], ends[keep], q0[keep], scale[keep]
    survivors = list(map(list, zip(starts.tolist(), ends.tolist())))
    return dict(kind='ceil' if ceil else 'floor', bits=bits, divisor_limit=divisor_limit,
        numerators=numerators, outputs_low=ys.tolist(), intervals_tested=count,
        covered_divisors=covered, endpoint_constancy_checked=True,
        boundary_sha256=digest, survivor_intervals=survivors,
        excluded=not survivors)


def positive_control(numerators, ceil=False):
    # Use a real affine-of-division function. The constants are only for the
    # control; the negative screen allows every affine coefficient.
    for divisor in (3, 5, 7, 1, 16, 255, 256, 257, 65535, 65536, 1 << 25):
        qs = [(n+divisor-1)//divisor if ceil else n//divisor for n in numerators]
        ys = [(17*q+23)&MASK for q in qs]
        if len(set(ys)) == 1 or any((y-ys[0])&1 for y in ys):
            result = affine_intervals(numerators, ys, ceil=ceil)
            kept = any(a <= divisor <= b for a, b in result['survivor_intervals'])
            assert kept
            return dict(divisor=divisor, scale=17, offset=23,
                numerator_values=numerators, target_values=ys,
                retained=True, tested_intervals=result['intervals_tested'])
    raise AssertionError('no suitable control divisor')


def prove():
    started = time.monotonic()
    cert = congruence_certificate()
    words = cert['live_words']
    outputs = cert['target_outputs']
    assert len({y & 3 for y in outputs}) == 4
    rows = []
    for source in range(8):
        nums = [row[source] for row in words]
        assert len({n & 0xffff for n in nums}) == 1
        for ceil in (False, True):
            row = affine_intervals(nums, outputs, ceil=ceil)
            row.update(source=source, source_name=LIVE_NAMES[source],
                       positive_control=positive_control(nums, ceil))
            rows.append(row)
    return dict(scope='tail', width=32, low_bits=16,
        original_input_certificate=cert, zero_divisor_witnesses=zero_witnesses('tail'),
        rows=rows, intervals_tested=sum(row['intervals_tested'] for row in rows),
        excluded=all(row['excluded'] for row in rows),
        closed_sequences=[['//', 'mad'], ['cdiv', 'mad'], ['%', 'mad']],
        constant_numerator_reason='A constant first result and the live words all have fixed low16 '
                                  'values on the certificate; a final MAD cannot vary its low output.',
        remainder_reduction='If all S_i equal s modulo 65536 and Q_i=floor(S_i/D), '
            'then R_i=S_i%D equals s-D*Q_i modulo 65536. Any A*R_i+B is affine in Q_i.',
        mad_roles='At fixed low16 live words: affine in q, q*q+C, or q*q+q. '
            'Square+C has at most two residues mod4; square+q is even. '
            'The actual target covers all four residues mod4.',
        seconds=round(time.monotonic()-started, 3), new_executable=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    report = prove()
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('excluded', 'intervals_tested', 'seconds', 'new_executable')}))


if __name__ == '__main__':
    main()
