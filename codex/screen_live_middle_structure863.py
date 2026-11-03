"""Structural necessary conditions with extra first-stage live inputs.

Restrict full32 a to [0,2**w), w<=12. Then b=a XOR C1 (low words),
shift=0, mix=b, and prior=a-C0 modulo 2**w. All b occur. On rectangles
that toggle disjoint input bits h,k, every live word is affine in the two
rectangle coordinates with increments divisible by h,k. Any integer
polynomial in those words has mixed difference divisible by h*k.

The final-bitwise screen enumerates a superset of all first-MAD forms,
without solving for the second MAD: its result can be ANY polynomial in
the live words. Survivors satisfy necessary conditions only.
"""
import argparse
from collections import Counter
import itertools
import json
import time

import numpy as np

from codex.screen_quadratic_xor_affine863 import target


def live(x, width):
    if not 1 <= width <= 12:
        raise ValueError('this first-MAD partition requires width 1..12')
    mask = (1 << width)-1
    a = x ^ (0xC761C23C & mask)
    return (x, a, (a-0x7ED55D16) & mask)


def rectangles(width, fn=target):
    mask = (1 << width)-1
    rows = []
    for j in range(1, width):
        for k in range(1, j):
            if j+k < width:
                continue
            h, step = 1 << j, 1 << k
            for seed in (0, 1, 2, 3, 4, 5, 7, 15, 31, 63, 127, 255, 511, 1023, 2047):
                x = (seed & mask) & ~(h | step)
                row = (x, x+step, x+h, x+step+h)
                if row not in rows:
                    rows.append(row)
    return sorted(rows, key=lambda r: (fn(r[3])-fn(r[2])-fn(r[1])+fn(r[0])) & mask,
                  reverse=True)


def prefix_certificate(width=11, fn=target):
    mask = (1 << width)-1
    for row in rectangles(width, fn):
        values = [live(x, width) for x in row]
        horizontal = vertical = 0
        for column in zip(*values):
            horizontal |= (column[0] ^ column[2]) | (column[1] ^ column[3])
            vertical |= (column[0] ^ column[1]) | (column[2] ^ column[3])
        h, k = row[2]-row[0], row[1]-row[0]
        residue = (fn(row[3])-fn(row[2])-fn(row[1])+fn(row[0])) & mask
        if (residue and horizontal & vertical == 0 and horizontal & (h-1) == 0
                and vertical & (k-1) == 0):
            # Every bit of every live input depends on at most one rectangle
            # coordinate, even after arbitrary pointwise bitwise operations.
            return {'excluded': True, 'width': width, 'inputs': list(row),
                    'live_values': values, 'outputs': [fn(x) & mask for x in row],
                    'strides': [h, k], 'horizontal_mask': horizontal,
                    'vertical_mask': vertical, 'mixed_difference': residue,
                    'scope': 'Any bitwise-only prefix followed by any integer polynomial '
                             'in b,a,prior and prefix results; constants allowed'}
    return {'excluded': False, 'width': width}


def families():
    rows = [(name, ()) for name in ('affine_b', 'affine_a', 'mixed_ba', 'mixed_ab')]
    for i, j in itertools.combinations_with_replacement(range(3), 2):
        rows.append(('product_key', (i, j)))
        rows.extend(('product_live', (i, j, k)) for k in range(3))
    return rows


def operand(x, width, name, indices, d=0, e=0):
    b, a, p = words = live(x, width)
    if name == 'affine_b':
        return d*b+e
    if name == 'affine_a':
        return d*a+e
    if name == 'mixed_ba':
        return d*b+a+e
    if name == 'mixed_ab':
        return d*a+b+e
    if name == 'product_key':
        i, j = indices
        return words[i]*words[j]+e
    if name == 'product_live':
        i, j, k = indices
        return words[i]*words[j]+words[k]+0*e
    raise ValueError(name)


def parameter_count(width, name):
    return (1 << (2*width)) if name.startswith(('affine', 'mixed')) else (
        (1 << width) if name == 'product_key' else 1)


def parameters(ids, width, name):
    if name.startswith(('affine', 'mixed')):
        return ids >> width, ids & ((1 << width)-1)
    if name == 'product_key':
        return np.zeros_like(ids), ids
    return np.zeros_like(ids), np.zeros_like(ids)


def condition(values, outputs, width, bitwise):
    mask = (1 << width)-1
    if bitwise == '^':
        q = [(f & mask) ^ (p & mask) for p, f in zip(values, outputs)]
        return ((q[3]-q[2]-q[1]+q[0]) & mask) == 0
    p, f = values[0] & mask, outputs[0] & mask
    if bitwise == '&':
        return (f & (mask ^ p)) == 0
    if bitwise == '|':
        return (p & (mask ^ f)) == 0
    raise ValueError(bitwise)


def search(width=11, bitwise='^', fn=target, chunk_size=131072):
    if not 2 <= width <= 12 or chunk_size <= 0 or bitwise not in ('^', '&', '|'):
        raise ValueError('requires width 2..12, positive chunk, XOR/AND/OR')
    mask = (1 << width)-1
    samples = list(dict.fromkeys([v & mask for v in list(range(32))+
        [63, 127, 255, 511, 512, 513, 1023, 1024, 1025, 2047]]+list(range(mask+1))))
    constraints = rectangles(width, fn) if bitwise == '^' else [(x,) for x in samples]
    started = time.monotonic()
    rows = []
    for name, indices in families():
        total = parameter_count(width, name)
        survivors, rejected = [], Counter()
        for start in range(0, total, chunk_size):
            ids = np.arange(start, min(start+chunk_size, total), dtype=np.int64)
            d, e = parameters(ids, width, name)
            for rank, points in enumerate(constraints):
                ps = [operand(x, width, name, indices, d, e) for x in points]
                keep = condition(ps, [fn(x) for x in points], width, bitwise)
                rejected[rank] += int((~keep).sum())
                ids, d, e = ids[keep], d[keep], e[keep]
                if not len(ids):
                    break
            survivors.extend(ids.tolist())
        assert sum(rejected.values())+len(survivors) == total
        rows.append({'family': name, 'indices': indices, 'tested': total,
                     'survivor_parameter_ids': survivors,
                     'rejection_counts': {str(k): v for k, v in rejected.items() if v}})
    if bitwise == '^':
        excluded = all(not row['survivor_parameter_ids'] for row in rows)
    else:
        neutral = mask if bitwise == '&' else 0
        excluded = (prefix_certificate(width, fn)['excluded'] and all(
            row['survivor_parameter_ids'] == ([neutral] if row['family'].startswith('affine') else [])
            for row in rows))
    return {'width': width, 'bitwise': bitwise, 'families': rows,
            'constraints': constraints if bitwise == '^' else {
                'all_inputs_inclusive': [0,mask], 'first_inputs': samples[:42],
                'remaining_order': 'ascending, skipping inputs already listed'},
            'constraint_count': len(constraints),
            'parameter_cases': sum(row['tested'] for row in rows),
            'survivors': sum(len(row['survivor_parameter_ids']) for row in rows),
            'excluded': excluded, 'seconds': round(time.monotonic()-started, 3),
            'scope': 'MAD,MAD,bitwise with live b,a,prior,shift=0,mix=b on '
                     'a valid full32 restricted domain; first-MAD supersets enumerated'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = {'prefix': prefix_certificate(),
              'last_bitwise': [search(bitwise=op) for op in ('^', '&', '|')]}
    with open(args.output, 'w') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps({'prefix_excluded': report['prefix']['excluded'],
        'last_bitwise': [{k: r[k] for k in ('bitwise','parameter_cases','survivors','excluded','seconds')}
                         for r in report['last_bitwise']]}))


if __name__ == '__main__':
    main()
