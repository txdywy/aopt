"""Screen affine-first bitwise-prefix polynomial tails on multiples of eight.

Restrict x=8*y modulo 2048. Then t=A*x+B has upper eight bits
(A*y+floor(B/8)) modulo 256 and constant low three bits. Enumerating 256**2
pairs therefore covers all original A,B. Bit-separated rectangles preserve
their axes under arbitrary bitwise prefixes over x,t, and constants.
Every subsequent integer polynomial has mixed difference divisible by h*k
in ORIGINAL x units. Output-key low bits cancel; all 256 upper keys are tested.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
import zlib

import numpy as np

from codex.search_cross_encoding863 import C1, middle

WIDTH, SHIFT = 11, 3
SIZE, MASK = 256, 255


def rectangles():
    axes = [(1 << i, 1 << j) for i in range(8) for j in range(i+1, 8) if i+j >= 5]
    axes.sort(key=lambda pair: (pair[0]*pair[1], pair[1]-pair[0]))
    for h, k in axes:
        for base in range(SIZE):
            if base & (h | k) or base+h+k >= SIZE:
                continue
            yield base, h, k


def axis_separated(a, b, base, h, k):
    values = [(a*(base+offset)+b) & MASK for offset in (0, h, k, h+k)]
    p, q, r, s = values
    return ((p ^ q ^ r ^ s) == 0) & (((p ^ q) | h) & ((p ^ r) | k) == 0)


def control(x):
    t = (5*x+19) & 0xffffffff
    q = t ^ 0xdeadbeef
    return (q*q*x + 17*q*x + 23*x*x + 31) & 0xffffffff


def search(positive=False, residue=0):
    if residue not in range(8):
        raise ValueError('residue must be in 0..7')
    target = control if positive else lambda x: middle(x ^ C1)
    values = np.array([(target((y << SHIFT)+residue) >> SHIFT) & MASK for y in range(SIZE)], dtype=np.int64)
    # Low three target bits are constant on this restricted input domain.
    assert len({target((y << SHIFT)+residue) & 7 for y in range(SIZE)}) == 1
    ids = np.arange(SIZE*SIZE, dtype=np.int64)
    a, b = ids >> 8, ids & MASK
    keys = np.arange(SIZE, dtype=np.int64)
    history = []
    total, bad = 0, 0
    started = time.monotonic()
    for base, h, k in rectangles():
        total += 1
        corners = [int(values[base+d]) for d in (0, h, k, h+k)]
        p, q, r, s = [v ^ keys for v in corners]
        deltas = (s-q-r+p) & MASK
        if np.any(deltas == 0):
            continue
        bad += 1
        separated = axis_separated(a, b, base, h, k)
        removed = int(np.count_nonzero(separated))
        if removed:
            history.append(dict(base_y=base, h_y=h, k_y=k,
                original_x_corners=[((base+d) << SHIFT)+residue for d in (0, h, k, h+k)],
                target_upper_values=corners, target_mixed_differences=sorted(set(map(int, deltas))),
                removed_pairs=removed, before=len(a), after=len(a)-removed))
            a, b = a[~separated], b[~separated]
        if not len(a):
            break
    return dict(width=WIDTH, input_stride=1 << SHIFT, upper_width=8, residue=residue,
        control=positive, input_key=C1 if not positive else None,
        total_affine_pairs=SIZE*SIZE, remaining_affine_pairs=len(a),
        rectangles_tested=total, universally_bad_rectangles=bad, witnesses=history,
        surviving_pairs=[[int(x), int(y)] for x, y in zip(a, b)],
        positive_pair_retained=bool(np.any((a == 5) & (b == ((5*residue+19) >> SHIFT)))) if positive else None,
        result='affine_prefix_polynomial_tail_excluded' if not len(a) else 'necessary_pairs_survive',
        seconds=round(time.monotonic()-started, 3), new_executable=False,
        scope='First t=A*x+B, arbitrary XOR/AND/OR prefix over x,t and constants, then '
              'arbitrary integer-polynomial continuation. Only universally bad output-key '
              'rectangles are used; surviving pairs are necessary conditions, not candidates.')


def joint_search(positive=False):
    """Intersect residue screens in the original 11-bit A,B parameter space."""
    ids = np.arange(1 << (2*WIDTH), dtype=np.int64)
    a, b = ids >> WIDTH, ids & ((1 << WIDTH)-1)
    reports = []
    for residue in range(8):
        row = search(positive, residue)
        allowed = np.zeros((SIZE, SIZE), dtype=np.bool_)
        for aa, bb in row['surviving_pairs']:
            allowed[aa, bb] = True
        selected = allowed[a & MASK, ((a*residue+b) >> SHIFT) & MASK]
        a, b = a[selected], b[selected]
        row.pop('surviving_pairs')
        row['joint_remaining_full_pairs'] = len(a)
        reports.append(row)
        if not len(a):
            break
    return dict(width=WIDTH, control=positive, total_full_pairs=1 << (2*WIDTH),
        remaining_full_pairs=len(a), residues=reports,
        surviving_pairs=[[int(aa), int(bb)] for aa, bb in zip(a, b)],
        positive_pair_retained=bool(np.any((a == 5) & (b == 19))) if positive else None,
        result='affine_prefix_polynomial_tail_excluded' if not len(a) else 'necessary_pairs_survive',
        new_executable=False, scope=reports[0]['scope'])


def compact_pairs(result):
    """Store an exact membership bitmap instead of millions of JSON integers."""
    pairs = np.asarray(result.pop('surviving_pairs'), dtype=np.int64).reshape((-1, 2))
    bits = 11 if 'total_full_pairs' in result else 8
    membership = np.zeros(1 << (2*bits), dtype=np.uint8)
    membership[(pairs[:, 0] << bits) | pairs[:, 1]] = 1
    raw = np.packbits(membership, bitorder='little').tobytes()
    result['pair_membership'] = dict(coefficient_bits=bits, bitorder='little',
        pairs=len(pairs), sha256=hashlib.sha256(raw).hexdigest(),
        zlib_base64=base64.b64encode(zlib.compress(raw, 9)).decode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', action='store_true')
    parser.add_argument('--residue', type=int, default=0)
    parser.add_argument('--joint', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = joint_search(args.control) if args.joint else search(args.control, args.residue)
    compact_pairs(result)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result.get(k) for k in ('result', 'remaining_affine_pairs',
        'remaining_full_pairs', 'universally_bad_rectangles', 'rectangles_tested',
        'seconds', 'positive_pair_retained')}))


if __name__ == '__main__':
    main()
