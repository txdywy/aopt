"""Necessary operand-domination test for MAD,MAD,AND/OR.

For Q AND P=F, P must contain every set bit of F. For Q OR P=F,
P can contain only set bits of F. A one-input first MAD is affine,
x*x+K or x*x+x; the final operand besides the second result is one
of those forms, x, or a constant. Enumerate their necessary conditions.
Only neutral constants survive for this target, leaving Q=F; an
independent degree-four polynomial consistency check excludes that too.
"""
import argparse
import json
import time

import numpy as np

from codex.screen_quadratic_xor_affine863 import target
from codex.screen_xor_mad_polynomial import consistent


def allows(p, output, mask, bitwise):
    if bitwise == '&':
        return (output & (mask ^ p)) == 0
    if bitwise == '|':
        return (p & (mask ^ output)) == 0
    raise ValueError('requires AND or OR')


def search(width=11, bitwise='&', fn=target, chunk_size=131072):
    if not 2 <= width <= 12 or chunk_size <= 0:
        raise ValueError('width must be 2..12 and chunk size positive')
    mask = (1 << width)-1
    outputs = [fn(x) & mask for x in range(mask+1)]
    samples = list(dict.fromkeys(list(range(min(32, mask+1)))+
        [x & mask for x in [63, 127, 255, 511, 513, 1023, 1025, 2047]]+
        list(range(mask+1))))
    started = time.monotonic()
    affine = []
    total = (mask+1)**2
    for base in range(0, total, chunk_size):
        ids = np.arange(base, min(base+chunk_size, total), dtype=np.int64)
        a, b = ids >> width, ids & mask
        for x in samples:
            if not len(a):
                break
            keep = allows((a*x+b) & mask, outputs[x], mask, bitwise)
            a, b = a[keep], b[keep]
        affine.extend([[int(aa), int(bb)] for aa, bb in zip(a, b)])
    square = []
    for key in range(mask+1):
        if all(allows((x*x+key) & mask, outputs[x], mask, bitwise) for x in samples):
            square.append(key)
    square_input = all(allows((x*x+x) & mask, outputs[x], mask, bitwise) for x in samples)
    # The second MAD is at most degree four even for a quadratic first MAD.
    poly_samples = sorted(set([x & mask for x in list(range(64))+
        [127, 255, 511, 512, 513, 1023, 1024, 1025, 2047]]))
    matrix = [[pow(x, degree, mask+1) for degree in range(5)] for x in poly_samples]
    polynomial_consistent = consistent(matrix, [outputs[x] for x in poly_samples], width)
    neutral = mask if bitwise == '&' else 0
    excluded = (affine == [[0, neutral]] and not square and not square_input
                and not polynomial_consistent)
    return {'width': width, 'bitwise': bitwise, 'affine_pairs_tested': total,
            'affine_survivors': affine, 'square_keys_tested': mask+1,
            'square_survivors': square, 'square_input_survives': square_input,
            'degree_four_samples': poly_samples,
            'degree_four_consistent': polynomial_consistent,
            'excluded': excluded, 'seconds': round(time.monotonic()-started, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=11)
    parser.add_argument('--bitwise', choices=('&', '|'), required=True)
    args = parser.parse_args()
    print(json.dumps(search(args.width, args.bitwise)))


if __name__ == '__main__':
    main()
