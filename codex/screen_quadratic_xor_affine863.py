"""Exhaustive necessary-condition search for affine MAD, MAD, XOR.

Relax the program to (a*x*x+b*x+c) XOR (d*x+e) modulo 2**width.
For each d,e, inputs 0,1,2 determine the quadratic as a FUNCTION:
the two possible coefficient pairs differ by 2**(width-1)*x*(x-1),
which vanishes modulo 2**width. Exclusion is therefore exhaustive.
This does not cover a quadratic first MAD, or programs with other live inputs.
"""
import argparse
import json
import time

import numpy as np


def target(x):
    return 9*((33*x+0xE9F8CC1D) ^ (16896*x+0xACCF6200))+0xFD7046C5


def control(x):
    return (7*x*x+13*x+19) ^ (33*x+42)


def quadratic_coefficients(g0, g1, g2, width):
    mask = (1 << width)-1
    delta = (g1-g0) & mask
    rhs = (g2-g0-2*delta) & mask
    if rhs & 1:
        return None
    a = rhs//2
    return a, (delta-a) & mask, g0 & mask


def search(width, fn=target, chunk_size=131072):
    if not 1 <= width <= 16:
        raise ValueError('width must be 1..16')
    if chunk_size <= 0:
        raise ValueError('chunk size must be positive')
    mask = (1 << width)-1
    outputs = [fn(x) & mask for x in range(mask+1)]
    # Early witnesses speed the search; all inputs are still checked for survivors.
    samples = list(dict.fromkeys([x & mask for x in
        [3, 4, 5, 7, 15, 31, 63, 127, 255, 511, 513, 1023, 1025]]
        + list(range(mask+1))))
    total = (mask+1)**2
    rejected = {}
    survivors = []
    started = time.monotonic()
    for start in range(0, total, chunk_size):
        ids = np.arange(start, min(start+chunk_size, total), dtype=np.int64)
        d, e = ids >> width, ids & mask
        c = outputs[0] ^ e
        delta = ((outputs[1] ^ ((d+e) & mask))-c) & mask
        rhs = ((outputs[2 & mask] ^ ((2*d+e) & mask))-c-2*delta) & mask
        keep = (rhs & 1) == 0
        rejected['odd_second_difference'] = rejected.get('odd_second_difference', 0)+int((~keep).sum())
        d, e, c, delta, rhs = [v[keep] for v in (d, e, c, delta, rhs)]
        a = rhs//2
        b = (delta-a) & mask
        for x in samples:
            if not len(d):
                break
            keep = (((a*x*x+b*x+c) & mask) ^ ((d*x+e) & mask)) == outputs[x]
            count = int((~keep).sum())
            if count:
                rejected[str(x)] = rejected.get(str(x), 0)+count
            a, b, c, d, e = [v[keep] for v in (a, b, c, d, e)]
        survivors.extend([[int(v) for v in row] for row in zip(a, b, c, d, e)])
    assert sum(rejected.values())+len(survivors) == total
    return {'width': width, 'pairs_tested': total, 'domain_size': mask+1,
            'rejection_counts': rejected, 'survivors': survivors,
            'seconds': round(time.monotonic()-started, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=11)
    parser.add_argument('--control', action='store_true')
    args = parser.parse_args()
    result = search(args.width, control if args.control else target)
    result['control'] = args.control
    if args.control:
        mask = (1 << args.width)-1
        assert any(row[3:] == [33 & mask, 42 & mask] for row in result['survivors'])
    print(json.dumps(result))


if __name__ == '__main__':
    main()
