"""Exact reduced search for D*((A*x+B) XOR K)+C.

Requires a bijective target whose low m bits are affine with odd slope.
Thus A,D are odd, and XOR K itself must be affine modulo 2**m.
Evaluating that affine map at 0,1,2,4,... proves K's low m-1 bits
are identical. Complementing K and negating D normalizes them to zero;
flipping K's top bit is absorbed into C. Enumerate remaining K, odd A,B.
Inputs 0,1 uniquely determine D,C since the XOR difference is odd.
"""
import argparse
import json
import time

import numpy as np

from codex.screen_quadratic_xor_affine863 import target


def search(width=11, affine_bits=9, fn=target, chunk_size=131072):
    if not 2 <= affine_bits <= width <= 16 or chunk_size <= 0:
        raise ValueError('require 2 <= affine_bits <= width <= 16 and positive chunk size')
    modulus = 1 << width
    mask = modulus-1
    outputs = [fn(x) & mask for x in range(modulus)]
    low_mask = (1 << affine_bits)-1
    slope = (outputs[1]-outputs[0]) & low_mask
    if not (slope & 1 and len(set(outputs)) == modulus and
            all((y-slope*x-outputs[0]) & low_mask == 0 for x, y in enumerate(outputs))):
        raise ValueError('target must be a permutation with odd affine low bits')
    inverse = np.zeros(modulus, dtype=np.int64)
    for odd in range(1, modulus, 2):
        inverse[odd] = pow(odd, -1, modulus)
    samples = list(dict.fromkeys([x & mask for x in
        [2, 3, 4, 5, 7, 15, 31, 127, 255, 511, 513, 1023, 1025]]+list(range(modulus))))
    keys = list(range(0, modulus//2, 1 << (affine_bits-1)))
    pairs_per_key = modulus*modulus//2
    rejected, survivors = {}, []
    started = time.monotonic()
    for key in keys:
        for start in range(0, pairs_per_key, chunk_size):
            ids = np.arange(start, min(start+chunk_size, pairs_per_key), dtype=np.int64)
            a, b = 2*(ids >> width)+1, ids & mask
            u0 = b ^ key
            delta = ((((a+b) & mask) ^ key)-u0) & mask
            d = ((outputs[1]-outputs[0])*inverse[delta]) & mask
            c = (outputs[0]-d*u0) & mask
            for x in samples:
                if not len(a):
                    break
                keep = ((d*(((a*x+b) & mask) ^ key)+c) & mask) == outputs[x]
                count = int((~keep).sum())
                if count:
                    rejected[str(x)] = rejected.get(str(x), 0)+count
                a, b, d, c = [v[keep] for v in (a, b, d, c)]
            survivors.extend([[int(aa), int(bb), key, int(dd), int(cc)]
                              for aa, bb, dd, cc in zip(a, b, d, c)])
    tested = pairs_per_key*len(keys)
    assert sum(rejected.values())+len(survivors) == tested
    return {'width': width, 'affine_bits': affine_bits, 'low_slope': slope,
            'low_intercept': outputs[0] & low_mask, 'normalized_keys': keys,
            'pairs_tested': tested, 'survivors': survivors,
            'rejection_counts': rejected, 'seconds': round(time.monotonic()-started, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=11)
    parser.add_argument('--affine-bits', type=int, default=9)
    args = parser.parse_args()
    print(json.dumps(search(args.width, args.affine_bits)))


if __name__ == '__main__':
    main()
