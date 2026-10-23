"""Necessary polynomial relaxation for a quadratic MAD followed by MAD,XOR.

With one input x, a non-affine first MAD is x*x+K or x*x+x.
The second MAD is a polynomial Q of degree <=4. The final XOR's other
operand is x, a constant, or the first MAD. We allow independent Q
coefficients, a strict relaxation. Modular inconsistency excludes programs;
consistency alone does not establish a realizable program.
"""
import argparse
import json
import time

from codex.screen_quadratic_xor_affine863 import target
from codex.screen_xor_mad_polynomial import consistent


def candidates(width):
    yield 'input', 0, lambda x: x
    yield 'square-input', 0, lambda x: x*x+x
    for key in range(1 << width):
        yield 'constant', key, lambda x, key=key: key
        yield 'square-constant', key, lambda x, key=key: x*x+key


def search(width=11, fn=target):
    mask = (1 << width)-1
    samples = sorted(set([x & mask for x in
        list(range(64))+[127, 255, 511, 512, 513, 1023, 1024, 1025, 2047]]))
    matrix = [[pow(x, degree, mask+1) for degree in range(5)] for x in samples]
    counts = {}
    survivors = []
    started = time.monotonic()
    for kind, key, other in candidates(width):
        counts[kind] = counts.get(kind, 0)+1
        rhs = [(fn(x) ^ other(x)) & mask for x in samples]
        if consistent(matrix, rhs, width):
            survivors.append({'kind': kind, 'key': key})
    return {'width': width, 'samples': samples, 'tested': counts,
            'survivors': survivors, 'seconds': round(time.monotonic()-started, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=11)
    args = parser.parse_args()
    if not 1 <= args.width <= 12:
        parser.error('width must be 1..12')
    print(json.dumps(search(args.width)))


if __name__ == '__main__':
    main()
