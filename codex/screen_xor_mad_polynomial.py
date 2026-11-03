"""Necessary degree-four polynomial test for bitwise,MAD,MAD (no shifts).

First XOR/AND/OR can only produce x OP K or a constant. Constant cases are covered
by K=0. Two subsequent MADs produce a polynomial in x,y of total degree <=4.
We allow ALL 15 monomials with independent coefficients: a strict relaxation.
Exclusion at a low width excludes uint32 programs; survival proves nothing.
"""
import argparse
import json
import random
import time


def consistent(matrix, rhs, width):
    """Gaussian elimination over Z/(2**width), using least-valuation pivots."""
    mask = (1 << width) - 1
    a = [[v & mask for v in row] + [b & mask]
         for row, b in zip(matrix, rhs)]
    if not a:
        return True
    cols = len(a[0]) - 1
    row = 0
    for col in range(cols):
        best = None
        for i in range(row, len(a)):
            for j in range(col, cols):
                v = a[i][j]
                if v and (best is None or (v & -v) < best[0]):
                    best = (v & -v, i, j)
        if best is None:
            return all(a[i][-1] == 0 for i in range(row, len(a)))
        factor, i, j = best
        # Every remaining coefficient is divisible by factor. Thus every
        # remaining RHS must be too, regardless of the unknown variables.
        if any(a[k][-1] % factor for k in range(row, len(a))):
            return False
        a[row], a[i] = a[i], a[row]
        for line in a:
            line[col], line[j] = line[j], line[col]
        inverse = pow(a[row][col] // factor, -1, 1 << width)
        a[row] = [(v * inverse) & mask for v in a[row]]
        pivot = a[row]
        for k in range(row + 1, len(a)):
            multiple = a[k][col] // factor
            if multiple:
                a[k] = [(v - multiple * p) & mask
                        for v, p in zip(a[k], pivot)]
        row += 1
        if row == len(a):
            return True
    return all(a[k][-1] == 0 for k in range(row, len(a)))


def monomials(x, y, mask):
    return [(x**i * y**j) & mask for i in range(5) for j in range(5-i)]


def target(x):
    b = x ^ 0xC761C23C
    return (33*b + 0xE9F8CC1D) ^ (16896*b + 0xACCF6200)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=11)
    parser.add_argument('--control', action='store_true')
    parser.add_argument('--slice', choices=('absorb-c1', 'middle'), default='absorb-c1')
    parser.add_argument('--bitwise', choices=('^', '&', '|'), default='^')
    args = parser.parse_args()
    if not 1 <= args.width <= 12:
        parser.error('width must be 1..12')
    mask = (1 << args.width) - 1
    rng = random.Random(900)
    samples = sorted(set(list(range(min(64, mask+1))) +
                         [rng.randrange(mask+1) for _ in range(64)]))
    def selected_target(x):
        if args.slice == 'middle':
            return 9*((33*x+0xE9F8CC1D) ^ (16896*x+0xACCF6200))+0xFD7046C5
        return target(x)
    def bitwise(x, key):
        return {'^': lambda: x ^ key, '&': lambda: x & key,
                '|': lambda: x | key}[args.bitwise]()
    rhs = [((x * bitwise(x, 37) + 19)**2 + x if args.control else selected_target(x)) & mask
           for x in samples]
    start = time.monotonic()
    survivors = []
    for key in range(mask+1):
        matrix = [monomials(x, bitwise(x, key), mask) for x in samples]
        if consistent(matrix, rhs, args.width):
            survivors.append(key)
    if args.control:
        assert (37 & mask) in survivors
    print(json.dumps({'width': args.width, 'slice': args.slice, 'samples': len(samples),
                      'bitwise': args.bitwise,
                      'tested_keys': mask+1, 'survivors': survivors,
                      'control': args.control,
                      'seconds': round(time.monotonic()-start, 3)}))


if __name__ == '__main__':
    main()
