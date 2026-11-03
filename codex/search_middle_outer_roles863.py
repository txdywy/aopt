"""Split affine MAD,XOR,MAD by XOR source and final operand roles.

This searches only t=A*x+B; u=t OP x/K for OP in XOR, AND, OR.
It is not a complete three-op
grammar. Every UNSAT row excludes its stated low-bit family. UNKNOWN is
inconclusive, and even a low-bit equivalent needs uint32 verification.
"""
import argparse
from collections import Counter
import itertools
import json
import random
import time

import z3

from codex.screen_quadratic_xor_affine863 import target


def roles():
    return [(a, b, c) for a in range(5) for b in range(a, 5)
            for c in range(5) if 2 in (a, b, c)]


def expression(x, constants, xor_source, role, bitwise='^'):
    a, b, key, k0, k1 = constants
    t = a*x+b
    other = x if xor_source == 'input' else key
    u = {'^': lambda: t ^ other, '&': lambda: t & other,
         '|': lambda: t | other}[bitwise]()
    values = (x, t, u, k0, k1)
    i, j, k = role
    return values[i]*values[j]+values[k]


def check(xor_source, role, width=11, timeout_ms=2000, fn=target,
          injective_cut=False, bitwise='^'):
    mask = (1 << width)-1
    rng = random.Random(864)
    samples = list(dict.fromkeys([x & mask for x in list(range(64))+
        [127, 255, 511, 512, 513, 1023, 1024, 1025, mask]+
        [rng.randrange(mask+1) for _ in range(32)]]))
    constants = z3.BitVecs('a b key k0 k1', width)
    solver = z3.SolverFor('QF_BV')
    solver.set(timeout=timeout_ms)
    if injective_cut:
        if fn is not target or xor_source != 'constant' or 0 in role:
            raise ValueError('injective cut requires target and a t-only output')
        # Target is a permutation: both affine multipliers are odd and
        # (33*x+B) XOR (512*33*x+D) is triangular with invertible bit diagonal.
        # With no direct x in the final MAD, its result depends only on t.
        # Therefore t=A*x+B must be injective, requiring odd A.
        solver.add(constants[0] & 1 == 1)
    for value in samples:
        x = z3.BitVecVal(value, width)
        solver.add(expression(x, constants, xor_source, role, bitwise) == fn(x))
    started = time.monotonic()
    status = solver.check()
    result = {'bitwise': bitwise, 'xor_source': xor_source, 'role': role, 'samples': len(samples),
              'injective_cut': injective_cut, 'result': str(status)}
    if status == z3.unknown:
        result['reason'] = solver.reason_unknown()
    elif status == z3.sat:
        model = solver.model()
        values = [model.eval(k, model_completion=True) for k in constants]
        result['constants'] = [v.as_long() for v in values]
        x = z3.BitVec('verify_x', width)
        verify = z3.SolverFor('QF_BV')
        verify.set(timeout=timeout_ms)
        verify.add(expression(x, values, xor_source, role, bitwise) != fn(x))
        verified = verify.check()
        result['verification'] = str(verified)
        if verified == z3.sat:
            result['counterexample'] = verify.model().eval(x).as_long()
        elif verified == z3.unknown:
            result['verification_reason'] = verify.reason_unknown()
    result['seconds'] = round(time.monotonic()-started, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=11)
    parser.add_argument('--timeout-ms', type=int, default=2000)
    parser.add_argument('--bitwise', choices=('^', '&', '|'), default='^')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 1 <= args.width <= 32 or args.timeout_ms <= 0:
        parser.error('width must be 1..32; timeout must be positive')
    rows = []
    for xor_source, role in itertools.product(('input', 'constant'), roles()):
        row = check(xor_source, role, args.width, args.timeout_ms, bitwise=args.bitwise)
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {'width': args.width, 'timeout_ms': args.timeout_ms, 'bitwise': args.bitwise,
              'z3': z3.get_version_string(), 'operand_pool': ['x', 't', 'u', 'k0', 'k1'],
              'counts': dict(Counter(row['result'] for row in rows)), 'rows': rows}
    with open(args.output, 'w') as output:
        json.dump(report, output, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
