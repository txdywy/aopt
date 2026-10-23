"""Close affine-MAD / logical right shift / MAD for the first hash slice.

The arithmetic lemmas and their scope are detailed in
SUB863_LIVE_AND_SHIFT_20261023.md. All word operations are uint32. Literal
shifts use a complete final-operand partition; variable shifts use a small
necessary-condition SMT query, with no opcode or operand selectors.
This is an exclusion certificate, not a replacement kernel or a cycle score.
"""
import argparse
from collections import Counter
import itertools
import json
import time

import z3

from codex.prove_affine_shift_permutation import h

MASK = (1 << 32) - 1
C0 = 0x7ED55D16


def shapes():
    # Input, affine first result, shifted result, two arbitrary constants.
    return list(itertools.product(itertools.combinations_with_replacement(range(5), 2),
                                  range(5)))


def classify(pair, addend):
    i, j = pair
    if 2 not in (i, j, addend):
        return 'quadratic'
    if not any(v in (0, 1) for v in (i, j, addend)):
        return 'shift_cardinality'
    if 2 in pair:
        other = j if i == 2 else i
        if other in (0, 1):
            return 'bilinear_parity'
        if other == 2:
            return 'square_shift_intervals'
        return 'scaled_shift_intervals'
    if i == 0 and j in (0, 1):
        return 'zero_input_range'
    if (i, j) == (1, 1):
        return 'square_sum'
    if i in (0, 1) and j in (3, 4):
        return 'scaled_affine_sum_or_top_pair'
    raise AssertionError((pair, addend))


def third_difference(inputs):
    y = [h(x) for x in inputs]
    return (y[3] - 3*y[2] + 3*y[1] - y[0]) & MASK


def even_multiplier_certificate():
    # For A even, a(x+Q)=a(x), where Q=2**(32-v2(A)).
    # Every final MAD therefore preserves congruence mod Q, even if it uses x.
    # The same witness x rejects every possible Q, including constant a (Q=2).
    x = ((2**19 - 1 - C0) * pow(4097, -1, 1 << 32)) & MASK
    rows = []
    for exponent in range(1, 32):
        q = 1 << exponent
        residue = (h((x+q) & MASK) - h(x)) % q
        assert residue
        rows.append({'period': q, 'residue': residue})
    return {'input': x, 'affine_hash_value': (4097*x+C0) & MASK,
            'period_witnesses': rows}


def variable_count_query(timeout_ms=10000, fn=h, nonzero=True):
    """Necessary condition when the shift count is a=A*x+B.

Whenever a>=32, *any* uint32 logical-right-shift source produces zero.
The final MAD is then a quadratic in x. Relax it to an arbitrary quadratic.
A=0 is a constant shift, handled by the literal-shift proof; hence A!=0.
SAT of this relaxation alone does not produce a three-instruction program.
"""
    if timeout_ms <= 0:
        raise ValueError('timeout must be positive')
    a, b, c0, c1, c2 = z3.BitVecs('a b c0 c1 c2', 32)
    solver = z3.SolverFor('QF_BV')
    solver.set(timeout=timeout_ms)
    if nonzero:
        solver.add(a != 0)
    samples = list(range(64)) + [x ^ (1 << 31) for x in range(64)]
    for x in samples:
        solver.add(z3.Or(z3.ULT(a*x+b, 32), c2*(x*x)+c1*x+c0 == fn(x)))
    started = time.monotonic()
    status = solver.check()
    result = {'result': str(status), 'samples': samples,
              'timeout_ms': timeout_ms, 'nonzero_multiplier': nonzero,
              'seconds': round(time.monotonic()-started, 3)}
    if status == z3.sat:
        result['relaxation_model'] = {str(v): solver.model().eval(
            v, model_completion=True).as_long() for v in (a, b, c0, c1, c2)}
    elif status == z3.unknown:
        result['reason'] = solver.reason_unknown()
    return result


def certificate(timeout_ms=10000):
    rows = [{'operands': [*pair, c], 'reason': classify(pair, c)}
            for pair, c in shapes()]
    assert len(rows) == 75
    # Input/constant shift sources are affine on one of these short progressions.
    finite_differences = {str(xs): third_difference(xs) for xs in
                          ([0, 1, 2, 3], [0, 2, 4, 6], [32, 33, 34, 35])}
    assert all(finite_differences.values())
    assert h(0) >= 1 << 31
    # A scaled shifted word plus x/a must be affine modulo 2**s.
    low_witness = (h(0) ^ h(1 << 19)) & 1
    second_difference = (h(2)-2*h(1)+h(0)) & MASK
    assert low_witness == 1 and second_difference == 12
    # For odd A, pairing x and x+2**31 toggles a's top bit. For even K,
    # K*x+(a>>s), or K*a+(a>>s), changes by +/-2**(31-s).
    target_pair_differences = {(1 << 31)-4096, (1 << 31)+4096}
    for s in range(1, 32):
        assert not target_pair_differences & {1 << (31-s), (-1 << (31-s)) & MASK}
    query = variable_count_query(timeout_ms)
    return {'target': 'H(x)=(4097*x+C0) XOR ((4097*x+C0)>>19) XOR C1; uint32',
            'scope': 'first MAD=A*x+B; arbitrary right-shift source/count and '
                     'final MAD operands from x, previous results and constants',
            'literal_shifts': '0..31 and >=32',
            'shape_counts': dict(Counter(row['reason'] for row in rows)),
            'shapes': rows, 'even_multiplier': even_multiplier_certificate(),
            'third_differences': finite_differences,
            'target_at_zero': h(0), 'low_bit_congruence_violation': low_witness,
            'second_difference': second_difference,
            'top_pair_xor': 0x80001000,
            'target_top_pair_differences': sorted(target_pair_differences),
            'variable_count_query': query, 'z3': z3.get_version_string(),
            'excluded': query['result'] == 'unsat'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout-ms', type=int, default=10000)
    parser.add_argument('--output')
    args = parser.parse_args()
    result = certificate(args.timeout_ms)
    rendered = json.dumps(result, indent=2) + '\n'
    if args.output:
        with open(args.output, 'w') as output:
            output.write(rendered)
    print(rendered, end='')


if __name__ == '__main__':
    main()
