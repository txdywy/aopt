"""Necessary low-bit screen allowing live values from the first hash stage.

Restrict full32 a=4097*prior+C0 to 0..2**w-1, w<=19. Then a>>19=0
and b=a XOR C1. The available low-bit words are b,a,prior,zero; mix=b.
Every b low word occurs, and prior follows by the odd inverse of 4097.
Consequently low-bit UNSAT excludes the stated full32 three-op grammar
even with these additional live inputs. SAT only gives a low-bit candidate;
it must not be promoted to full32 or a kernel score.
"""
import argparse
import json
import random
import time

import z3

from codex.synthesize_hash_cegis import OPS, apply, choose


def live_words(x, width):
    a = x ^ z3.BitVecVal(0xC761C23C & ((1 << width)-1), width)
    prior = (a-0x7ED55D16)*pow(4097, -1, 1 << width)
    return [x, a, prior, z3.BitVecVal(0, width)]


def target(x):
    return 9*((33*x+0xE9F8CC1D) ^ (16896*x+0xACCF6200))+0xFD7046C5


def search(sequence, width=12, timeout_ms=5000, control=False, pin_control=False,
           target_fn=None, last_mad_role=None, middle_uses_first=False,
           middle_other=None):
    if not 1 <= width <= 19:
        raise ValueError('width must be 1..19 for the zero-SHR necessary condition')
    if len(sequence) != 3 or any(op not in OPS or op in ('<<', '>>') for op in sequence):
        raise ValueError('requires three nonshift ISA operations')
    if timeout_ms <= 0:
        raise ValueError('timeout must be positive')
    if pin_control and (not control or list(sequence) != ['mad', 'mad', '^']):
        raise ValueError('pinned control requires control and mad,mad,^')
    if target_fn is not None and (control or pin_control):
        raise ValueError('custom target and built-in controls are mutually exclusive')
    if last_mad_role is not None:
        if (sequence[-1] != 'mad' or len(last_mad_role) != 3 or
                any(not 0 <= i < 8 for i in last_mad_role) or last_mad_role[0] > last_mad_role[1]):
            raise ValueError('last MAD role must be three canonical final-pool indices')
    if middle_uses_first and (sequence[0] != 'mad' or sequence[1] not in ('^','&','|')
                             or sequence[2] != 'mad'):
        raise ValueError('middle split requires MAD,bitwise,MAD')
    if middle_other is not None and (not middle_uses_first or middle_other not in (0,1,2,5)):
        raise ValueError('middle source split requires first-result use and b/a/prior/K')
    solver = z3.SolverFor('QF_BV')
    solver.set(timeout=timeout_ms)
    spec = []
    for i, op in enumerate(sequence):
        a, b, c = z3.BitVecs(f'a{i} b{i} c{i}', 4)
        k0, k1 = z3.BitVecs(f'k{i}_0 k{i}_1', width)
        for selector in (a, b, c):
            solver.add(z3.ULT(selector, i+6))
        if op in ('mad', '+', '*', '^', '&', '|'):
            solver.add(z3.ULE(a, b))
        if op != 'mad':
            solver.add(c == 0)
        spec.append((op, a, b, c, k0, k1))
    if last_mad_role is not None:
        solver.add(*(v == index for v,index in zip(spec[-1][1:4],last_mad_role)))
    if middle_uses_first:
        a,b = spec[1][1:3]
        # t OP b/a/prior/K. Constants are interchangeable and OP is commutative.
        # t OP t and bypassing t reduce to polynomial/prefix-bitwise cases.
        solver.add(z3.Or(*(z3.And(a == i,b == j) for i,j in
                          ((0,4),(1,4),(2,4),(4,5)))))
        if middle_other is not None:
            i,j=sorted((4,middle_other))
            solver.add(a == i,b == j)
    if pin_control:
        # Validate the encoding and full-domain verifier on a known program.
        # This is a supplied positive witness, not a discovered optimization.
        witness = [(0,4,5,33,17), (1,5,6,512,29), (4,5,0,0,0)]
        for row, values in zip(spec, witness):
            solver.add(*(var == value for var, value in zip(row[1:], values)))

    def program(x, model=None):
        values = live_words(x, width)
        for op, a, b, c, k0, k1 in spec:
            if model is not None:
                a, b, c, k0, k1 = [model.eval(v, model_completion=True) for v in (a,b,c,k0,k1)]
            pool = values+[k0, k1]
            values.append(apply(z3.BitVecVal(OPS.index(op), 4), choose(a,pool),
                                choose(b,pool), choose(c,pool)))
        return z3.simplify(values[-1])

    def reference(x):
        if target_fn is not None:
            return target_fn(x)
        if control:
            a = live_words(x, width)[1]
            return (33*x+17) ^ (512*a+29)
        return target(x)

    rng = random.Random(863)
    samples = list(dict.fromkeys(list(range(min(32,1<<width)))+
                                [rng.randrange(1<<width) for _ in range(32)]))
    for value in samples:
        x = z3.BitVecVal(value, width)
        solver.add(program(x) == reference(x))
    started = time.monotonic()
    status = solver.check()
    result = {'sequence': sequence, 'width': width, 'samples': len(samples),
              'timeout_ms': timeout_ms, 'control': control, 'pin_control': pin_control,
              'custom_target': target_fn is not None,
              'last_mad_role': last_mad_role, 'middle_uses_first': middle_uses_first,
              'middle_other': middle_other,
              'result': str(status),
              'live_pool': ['b', 'a=b XOR C1', 'prior=(a-C0)/4097', 'zero']}
    if status == z3.unknown:
        result['reason'] = solver.reason_unknown()
    elif status == z3.sat:
        model = solver.model()
        x = z3.BitVec('verify_x', width)
        verify = z3.SolverFor('QF_BV')
        verify.set(timeout=timeout_ms)
        verify.add(program(x, model) != reference(x))
        verified = verify.check()
        result['verification'] = str(verified)
        result['program'] = [{'op': row[0], 'selectors_constants':
                              [model.eval(v,model_completion=True).as_long() for v in row[1:]]}
                             for row in spec]
        if verified == z3.sat:
            result['counterexample'] = verify.model().eval(x).as_long()
    result['seconds'] = round(time.monotonic()-started,3)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--timeout-ms', type=int, default=5000)
    parser.add_argument('--width', type=int, default=12)
    parser.add_argument('--control', action='store_true')
    parser.add_argument('--pin-control', action='store_true')
    args = parser.parse_args()
    print(json.dumps(search(args.sequence.split(','), args.width, args.timeout_ms,
                            args.control, args.pin_control)))


if __name__ == '__main__':
    main()
