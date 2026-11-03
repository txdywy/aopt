"""Search the XOR-plus-polynomial relaxation in a narrower AND basis.

(P XOR Q) + R = S - 2*(P AND Q), S=P+Q+R. P is affine, Q quadratic,
R/S degree at most four. Thus the AND operands only need width-1 bits.
UNSAT is a necessary-family exclusion; a reduced SAT is not executable code.
"""
import argparse
import json
from pathlib import Path
import time

import z3

from codex.search_cross_encoding863 import C1, middle, necessary_inputs
from codex.search_head_polynomial863 import (canonical_poly, control_coefficients,
    evaluate as xor_evaluate, interpolate_falling, normalize, poly)


def convert(coefficients, width):
    p, q, r = (coefficients[k] for k in ('p', 'q', 'r'))
    s = [r[i] + (p[i] if i < len(p) else 0) + (q[i] if i < len(q) else 0)
         for i in range(5)]
    return dict(p=canonical_poly(p, width-1), q=canonical_poly(q, width-1),
                s=canonical_poly(s, width))


def evaluate(x, coefficients, width):
    mask = (1 << (width-1))-1
    p = poly(x, coefficients['p']) & mask
    q = poly(x, coefficients['q']) & mask
    return (poly(x, coefficients['s']) - 2*(p & q)) & ((1 << width)-1)


def search(width=11, seconds=20, input_key=C1, fixed_output_key=None,
           control=False, pin_control=False):
    if not 5 <= width <= 16 or seconds <= 0:
        raise ValueError('width 5..16 and positive time limit required')
    if pin_control and not control:
        raise ValueError('pinning requires a positive control')
    if not 0 <= input_key <= 0xffffffff or (fixed_output_key is not None
            and not 0 <= fixed_output_key <= 0xffffffff):
        raise ValueError('uint32 endpoint keys required')
    mask, high = (1 << width)-1, 1 << (width-1)
    small = width-1
    solver = z3.SolverFor('QF_BV')
    p = [z3.BitVec('and_p'+str(i), small) for i in range(2)]
    q = [z3.BitVec('and_q'+str(i), small) for i in range(3)]
    solver.add(p[0] & 1 == 0, z3.ULT(q[2], 1 << (small-1)))
    outkey = z3.BitVec('and_output', width)
    if fixed_output_key is None:
        solver.add(outkey & (high | 1) == 0)
    else:
        solver.add(outkey == fixed_output_key & mask)
    witness = control_coefficients('quadratic_xor_bypass')

    def target(x):
        return ((xor_evaluate(x, witness, 'quadratic_xor_bypass', width)
                 ^ (fixed_output_key or 0)) if control else middle(x ^ input_key)) & mask

    def product_term(x):
        a = z3.BitVecVal(x, small)
        return 2*z3.ZeroExt(1, poly(a, p) & poly(a, q))

    samples = list(dict.fromkeys(x & mask for x in necessary_inputs(width)+[32, 64, 96]))
    values = [(z3.BitVecVal(target(x), width) ^ outkey) + product_term(x) for x in range(5)]
    s, valid = interpolate_falling(values, width)
    solver.add(valid)
    if pin_control:
        normalized, key = normalize(witness, (fixed_output_key or 0) & mask,
            'quadratic_xor_bypass', width, fixed_output=fixed_output_key is not None)
        converted = convert(normalized, width)
        solver.add(outkey == key)
        for name, variables in (('p', p), ('q', q), ('s', s)):
            solver.add(*(v == n for v, n in zip(variables, converted[name])))

    def constrain(x):
        solver.add(poly(z3.BitVecVal(x, width), s)-product_term(x)
                   == (z3.BitVecVal(target(x), width) ^ outkey))

    for x in samples:
        constrain(x)
    started = time.monotonic()
    report = dict(width=width, input_key=input_key, fixed_output_key=fixed_output_key,
        control=control, pin_control=pin_control, seconds_limit=seconds,
        initial_samples=samples[:], events=[], result='deadline', new_executable=False,
        scope='Independent affine P, quadratic Q and degree-four S, output S-2*(P AND Q). '
              'Equivalent relaxation to (affine XOR quadratic)+degree-four polynomial; '
              'narrowed operands are sound only because multiplication by two follows.')
    while time.monotonic()-started < seconds:
        solver.set(timeout=max(1, int(1000*(seconds-time.monotonic()+started))))
        status = solver.check()
        event = dict(samples=len(samples), synthesis=str(status))
        report['events'].append(event)
        if status != z3.sat:
            report['result'] = 'relaxed_family_unsat' if status == z3.unsat else 'unknown'
            if status == z3.unknown:
                report['reason'] = solver.reason_unknown()
            break
        model = solver.model()
        coefficients = {name: [model.eval(v, model_completion=True).as_long() for v in variables]
                        for name, variables in (('p', p), ('q', q), ('s', s))}
        key = model.eval(outkey, model_completion=True).as_long()
        report.update(coefficients=coefficients, output_key=key)
        bad = [x for x in range(1 << width)
               if evaluate(x, coefficients, width) != (target(x) ^ key)][:8]
        if not bad:
            report.update(result='equivalent_on_relaxed_low_domain', verified_inputs=1 << width)
            break
        event['counterexamples'] = bad
        for x in bad:
            assert x not in samples
            samples.append(x)
            constrain(x)
    report['seconds'] = round(time.monotonic()-started, 3)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=11)
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--input-key', type=lambda x: int(x, 0), default=C1)
    parser.add_argument('--output-key', type=lambda x: int(x, 0))
    parser.add_argument('--control', action='store_true')
    parser.add_argument('--pin-control', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = search(args.width, args.seconds, args.input_key, args.output_key,
                    args.control, args.pin_control)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('result', 'seconds', 'width', 'control')}))


if __name__ == '__main__':
    main()
