"""Try changing all constants across the middle-hash encoding boundaries.

Searches G*((A*x+B) XOR (D*x+E))+H, with all six coefficients free.
The target is F(x XOR input_key) XOR output_key. All operations in this
search commute with low-bit projection; UNSAT excludes this DAG at uint32.
A small-width SAT is only a small-width program, never a kernel candidate.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import random
import time

import z3


C1 = 0xC761C23C
C5 = 0xB55A4F09
# L(x)=x XOR (x>>16) is an involution at width 32. Applying this key before
# the final L has exactly the effect of applying C5 after L.
PRE_C5 = C5 ^ (C5 >> 16)


def middle(x):
    return 9*((33*x+0xE9F8CC1D) ^ (16896*x+0xACCF6200))+0xFD7046C5


def program(x, coefficients, width):
    a, b, d, e, g, h = coefficients
    mask = (1 << width)-1
    return (g*(((a*x+b) & mask) ^ ((d*x+e) & mask))+h) & mask


def normalize(coefficients, width):
    """Canonicalize a bijective parallel-affine/XOR/affine program exactly."""
    a, b, d, e, g, h = coefficients
    mask = (1 << width)-1
    if not (g & 1) or (a & 1) == (d & 1):
        raise ValueError('program is not bijective: its output parity is constant')
    if d & 1:
        a, b, d, e = d, e, a, b
    if b & 1:
        # Complementing both XOR operands preserves the XOR result.
        a, b, d, e = -a, -b-1, -d, -e-1
    return tuple(v & mask for v in (a, b, d, e, g, h))


def necessary_inputs(width):
    rng = random.Random(864)
    mask = (1 << width)-1
    return list(dict.fromkeys([x & mask for x in
        [*range(32), 256, 264, 511, 512, 513, 1023, 1024, 1025,
         *(rng.randrange(1 << width) for _ in range(20))]]))


def search(width=11, input_key=0, output_key=0, timeout_ms=5000,
           control_coefficients=None, target_values=None):
    if not 1 <= width <= 12:
        raise ValueError('this low-bit search supports widths 1..12')
    if timeout_ms <= 0:
        raise ValueError('timeout must be positive')
    if control_coefficients is not None and target_values is not None:
        raise ValueError('choose one custom reference')
    mask = (1 << width)-1
    if target_values is not None:
        if len(target_values) != 1 << width:
            raise ValueError('custom target must specify the complete domain')
        values = [v & mask for v in target_values]
    elif control_coefficients is not None:
        values = [program(x, control_coefficients, width) for x in range(1 << width)]
    else:
        values = [(middle(x ^ input_key) ^ output_key) & mask
                  for x in range(1 << width)]
    if len(set(values)) != 1 << width:
        raise ValueError('the parity cuts require a bijective target')

    a, b, d, e, g = z3.BitVecs('cross_a cross_b cross_d cross_e cross_g', width)
    # x=0 determines the sixth otherwise-arbitrary coefficient exactly.
    h = z3.BitVecVal(values[0], width)-g*(b ^ e)
    solver = z3.SolverFor('QF_BV')
    solver.add(a & 1 == 1, d & 1 == 0, g & 1 == 1, b & 1 == 0)
    samples = necessary_inputs(width)

    def constrain(x):
        solver.add(g*((a*x+b) ^ (d*x+e))+h == values[x])

    for x in samples:
        constrain(x)
    started = time.monotonic()
    result = dict(width=width, input_key=input_key, output_key=output_key,
        control_coefficients=control_coefficients, custom_table=target_values is not None,
        timeout_ms=timeout_ms, samples=samples, rounds=0,
        family='G*((A*x+B) XOR (D*x+E))+H; six arbitrary coefficients',
        cuts=['G odd', 'swap XOR arms so A odd and D even',
              'complement both arms so B even', 'infer H exactly from x=0'])
    while True:
        left = timeout_ms-int(1000*(time.monotonic()-started))
        if left <= 0:
            result.update(result='unknown', reason='total search timeout')
            break
        solver.set(timeout=left)
        status = solver.check()
        result['rounds'] += 1
        if status != z3.sat:
            result['result'] = str(status)
            if status == z3.unknown:
                result['reason'] = solver.reason_unknown()
            break
        model = solver.model()
        coefficients = [model.eval(v, model_completion=True).as_long()
                        for v in (a, b, d, e, g, h)]
        # Independent Python evaluation over every low word, rather than
        # trusting the sample model or the symbolic expression builder.
        counterexample = next((x for x, y in enumerate(values)
            if program(x, coefficients, width) != y), None)
        if counterexample is None:
            result.update(result='sat', coefficients=coefficients,
                verification='exhaustive_low_domain', verified_inputs=len(values))
            break
        assert counterexample not in samples
        samples.append(counterexample)
        constrain(counterexample)
    result['seconds'] = round(time.monotonic()-started, 3)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--timeout-ms', type=int, default=5000)
    args = ap.parse_args()
    report = dict(z3=z3.get_version_string(),
        executable_best_cycles=864, new_executable=False,
        pre_final_c5_key=PRE_C5, experiments=[], controls=[],
        scope='This fixed four-instruction DAG, one input, arbitrary constants. '
              'No general four-op or whole-hash lower bound.')
    targets = [('absorb_c1', C1, 0), ('absorb_c5', 0, PRE_C5),
               ('absorb_both', C1, PRE_C5)]
    for (name, ik, ok), width in itertools.product(targets, (11, 12)):
        row = search(width, ik, ok, args.timeout_ms)
        row['name'] = name
        report['experiments'].append(row)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:row[k] for k in ('name','width','result','seconds')}), flush=True)
    for coefficients in ((33,0xE9F8CC1D,16896,0xACCF6200,9,0xFD7046C5),
                         (21,19,128,34,5,23), (64,91,11,27,3,47)):
        row = search(11, timeout_ms=args.timeout_ms, control_coefficients=coefficients)
        report['controls'].append(row)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(control=True, result=row['result'], seconds=row['seconds'])), flush=True)


if __name__ == '__main__':
    main()
