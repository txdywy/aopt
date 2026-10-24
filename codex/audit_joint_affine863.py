"""A full32 three-point certificate for constant-factor MAD plus one SHR16.

With affine live inputs and exactly one right shift, every such program
has the form P*b + Q*((A*b+B mod 2**32)>>16) + R. On inputs separated
by 2**16 its low 16 bits are affine in the input number. The real target
violates that necessary condition. This does not cover a variable MAD
multiplier, XOR/AND/OR, multiple shifts, or non-affine earlier live inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from codex.search_joint_tail863 import DIRECT_STAGES, reference_value


def certificate():
    inputs = [0, 65536, 131072]
    outputs = [reference_value(b, 'b') for b in inputs]
    difference = (outputs[2]-2*outputs[1]+outputs[0]) & 65535
    assert difference != 0
    return dict(width=32, shift=16, inputs=inputs, outputs=outputs,
        low16_outputs=[y & 65535 for y in outputs],
        target_second_difference_mod65536=difference,
        required_second_difference=0,
        scope='Any number of constant-factor MADs and exactly one SHR16; '
              'all initial live values affine in b, with arbitrary constants',
        new_executable=False)


def refine(paths):
    proof = certificate()
    rows = []
    for path in paths:
        report = json.loads(Path(path).read_text())
        for row in report['rows']:
            if row['stage'] not in DIRECT_STAGES or row['literal_shift'] != 16:
                raise ValueError('certificate requires direct-b inputs and SHR16')
            order = row['sequence']
            applies = set(order) <= {'mad', '>>'} and order.count('>>') == 1
            if applies and row['result'] == 'proved_equivalent_full32':
                raise RuntimeError('positive result contradicts the certificate')
            rows.append(dict(source=str(path), sequence=order,
                original_result=row['result'], result=(
                    'excluded_by_affine_fold_certificate' if applies else row['result']),
                arbitrary_constant_factors=row['arbitrary_mad_factors'],
                extra_arbitrary_constants=row['free_constant_count']))
    return dict(certificate=proof, rows=rows,
        totals=dict(Counter(row['result'] for row in rows)), new_executable=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('inputs', nargs='*', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    report = refine(args.inputs) if args.inputs else certificate()
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report.get('totals', report)))


if __name__ == '__main__':
    main()
