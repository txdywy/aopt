"""Full32 necessary polynomial screen for SHR followed by two MADs.

The shift is evaluated as a uint32 operation on the full concrete input.
Two arbitrary MADs then lie in the 15-dimensional degree-four polynomial
space in x and q. Inconsistency of that relaxation excludes the template.
No narrowed-word right shift is used anywhere in this screen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from codex.screen_xor_mad_polynomial import consistent, monomials


MASK = (1 << 32)-1


def target(x):
    return (9*(((33*x+0xE9F8CC1D) & MASK) ^
               ((16896*x+0xACCF6200) & MASK))+0xFD7046C5) & MASK


def full_shift(x, amount):
    if not 0 <= x <= MASK or not 0 <= amount <= MASK:
        raise ValueError('uint32 operands required')
    return x >> amount if amount < 32 else 0


def samples_for(shift):
    if not 0 <= shift <= 32:
        raise ValueError('shift must be one of the 33 representative counts')
    rng = random.Random(864+shift)
    return sorted(set([*range(64), 256, 264, 512, 513, 1024, 1032] +
        [rng.getrandbits(32) for _ in range(32)] +
        [((q << shift)+r) & MASK for q in (0,1,2,7,19,123,4567)
         for r in (-1,0,1)]))


def literal_screen(shift):
    samples = samples_for(shift)
    q = [full_shift(x, shift) for x in samples]
    matrix = [monomials(x, y, MASK) for x, y in zip(samples, q)]
    rhs = [target(x) for x in samples]
    # This is an actual SHR,MAD,MAD program used as an independent positive
    # control: t=x*q+17, output=t*t+x, with uint32 reduction at each MAD.
    t = [(x*y+17) & MASK for x,y in zip(samples,q)]
    control = [(y*y+x) & MASK for x,y in zip(samples,t)]
    return dict(shift=shift, samples=samples,
        inconsistent_full32=not consistent(matrix, rhs, 32),
        two_mad_positive_control=consistent(matrix, control, 32))


def variable_count_certificate():
    # Any first SHR whose count is x has q=0 on this domain, regardless of
    # whether its numerator is x or an arbitrary constant. Every remaining
    # integer polynomial has mixed difference divisible by 256*8=2048.
    samples = [32, 40, 288, 296]
    values = [target(x) & 2047 for x in samples]
    delta = (values[3]-values[2]-values[1]+values[0]) & 2047
    assert all(x >= 32 for x in samples) and delta != 0
    return dict(samples=samples, target_mod2048=values, mixed_difference=delta,
        polynomial_required_difference=0,
        scope='q=x>>x or K>>x is zero here, for every uint32 K')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    rows = [literal_screen(s) for s in range(33)]
    report = dict(sequence=['>>','mad','mad'], arithmetic_width=32,
        shift_semantics='full32 operands; counts >=32 produce zero',
        polynomial_degree=4, independent_coefficients=15,
        literal_counts=rows, variable_counts=variable_count_certificate(),
        constant_numerator_and_count='Constant q is covered by arbitrary '
            'polynomials in x, included in the q=0 (shift 32) relaxation.',
        all_representative_counts_excluded=all(r['inconsistent_full32'] for r in rows),
        positive_controls=sum(r['two_mad_positive_control'] for r in rows),
        scope='One input, arbitrary SSA operands and constants for SHR,MAD,MAD. '
              'Not all three-op shift programs or a whole-kernel lower bound.',
        new_executable=False)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in
        ('all_representative_counts_excluded','positive_controls','new_executable')}))


if __name__ == '__main__':
    main()
