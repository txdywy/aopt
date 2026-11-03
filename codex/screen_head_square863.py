"""A four-input obstruction to square-first absorption of the C1 XOR.

On x=0,32,64,96 modulo 2048, x and x*x+B have disjoint horizontal
and vertical varying bit sets {5,10} and {6}, for every B. The same is
true of x and x*x+x. Any bitwise prefix preserves this property; any
subsequent integer polynomial has mixed difference divisible by 32*64.
F(x XOR C1) XOR O has mixed difference 1024 for every O, a contradiction.
"""
import argparse
import json
from pathlib import Path

from codex.search_cross_encoding863 import C1,middle


WIDTH=11
MASK=(1 << WIDTH)-1
POINTS=(0,32,64,96)
H_BITS=(1 << 5)|(1 << 10)
V_BITS=1 << 6


def difference(values):
    a,b,c,d=values
    return (d-b-c+a)&MASK


def orthogonal(values):
    a,b,c,d=values
    return (a^b)==(c^d)and(a^c)==(b^d)and not((a^b)&~H_BITS)and not((a^c)&~V_BITS)


def certificate():
    assert orthogonal(POINTS)
    for offset in range(1 << WIDTH):
        assert orthogonal([(x*x+offset)&MASK for x in POINTS])
    square_x=[(x*x+x)&MASK for x in POINTS]
    assert orthogonal(square_x)
    values=[middle(x^C1)&MASK for x in POINTS]
    differences={difference([v^key for v in values])for key in range(1 << WIDTH)}
    assert differences=={1024}
    return dict(width=WIDTH,points=list(POINTS),target_values=values,input_key=C1,
        checked_square_offsets=1 << WIDTH,checked_output_keys=1 << WIDTH,
        horizontal_bit_mask=H_BITS,vertical_bit_mask=V_BITS,
        horizontal_divisor=32,vertical_divisor=64,polynomial_mixed_difference=0,
        target_mixed_differences=sorted(differences),square_input_values=square_x,
        result='square_first_bitwise_prefix_polynomial_tail_excluded',new_executable=False,
        proof='Each bit of every prefix word depends on at most one rectangle axis. '
              'Its horizontal and vertical numeric differences are multiples of 32 and 64. '
              'Every mixed term of an integer polynomial is divisible by 2048.',
        scope='Input x and first x*x+B or x*x+x, any constants, then arbitrary '
              'XOR/AND/OR prefix, then arbitrary integer-polynomial continuation; '
              'target F(x XOR C1) XOR O for arbitrary O.',
        exclusions='Interleaved later bitwise operations, affine first, shifts or other '
              'opcodes, additional live inputs, whole hash and kernel are not covered.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    report=certificate();args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
