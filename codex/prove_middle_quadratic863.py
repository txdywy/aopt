"""Exclude an affine first instruction followed by MAD via low-bit witnesses.

Any final MAD with operands selected from x, D*x+E and constants is a
polynomial of degree at most two. A mismatch modulo 2**width excludes an
identity modulo 2**32 for this shift-free target. This covers all constants
and operand choices in that template, not all two-instruction programs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def target(x, mask):
    k = 0xaccf6200 - 512*0xe9f8cc1d
    return (9*(x ^ ((512*x+k) & mask)) + 0xfd7046c5) & mask


def certificate(width=11):
    if not 2 <= width <= 16:
        raise ValueError('width must be between 2 and 16')
    modulus = 1 << width
    mask = modulus-1
    c = target(0,mask)
    delta = (target(1,mask)-c) & mask
    rhs = (target(2,mask)-c-2*delta) & mask
    # q(0), q(1), q(2) force C=c, B=delta-A and 2*A=rhs.
    candidates = [a for a in range(modulus) if (2*a) & mask == rhs]
    rejected, survivors = [], []
    for a in candidates:
        b = (delta-a) & mask
        witness = next((x for x in range(modulus)
                        if (a*x*x+b*x+c) & mask != target(x,mask)),None)
        if witness is None:
            survivors.append(dict(a=a,b=b,c=c))
        else:
            rejected.append(dict(a=a,b=b,c=c,input=witness,
                candidate=(a*witness*witness+b*witness+c) & mask,
                expected=target(witness,mask)))
    return dict(objective='middle-tail',first='affine',op='mad',width=width,
        status='polynomial_template_excluded' if not survivors else 'inconclusive',
        coefficient_equation=dict(c=c,delta=delta,rhs=rhs),
        rejected=rejected,survivors=survivors,
        scope='All degree-at-most-two polynomials modulo 2**width; hence affine then MAD over uint32')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = certificate()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
