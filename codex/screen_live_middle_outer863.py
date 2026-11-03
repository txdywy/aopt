"""Exhaust MAD,XOR,MAD when the last MAD is D*u+E and XOR uses a live word.

Use the same valid low-bit domain and first-MAD supersets as the structural
screen. For F(1)-F(0) odd, u(1)-u(0) must be odd. It uniquely determines D,
then E; all remaining inputs are checked. This does not cover a constant
XOR key or a final MAD that also consumes another live word.
"""
import argparse
import json
import time

import numpy as np

from codex.screen_quadratic_xor_affine863 import target
from codex.screen_live_middle_structure863 import (families, live, operand,
    parameter_count, parameters)


def inverse_odd(values, width):
    mask = (1 << width)-1
    inverse = np.ones_like(values)
    bits = 1
    while bits < width:
        inverse = (inverse*(2-values*inverse)) & mask
        bits *= 2
    return inverse


def search(width=11, fn=target, chunk_size=131072):
    if not 2 <= width <= 12 or chunk_size <= 0:
        raise ValueError('requires width 2..12 and positive chunk')
    mask = (1 << width)-1
    df = (fn(1)-fn(0)) & mask
    if df & 1 == 0:
        raise ValueError('unique outer coefficients require odd F(1)-F(0)')
    samples = list(dict.fromkeys([v & mask for v in list(range(32))+
        [63, 127, 255, 511, 512, 513, 1023, 1024, 1025, 2047]]+list(range(mask+1))))
    started = time.monotonic()
    rows = []
    for other in range(3):
        for name, indices in families():
            total = parameter_count(width, name)
            survivors = []
            for start in range(0, total, chunk_size):
                ids = np.arange(start, min(start+chunk_size, total), dtype=np.int64)
                d, e = parameters(ids, width, name)
                u0 = (operand(0,width,name,indices,d,e) & mask) ^ live(0,width)[other]
                u1 = (operand(1,width,name,indices,d,e) & mask) ^ live(1,width)[other]
                delta = (u1-u0) & mask
                keep = (delta & 1) == 1
                ids, d, e, u0, delta = [v[keep] for v in (ids, d, e, u0, delta)]
                scale = (df*inverse_odd(delta,width)) & mask
                offset = (fn(0)-scale*u0) & mask
                for x in samples:
                    if not len(ids):
                        break
                    u = (operand(x,width,name,indices,d,e) & mask) ^ live(x,width)[other]
                    keep = ((scale*u+offset) & mask) == (fn(x) & mask)
                    ids, d, e, scale, offset = [v[keep] for v in (ids,d,e,scale,offset)]
                survivors.extend([[int(v) for v in row] for row in zip(ids,scale,offset)])
            rows.append({'family': name, 'indices': indices, 'xor_live': other,
                         'tested': total, 'survivors_id_scale_offset': survivors})
    return {'width': width, 'domain_size': mask+1,
            'parameter_cases': sum(row['tested'] for row in rows), 'rows': rows,
            'survivors': sum(len(row['survivors_id_scale_offset']) for row in rows),
            'excluded': all(not row['survivors_id_scale_offset'] for row in rows),
            'seconds': round(time.monotonic()-started,3),
            'scope': 'first MAD arbitrary over b,a,prior and constants; '
                     'u=t XOR {b,a,prior}; output=D*u+E. No constant XOR key, '
                     'other final-MAD roles, shifts, or whole-kernel bound.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = search()
    with open(args.output, 'w') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
