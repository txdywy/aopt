"""Necessary polynomial rectangles for every pair of XOR endpoint keys.

For any integer polynomial P, its mixed difference at x,x+h,x+k,x+h+k
is divisible by h*k. We check all low-11-bit I,O for F(x XOR I) XOR O.
This certificate concerns polynomial programs, not programs containing XOR.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


def search(width=11, *, affine_control=False):
    if not 2 <= width <= 11:
        raise ValueError('complete paired-key screen supports widths 2..11')
    modulus, mask = 1 << width, (1 << width)-1
    keys = np.arange(modulus,dtype=np.int32)
    ik, ok = np.repeat(keys,modulus), np.tile(keys,modulus)

    def fn(x):
        if affine_control:
            return (17*x+23)&mask
        return (9*(((33*x+(0xe9f8cc1d&mask))&mask)
                   ^((16896*x+(0xaccf6200&mask))&mask))+(0xfd7046c5&mask))&mask

    lead = [(0,8,256),(1,8,256),(2,8,256),(4,8,256),(0,16,128),(0,32,64)]
    rectangles = ([r for r in lead if max(r[1:]) < modulus]
        + [(b,1 << i,1 << j) for b in range(min(32,modulus))
           for i in range(1,width) for j in range(i,width) if i+j >= width])
    history = []
    started = time.monotonic()
    for base,h,k in rectangles:
        values = [fn(((base+d)&mask)^ik)^ok for d in (0,h,k,h+k)]
        required = min((h*k)&-(h*k),modulus)
        delta = values[3]-values[1]-values[2]+values[0]
        good = (delta&(required-1)) == 0
        previous = len(ik)
        ik,ok = ik[good],ok[good]
        history.append(dict(base=base,h=h,k=k,divisor=required,
                            before=previous,after=len(ik)))
        if not len(ik):
            break
    return dict(width=width, initial_key_pairs=modulus**2, remaining_pairs=len(ik),
        rectangles=history, affine_control=affine_control,
        result='all_key_pairs_excluded' if not len(ik) else 'necessary_conditions_survive',
        survivor_examples=list(zip(ik[:16].tolist(),ok[:16].tolist())),
        zero_key_control_retained=bool(np.any((ik == 0)&(ok == 0))) if affine_control else None,
        seconds=round(time.monotonic()-started,3), new_executable=False,
        scope='Integer-polynomial programs with arbitrary degree and arbitrary '
              'input/output XOR keys; no bitwise instructions inside the program')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    report = dict(experiment=search(),control=search(6,affine_control=True),new_executable=False)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report['experiment'][k] for k in ('result','initial_key_pairs','remaining_pairs','seconds')}))


if __name__ == '__main__':
    main()
