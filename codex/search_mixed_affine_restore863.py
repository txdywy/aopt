"""Exact low-domain normalization for mixed a/keyed affine restore arms.

On the legitimate a<2**w subdomain, w<=16, a>>19=0. Search
G*((A*a+B) XOR (C*(a XOR K)+D))+H = F(a XOR C1) XOR PRE_C5.
The constants of the projected family may be normalized independently of
whether they lift to a full32 program; UNSAT is a necessary-condition proof.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import z3

from codex.search_cross_encoding863 import C1, PRE_C5, middle, necessary_inputs


def evaluate(x, coefficients, width):
    a,b,c,d,g,h,k = coefficients
    mask = (1 << width)-1
    return (g*((((a*x+b)&mask) ^ ((c*(x^k)+d)&mask)))+h)&mask


def normalize(coefficients, width):
    mask,high = (1 << width)-1,1 << (width-1)
    a,b,c,d,g,h,k = [v&mask for v in coefficients]
    if not g&1 or not (a+c)&1:
        raise ValueError('The target permutation requires odd G and opposite A/C parity')
    if k&1:
        k ^= mask
        c,d = -c,d-c
    if k&high:
        k ^= high
        d -= c*high
    if b&1:
        a,b = -a,-b-1
        g,h = -g,h-g
    if d&1:
        c,d = -c,-d-1
        g,h = -g,h-g
    if b&high:
        b ^= high
        h -= g*high
    if d&high:
        d ^= high
        h -= g*high
    if a&high:
        a ^= high
        c ^= high
    return [v&mask for v in (a,b,c,d,g,h,k)]


def search(width=11, seconds=30, control=None):
    if not 2 <= width <= 16 or seconds <= 0:
        raise ValueError('Width2..16 and a positive time limit required')
    mask,high = (1 << width)-1,1 << (width-1)
    a,b,c,d,e,k = z3.BitVecs('mixed_A mixed_B mixed_C mixed_D mixed_inverse_G mixed_K',width)
    solver = z3.SolverFor('QF_BV')
    solver.add(e&1==1,(a+c)&1==1,a&high==0)
    for value in (b,d,k):
        solver.add(value&(high|1)==0)
    target = (lambda x:evaluate(x,control,width)) if control is not None else (
        lambda x:(middle(x^C1)^PRE_C5)&mask)
    offset = (b^(c*k+d))-e*target(0)
    points = necessary_inputs(width)
    def constrain(x):
        solver.add(((a*x+b)^(c*(x^k)+d)) == e*target(x)+offset)
    for x in points:
        constrain(x)
    started = time.monotonic()
    report = dict(width=width,output_key=PRE_C5,control=control,
                  family='G*((A*a+B) XOR (C*(a XOR K)+D))+H',
                  target='F(a XOR C1) XOR PRE_C5 on the legal a<2^w domain',
                  cuts=['G odd; A/C opposite parity','K low/high bits zero',
                        'B/D low/high bits zero','A high bit zero',
                        'replace final G/H by odd inverse E and offset fixed by a=0'],
                  initial_points=points[:],events=[],status='unknown',new_executable=False)
    while time.monotonic()-started < seconds:
        solver.set(timeout=max(1,int(1000*(seconds-time.monotonic()+started))))
        status = solver.check()
        event = dict(samples=len(points),status=str(status));report['events'].append(event)
        if status != z3.sat:
            report['status'] = str(status)
            if status == z3.unknown:
                report['reason'] = solver.reason_unknown()
            break
        model = solver.model()
        aa,bb,cc,dd,ee,kk,ff = [model.eval(v,model_completion=True).as_long() for v in (a,b,c,d,e,k,offset)]
        gg = pow(ee,-1,1 << width)
        coefficients = [aa,bb,cc,dd,gg,(-gg*ff)&mask,kk]
        bad = [x for x in range(1 << width) if evaluate(x,coefficients,width) != target(x)][:8]
        if not bad:
            report.update(status='reduced_sat',coefficients=coefficients,verified_inputs=1 << width)
            break
        event['counterexamples'] = bad
        for x in bad:
            assert x not in points
            points.append(x);constrain(x)
    report.update(seconds=round(time.monotonic()-started,4),samples=len(points))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--width',type=int,default=11)
    ap.add_argument('--seconds',type=float,default=30)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    report = search(args.width,args.seconds)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],report['seconds'],flush=True)


if __name__ == '__main__':
    main()
