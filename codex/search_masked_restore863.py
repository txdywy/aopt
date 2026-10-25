"""Four-op masked affine families with an arbitrary input XOR encoding.

The extra AND is a genuinely different graph from three MADs and one XOR.
Only projected UNSAT or exhaustive low-domain SAT is reported here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import z3

from codex.search_cross_encoding863 import PRE_C5, middle, necessary_inputs


def evaluate(x, coefficients, family, mask):
    a,b,d,e,k = coefficients
    if family == 'parallel':
        return (((a*x+b)&mask)^((d*x+e)&k))&mask
    return (d*(x ^ ((a*x+b)&k))+e)&mask


def search(family='parallel', width=11, seconds=30, output_key=PRE_C5, control=False):
    if family not in ('parallel','nested') or not 2 <= width <= 16 or seconds <= 0:
        raise ValueError('parallel/nested, width 2..16 and positive seconds required')
    mask = (1 << width)-1
    a,b,d,e,k,ik = z3.BitVecs('mr_a mr_b mr_d mr_e mr_k mr_ik',width)
    solver = z3.SolverFor('QF_BV')
    # No endpoint-key normalization is needed for these queries; every key
    # bit remains free. The only cuts below are necessary output parity.
    control_cv = ([297,109,66,22,mask ^ 3] if family == 'parallel' else
                  [66,22,9,109,mask ^ 3])
    def reference(x):
        if not control:
            return middle(x)
        aa,bb,dd,ee,kk = control_cv
        t = x ^ 42
        value = ((aa*t+bb)^((dd*t+ee)&kk) if family == 'parallel'
                 else dd*(t ^ ((aa*t+bb)&kk))+ee)
        return value ^ output_key
    target = lambda x: reference(x ^ ik)^output_key
    if family == 'parallel':
        solver.add((a ^ (d & k)) & 1 == 1)
        offset = target(z3.BitVecVal(0,width))^(e & k)
        cv = [a,offset,d,e,k]
        expression = lambda x: (a*x+offset)^((d*x+e)&k)
    else:
        solver.add(d & 1 == 1, (a & k) & 1 == 0)
        offset = target(z3.BitVecVal(0,width))-d*(b & k)
        cv = [a,b,d,offset,k]
        expression = lambda x: d*(x ^ ((a*x+b)&k))+offset
    points = necessary_inputs(width)
    for x in points:
        solver.add(expression(x) == target(z3.BitVecVal(x,width)))
    started = time.monotonic()
    report = dict(family=family,width=width,output_key=output_key,
        arbitrary_input_key=True,control=control,seconds_limit=seconds,
        instructions=(['MAD','MAD','AND','XOR'] if family == 'parallel' else
                      ['MAD','AND','XOR','MAD']),
        initial_samples=points[:],events=[],result='deadline',new_executable=False)
    while time.monotonic()-started < seconds:
        solver.set(timeout=max(1,int(1000*(seconds-(time.monotonic()-started)))))
        status = solver.check()
        event = dict(samples=len(points),synthesis=str(status))
        report['events'].append(event)
        if status != z3.sat:
            report['result'] = 'projected_family_unsat' if status == z3.unsat else 'unknown'
            if status == z3.unknown:
                report['reason'] = solver.reason_unknown()
            break
        model = solver.model()
        coeff = [model.eval(v,model_completion=True).as_long() for v in cv]
        iv = model.eval(ik,model_completion=True).as_long()
        report.update(coefficients=coeff,input_key=iv)
        bad = [x for x in range(1 << width)
               if evaluate(x,coeff,family,mask) != ((reference(x ^ iv)^output_key)&mask)][:8]
        if not bad:
            report.update(result='equivalent_on_reduced_domain',verified_inputs=1 << width)
            break
        event['counterexamples'] = bad
        for x in bad:
            assert x not in points
            points.append(x)
            solver.add(expression(x) == target(z3.BitVecVal(x,width)))
    report['seconds'] = round(time.monotonic()-started,3)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--family',choices=('parallel','nested'),required=True)
    ap.add_argument('--width',type=int,default=11)
    ap.add_argument('--seconds',type=float,default=30)
    ap.add_argument('--output-key',type=lambda x:int(x,0),default=PRE_C5)
    ap.add_argument('--control',action='store_true')
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    report = search(args.family,args.width,args.seconds,args.output_key,args.control)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report.get(k) for k in ('family','width','result','seconds','input_key')}))


if __name__ == '__main__':
    main()
