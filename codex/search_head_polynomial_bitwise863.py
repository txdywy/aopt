"""Relax a final AND/OR to arbitrary degree-four and degree-two polynomials.

No XOR-specific endpoint/coefficient normalization is applied. Low-width
UNSAT excludes the relaxed family; a surviving polynomial is not executable.
"""
import argparse
import json
from math import factorial
from pathlib import Path
import time

import z3

from codex.search_cross_encoding863 import C1,middle,necessary_inputs
from codex.search_head_polynomial863 import interpolate_falling,poly


def control_coefficients(width):
    small=lambda x:(6*x+17)**2+21
    large=lambda x:small(x)**2+29
    p,valid_p=interpolate_falling([large(x) for x in range(5)],width)
    q,valid_q=interpolate_falling([small(x) for x in range(3)],width)
    assert valid_p and valid_q
    return dict(p=p,q=q)


def evaluate(x,c,op,width):
    mask=(1<<width)-1
    p,q=poly(x,c['p'])&mask,poly(x,c['q'])&mask
    return (p&q) if op=='&' else (p|q)


def search(op='&',width=11,seconds=8,control=False,pin_control=False):
    if op not in ('&','|') or not 5<=width<=16 or seconds<=0:
        raise ValueError('AND/OR, width 5..16 and positive seconds required')
    if pin_control and not control:raise ValueError('pinning requires a control')
    mask=(1<<width)-1
    solver=z3.SolverFor('QF_BV')
    coeff={}
    for name,degree in (('p',4),('q',2)):
        coeff[name]=[z3.BitVec(f'{name}_bitwise_{i}',width) for i in range(degree+1)]
        for i,c in enumerate(coeff[name]):
            f=factorial(i)
            v=(f&-f).bit_length()-1
            if v:solver.add(z3.ULT(c,1<<(width-v)))
    witness=control_coefficients(width)
    target=lambda x:evaluate(x,witness,op,width) if control else middle(x^C1)&mask
    zero=z3.BitVecVal(0,width)
    output_key=evaluate(zero,coeff,op,width)^target(0)
    if pin_control:
        for name,values in coeff.items():
            solver.add(*(variable==value for variable,value in zip(values,witness[name])))
    samples=list(dict.fromkeys(x&mask for x in necessary_inputs(width)+[32,64,96]))
    def constrain(x):
        solver.add(evaluate(z3.BitVecVal(x,width),coeff,op,width)==(target(x)^output_key))
    for x in samples:constrain(x)
    started=time.monotonic()
    report=dict(width=width,operator=op,control=control,pin_control=pin_control,
        fixed_input_key=None if control else C1,free_output_key=True,
        degrees=[4,2],samples=samples[:],events=[],seconds_limit=seconds,
        result='deadline',new_executable=False,
        scope='Final AND/OR of independently relaxed integer polynomials P4 and Q2; '
              'arbitrary output XOR. No claim about an entire kernel or other bitwise positions.')
    while time.monotonic()-started<seconds:
        solver.set(timeout=max(1,int(1000*(seconds-time.monotonic()+started))))
        status=solver.check()
        event=dict(samples=len(samples),synthesis=str(status));report['events'].append(event)
        if status!=z3.sat:
            report['result']='relaxed_family_unsat' if status==z3.unsat else 'unknown'
            if status==z3.unknown:report['reason']=solver.reason_unknown()
            break
        model=solver.model()
        concrete={name:[model.eval(c,model_completion=True).as_long() for c in values]
                  for name,values in coeff.items()}
        key=model.eval(output_key,model_completion=True).as_long()
        report.update(coefficients=concrete,output_key=key)
        bad=[x for x in range(1<<width) if evaluate(x,concrete,op,width)!=(target(x)^key)][:8]
        if not bad:
            report.update(result='equivalent_on_relaxed_low_domain',verified_inputs=1<<width)
            break
        event['counterexamples']=bad
        for x in bad:
            assert x not in samples
            samples.append(x);constrain(x)
    report['seconds']=round(time.monotonic()-started,3)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator',choices=('&','|'),default='&')
    parser.add_argument('--width',type=int,default=11)
    parser.add_argument('--seconds',type=float,default=8)
    parser.add_argument('--control',action='store_true')
    parser.add_argument('--pin-control',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=search(args.operator,args.width,args.seconds,args.control,args.pin_control)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('operator','width','result','seconds')}))


if __name__=='__main__':
    main()
