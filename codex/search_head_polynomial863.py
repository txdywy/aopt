"""Replace nonlinear selector products by necessary polynomial families.

The polynomial coefficients are relaxed independently, so UNSAT excludes
covered DAGs, while SAT is only a relaxed low-word candidate. Falling-factorial
bases remove coefficient aliases without losing integer polynomials.
"""
from __future__ import annotations

import argparse
import json
from math import comb,factorial
from pathlib import Path
import time

import z3

from codex.search_cross_encoding863 import C1,middle,necessary_inputs

FAMILIES=('polynomial_xor','scaled_quadratic_xor','quadratic_xor_bypass')


def poly(x,coefficients):
    value=coefficients[0];term=1
    for degree,c in enumerate(coefficients[1:],1):
        term=term*(x-degree+1)
        value=value+c*term
    return value


def canonical_poly(coefficients,width):
    out=[]
    for degree,c in enumerate(coefficients):
        f=factorial(degree);valuation=(f&-f).bit_length()-1
        bits=max(0,width-valuation)
        out.append(c&((1 << bits)-1))
    return out


def interpolate_falling(values,width):
    """Infer canonical integer-polynomial coefficients from consecutive values.

    Odd parts of j! are invertible, while powers of two impose necessary
    divisibility checks. Never divide by a nonunit as if this ring were a field.
    """
    mask=(1 << width)-1
    coefficients=[];valid=[]
    symbolic=any(z3.is_bv(v)for v in values)
    for j in range(len(values)):
        delta=sum((-1)**(j-i)*comb(j,i)*values[i]for i in range(j+1))&mask
        f=factorial(j);valuation=(f&-f).bit_length()-1
        bits=width-valuation
        if bits<=0:raise ValueError('degree too high for this interpolation width')
        valid.append(delta&((1 << valuation)-1)==0)
        quotient=z3.LShR(delta,valuation)if symbolic else delta >> valuation
        coefficients.append((quotient*pow(f >> valuation,-1,1 << bits))&((1 << bits)-1))
    return coefficients,z3.And(*valid)if symbolic else all(valid)


def evaluate(x,coefficients,family,width):
    mask=(1 << width)-1
    p,q=coefficients['p'],coefficients['q']
    v=(poly(x,p)&mask)^(poly(x,q)&mask)
    if family=='scaled_quadratic_xor':v=coefficients['scale']*v+coefficients['offset']
    elif family=='quadratic_xor_bypass':v=v+poly(x,coefficients['r'])
    return v&mask


def normalize(coefficients,outkey,family,width,fixed_output=False):
    """Keep evaluate(x)^outkey unchanged under all search normalizations."""
    mask,high=(1 << width)-1,1 << (width-1)
    c={k:list(v)if isinstance(v,(list,tuple))else v for k,v in coefficients.items()}
    if family=='polynomial_xor':
        if fixed_output:
            if c['p'][0]&1:
                for name in('p','q'):
                    c[name]=[-v for v in c[name]];c[name][0]-=1
            if c['p'][0]&high:
                c['p'][0]+=high;c['q'][0]+=high
        for name in(()if fixed_output else('p','q')):
            if c[name][0]&1:
                c[name]=[-v for v in c[name]];c[name][0]-=1;outkey^=mask
            if c[name][0]&high:c[name][0]+=high;outkey^=high
    else:
        if not fixed_output and outkey&1:
            if family=='scaled_quadratic_xor':
                c['scale']=-c['scale'];c['offset']=-c['offset']-1
            else:
                c['p']=[-v for v in c['p']];c['p'][0]-=1
                c['r']=[-v for v in c['r']]
            outkey^=mask
        if not fixed_output and outkey&high:
            if family=='scaled_quadratic_xor':c['offset']+=high
            else:c['r'][0]+=high
            outkey^=high
        if c['p'][0]&1:
            for name in('p','q'):
                c[name]=[-v for v in c[name]];c[name][0]-=1
        if c['p'][0]&high:
            c['p'][0]+=high
            if family=='scaled_quadratic_xor':c['offset']-=c['scale']*high
            else:c['r'][0]-=high
    c={k:canonical_poly(v,width)if isinstance(v,list)else v&mask for k,v in c.items()}
    return c,outkey&mask


def control_coefficients(family):
    # Each target has an explicit four-op MAD/XOR realization (see tests).
    if family=='polynomial_xor':
        a,b,c,d,e=6,17,21,9,29
        return dict(p=[d*(b*b+c)+e,d*(a*a+2*a*b),d*a*a,0,0],q=[b,a,0])
    if family=='scaled_quadratic_xor':
        return dict(p=[17,6],q=[21,23,6],scale=9,offset=29)
    return dict(p=[17,6],q=[21,8,0],r=[0,23,6,0,0])


def search(family,width=11,seconds=15,control=False,interpolate=False,pin_control=False,
           input_key=C1,fixed_output_key=None):
    if family not in FAMILIES or not 5<=width<=16 or seconds<=0:
        raise ValueError('supported family, width 5..16, positive time limit required')
    if pin_control and not control:raise ValueError('pinning requires a positive control')
    if interpolate and family!='quadratic_xor_bypass':raise ValueError('interpolate the bypass family only')
    if not 0<=input_key<=0xffffffff or fixed_output_key is not None and not 0<=fixed_output_key<=0xffffffff:
        raise ValueError('endpoint keys must be uint32')
    mask,high=(1 << width)-1,1 << (width-1)
    solver=z3.SolverFor('QF_BV')
    def polynomial(name,degree):
        cs=[z3.BitVec(f'{name}_{j}',width)for j in range(degree+1)]
        for j,c in enumerate(cs):
            f=factorial(j);valuation=(f&-f).bit_length()-1
            if valuation:solver.add(z3.ULT(c,1 << (width-valuation)))
        return cs
    p=polynomial('hp_p',4 if family=='polynomial_xor'else 1)
    q=polynomial('hp_q',2)
    witness=control_coefficients(family)
    target=lambda x:((evaluate(x,witness,family,width)^(fixed_output_key or 0))&mask
                     if control else middle(x^input_key)&mask)
    p0=p[0]
    solver.add(p0&(high|1)==0)
    if family=='polynomial_xor':
        if fixed_output_key is None:solver.add(q[0]&(high|1)==0)
        outkey=p0^q[0]^target(0)
        c=dict(p=p,q=q)
    else:
        outkey=z3.BitVec('hp_output_key',width)
        if fixed_output_key is None:solver.add(outkey&(high|1)==0)
        zero=(z3.BitVecVal(target(0),width)^outkey)
        if family=='scaled_quadratic_xor':
            scale=z3.BitVec('hp_scale',width)
            # The target flips parity between 0 and 1 and distinguishes 0
            # from the highest input bit. These two necessary observations
            # force G odd, p1+q1 odd and the falling-quadratic term even.
            solver.add(scale&1==1,(p[1]+q[1])&1==1,q[2]&1==0)
            c=dict(p=p,q=q,scale=scale,offset=zero-scale*(p0^q[0]))
        else:
            if interpolate:
                values=[(z3.BitVecVal(target(x),width)^outkey)-(poly(x,p)^poly(x,q))for x in range(5)]
                r,valid=interpolate_falling(values,width)
                solver.add(valid)
            else:
                r=polynomial('hp_r',4)
                r[0]=zero-(p0^q[0])
            c=dict(p=p,q=q,r=r)
    if fixed_output_key is not None:solver.add(outkey==(fixed_output_key&mask))
    if pin_control:
        normalized,key=normalize(witness,(fixed_output_key or 0)&mask,family,width,
                                 fixed_output=fixed_output_key is not None)
        solver.add(outkey==key)
        for name,variables in c.items():
            if isinstance(variables,list):
                solver.add(*(v==n for v,n in zip(variables,normalized[name])))
            else:solver.add(variables==normalized[name])
    points=list(dict.fromkeys(x&mask for x in necessary_inputs(width)+[32,64,96]))
    def constrain(x):solver.add(evaluate(x,c,family,width)==(target(x)^outkey))
    for x in points:constrain(x)
    started=time.monotonic()
    report=dict(family=family,width=width,control=control,input_key=input_key,
        fixed_output_key=fixed_output_key,
        interpolate=interpolate,pin_control=pin_control,
        seconds_limit=seconds,initial_samples=points[:],events=[],result='deadline',new_executable=False,
        scope='Independent integer-polynomial coefficients in falling-factorial basis; '
              'only a necessary relaxation of covered four-op DAGs. SAT is not a full32 or executable result.')
    while time.monotonic()-started<seconds:
        solver.set(timeout=max(1,int((seconds-time.monotonic()+started)*1000)))
        status=solver.check();event=dict(samples=len(points),synthesis=str(status));report['events'].append(event)
        if status!=z3.sat:
            report['result']='relaxed_family_unsat'if status==z3.unsat else'unknown'
            if status==z3.unknown:report['reason']=solver.reason_unknown()
            break
        model=solver.model()
        cv={k:[model.eval(t,model_completion=True).as_long()for t in v]if isinstance(v,list)
            else model.eval(v,model_completion=True).as_long()for k,v in c.items()}
        ov=model.eval(outkey,model_completion=True).as_long()
        report.update(coefficients=cv,output_key=ov)
        bad=[x for x in range(1 << width)if evaluate(x,cv,family,width)!=(target(x)^ov)][:8]
        if not bad:
            report.update(result='equivalent_on_relaxed_low_domain',verified_inputs=1 << width);break
        event['counterexamples']=bad
        for x in bad:
            assert x not in points
            points.append(x);constrain(x)
    report['seconds']=round(time.monotonic()-started,3)
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--family',choices=FAMILIES,required=True)
    ap.add_argument('--width',type=int,default=11)
    ap.add_argument('--seconds',type=float,default=15)
    ap.add_argument('--control',action='store_true')
    ap.add_argument('--interpolate',action='store_true')
    ap.add_argument('--pin-control',action='store_true')
    ap.add_argument('--input-key',type=lambda x:int(x,0),default=C1)
    ap.add_argument('--output-key',type=lambda x:int(x,0))
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    report=search(args.family,args.width,args.seconds,args.control,args.interpolate,args.pin_control,
                  args.input_key,args.output_key)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k]for k in('family','width','result','seconds')}))


if __name__=='__main__':main()
