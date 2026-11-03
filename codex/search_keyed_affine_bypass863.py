"""Refine two live-input cases with a freely relaxed affine bypass.

P(x)=((A*(x XOR I)+B) XOR K)+D*x+E must equal F(x XOR C1) XOR O.
The free E contains both actual tails D*a and D*prior on the low-a domain.
UNSAT excludes those cases; SAT still needs the actual tail and full32 checks.
"""
import argparse
import json
from pathlib import Path
import time
import z3

from codex.search_cross_encoding863 import C1,middle,necessary_inputs
from codex.search_keyed_live_middle863 import C0


def evaluate(x,coefficients,width):
    a,b,d,e,ik,key=coefficients
    mask=(1 << width)-1
    return ((((a*(x^ik)+b)&mask)^key)+d*x+e)&mask


def normalize(coefficients,outkey,width):
    mask,high=(1 << width)-1,1 << (width-1)
    a,b,d,e,ik,key=coefficients
    if outkey&1:
        a,b,d,e=-a,-b-1,-d,-e
        outkey ^= mask
    if outkey&high:
        key ^= high;outkey ^= high
    if ik&1:
        a,b,ik=-a,b-a,ik^mask
    if ik&high:
        b += a*high;ik ^= high
    if key&1:
        a,b,key=-a,-b-1,key^mask
    if key&high:
        b += high;key ^= high
    return [v&mask for v in (a,b,d,e,ik,key)],outkey&mask


def search(width=11,seconds=30,parity=None,control=False,bypass_source=None):
    if not 2<=width<=16 or seconds<=0 or parity not in(None,0,1):
        raise ValueError('width 2..16, positive time, parity 0/1 or unrestricted required')
    if bypass_source not in(None,'a','prior'):
        raise ValueError('a, prior, or a free affine offset required')
    mask,high=(1 << width)-1,1 << (width-1)
    solver=z3.SolverFor('QF_BV')
    a,b,d,ik,key,outkey=z3.BitVecs('kb_a kb_b kb_d kb_i kb_k kb_o',width)
    for v in (ik,key,outkey):solver.add(v&(high|1)==0)
    solver.add((a+d)&1==1)
    if parity is not None:solver.add(a&1==parity)
    control_coeff=[21,19,6,23,42,13]
    if parity==0:control_coeff[:3]=[6,19,21]
    prior_offset=(-C0*pow(4097,-1,1 << width))&mask
    # prior = inverse4097*a + prior_offset. At the proof width (11),
    # the inverse is one; require this for the exact scalar-multiple tail.
    if bypass_source=='prior'and pow(4097,-1,1 << width)!=1:
        raise ValueError('the prior-tail coefficient is used only at width <= 12')
    if bypass_source is not None:
        control_coeff[3]=0 if bypass_source=='a'else control_coeff[2]*prior_offset
    target=lambda x:evaluate(x,control_coeff,width)if control else middle(x^C1)&mask
    e=((z3.BitVecVal(target(0),width)^outkey)-((a*ik+b)^key)
       if bypass_source is None else z3.BitVecVal(0,width)if bypass_source=='a'else d*prior_offset)
    def expr(x):return ((a*(x^ik)+b)^key)+d*x+e
    points=necessary_inputs(width)
    def constrain(x):solver.add(expr(z3.BitVecVal(x,width))==(target(x)^outkey))
    for x in points:constrain(x)
    started=time.monotonic()
    report=dict(width=width,parity=parity,control=control,bypass_source=bypass_source,seconds_limit=seconds,
        normalizations=['A+D odd','I,K,O lowest and highest bits zero',
                        'infer E from x=0'if bypass_source is None else'exact offset from the bypass source'],
        family='((A*(x XOR I)+B) XOR K)+D*x+E',events=[],new_executable=False,result='deadline')
    while time.monotonic()-started<seconds:
        solver.set(timeout=max(1,int(1000*(seconds-time.monotonic()+started))))
        status=solver.check();report['events'].append(dict(samples=len(points),synthesis=str(status)))
        if status!=z3.sat:
            report['result']='relaxed_family_unsat'if status==z3.unsat else'unknown'
            if status==z3.unknown:report['reason']=solver.reason_unknown()
            break
        model=solver.model()
        cv=[model.eval(v,model_completion=True).as_long()for v in(a,b,d,e,ik,key)]
        ov=model.eval(outkey,model_completion=True).as_long()
        bad=[x for x in range(1 << width)if evaluate(x,cv,width)!=(target(x)^ov)][:8]
        if not bad:
            report.update(result='equivalent_on_relaxed_low_domain',coefficients=cv,
                output_key=ov,verified_inputs=1 << width);break
        report['events'][-1]['counterexamples']=bad
        for x in bad:
            assert x not in points
            points.append(x);constrain(x)
    report['seconds']=round(time.monotonic()-started,3)
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--width',type=int,default=11)
    ap.add_argument('--seconds',type=float,default=30)
    ap.add_argument('--parity',type=int,choices=(0,1))
    ap.add_argument('--control',action='store_true')
    ap.add_argument('--bypass-source',choices=('a','prior'))
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    report=search(args.width,args.seconds,args.parity,args.control,args.bypass_source)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k]for k in('result','parity','seconds')}))


if __name__=='__main__':main()
