"""Remove the first-stage C1 XOR, allowing arbitrary middle output encoding.

Search fixed four-op graphs for F(x XOR C1) XOR O, with O free. This differs
from fixed-output absorption and from three-op searches. Only a full32
proof could justify a new kernel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import z3

from codex.search_cross_encoding863 import middle,program,necessary_inputs


C1=0xc761c23c
FAMILIES=('parallel','triple_xor',*(f'{s}_{t}'for s in('input','key')for t in('input','first','middle')))


def evaluate(x,coefficients,family,width):
    mask=(1 << width)-1
    if family=='parallel':return program(x,coefficients,width)
    a,b,d,e=coefficients[:4]
    t=(a*x+b)&mask
    if family=='triple_xor':return (t^((d*x+e)&mask)^x)&mask
    source,tail=family.split('_')
    q=t^(x if source=='input'else coefficients[4])
    u=(d*q+e)&mask
    return (u^(x if tail=='input'else t if tail=='first'else q))&mask


def symbolic(x,coefficients,family):
    a,b,d,e=coefficients[:4]
    t=a*x+b
    if family=='parallel':return coefficients[4]*(t^(d*x+e))+coefficients[5]
    if family=='triple_xor':return t^(d*x+e)^x
    source,tail=family.split('_')
    q=t^(x if source=='input'else coefficients[4])
    return (d*q+e)^(x if tail=='input'else t if tail=='first'else q)


def control_coefficients(family):
    if family=='parallel':return [33,0xe9f8cc1d,16896,0xaccf6200,9,0xfd7046c5]
    if family=='triple_xor':return [21,19,5,29]
    return [6 if family=='input_middle'else 21,19,6,23,42]


def normalize_output(coefficients,output_key,family,width):
    """Preserve P(x) XOR O while clearing O's lowest and highest bits."""
    mask,high=(1 << width)-1,1 << (width-1)
    result=list(coefficients)
    mul,offset=(4,5)if family=='parallel'else(2,3)
    if output_key&1:
        result[mul]=-result[mul]
        result[offset]=-result[offset]-1
        output_key ^= mask
    if output_key&high:
        result[offset] += high
        output_key ^= high
    return [v&mask for v in result],output_key&mask


def search(width=11,seconds=25,input_key=C1,control=False,family='parallel'):
    if width not in (*range(2,17),32)or seconds<=0:
        raise ValueError('width 2..16 or 32 and positive seconds required')
    if family not in FAMILIES:raise ValueError('unsupported fixed four-op family')
    mask,high=(1 << width)-1,1 << (width-1)
    a,b,d,e,g,key,ok=z3.BitVecs('hm_a hm_b hm_d hm_e hm_g hm_key hm_ok',width)
    solver=z3.SolverFor('QF_BV')
    # Target is a permutation. Swapping/complementing both XOR arms gives
    # A odd, D even, B even. Output complement/top-bit flip can be absorbed
    # into G,H, permitting O's lowest and highest bits to be zero.
    solver.add(ok&1==0,ok&high==0)
    normalizations=['O lowest/highest bits zero']
    def reference(x):
        if not control:return middle(x)
        cv=control_coefficients(family)
        return evaluate(x^input_key,cv,family,width)if isinstance(x,int)else symbolic(x^input_key,cv,family)
    target=lambda x:reference(x^input_key)^ok
    zero=z3.BitVecVal(0,width)
    if family=='parallel':
        solver.add(a&1==1,d&1==0,g&1==1,b&1==0)
        normalizations+=['A odd','D even','G odd','B even']
        h=target(zero)-g*(b^e)
        coeffvars=[a,b,d,e,g,h]
    elif family=='triple_xor':
        coeffvars=[a,b,d,b^target(zero)]
    else:
        source,tail=family.split('_')
        q0=b if source=='input'else b^key
        tail0=zero if tail=='input'else b if tail=='first'else q0
        offset=(target(zero)^tail0)-d*q0
        coeffvars=[a,b,d,offset,key]
        if source=='key'and tail!='input':
            # This whole program reads x only through t=A*x+B.
            solver.add(a&1==1)
            normalizations+=['A odd by necessary first-map bijectivity']
    expr=lambda x:symbolic(x,coeffvars,family)
    points=necessary_inputs(width)

    def constrain(x):solver.add(expr(z3.BitVecVal(x,width))==target(z3.BitVecVal(x,width)))

    for x in points:constrain(x)
    checks=range(1 << width)if width<32 else points
    started=time.monotonic()
    result=dict(width=width,input_key=input_key,free_output_key=True,control=control,
        family=family,initial_samples=points[:],normalizations=normalizations,
        seconds_limit=seconds,events=[],result='deadline',new_executable=False)
    while time.monotonic()-started<seconds:
        solver.set(timeout=max(1,int(1000*(seconds-(time.monotonic()-started)))))
        status=solver.check()
        event=dict(samples=len(points),synthesis=str(status));result['events'].append(event)
        if status!=z3.sat:
            result['result']='projected_family_unsat'if status==z3.unsat else'unknown'
            if status==z3.unknown:result['reason']=solver.reason_unknown()
            break
        model=solver.model()
        coeff=[model.eval(v,model_completion=True).as_long()for v in coeffvars]
        ov=model.eval(ok,model_completion=True).as_long()
        result.update(coefficients=coeff,output_key=ov)
        bad=[x for x in checks if evaluate(x,coeff,family,width)!=((reference(x^input_key)^ov)&mask)][:8]
        if bad:
            event['counterexamples']=bad
            for x in bad:
                assert x not in points
                points.append(x);constrain(x)
            continue
        if width<32:
            result.update(result='equivalent_on_reduced_domain',verified_inputs=1 << width);break
        x=z3.BitVec('head_merge_full32_check',32)
        verify=z3.SolverFor('QF_BV')
        verify.set(timeout=max(1,int(1000*(seconds-(time.monotonic()-started)))))
        verify.add(symbolic(x,coeff,family)!=(reference(x^input_key)^ov))
        status=verify.check();event['full32_verification']=str(status)
        if status==z3.unsat:
            result['result']='proved_equivalent_full32';break
        if status==z3.unknown:
            result.update(result='verification_unknown',reason=verify.reason_unknown());break
        x=verify.model().eval(x,model_completion=True).as_long()
        event['symbolic_counterexample']=x
        assert x not in points
        points.append(x);constrain(x)
    result['seconds']=round(time.monotonic()-started,3)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--width',type=int,default=11)
    ap.add_argument('--seconds',type=float,default=25)
    ap.add_argument('--input-key',type=lambda x:int(x,0),default=C1)
    ap.add_argument('--control',action='store_true')
    ap.add_argument('--family',choices=FAMILIES,default='parallel')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result=search(args.width,args.seconds,args.input_key,args.control,args.family)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result.get(k)for k in('width','result','seconds','output_key')}))


if __name__=='__main__':main()
