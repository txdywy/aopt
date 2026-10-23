"""Exact reductions for the eight residual constant-mask live-input families.

See SUB863_LIVE_AND_SHIFT_20261023.md for the algebraic coverage argument.
No UNKNOWN is counted as proof. Three families use input 0/1 to eliminate
constants, two use forced mask bits, two use an invertible first-result split,
and D*(t XOR K)+b has a low-bit/period contradiction.
"""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import z3

from codex.screen_live_middle_structure863 import (families, live, operand,
    parameter_count, parameters, rectangles)
from codex.screen_quadratic_xor_affine863 import target


def enumerate_xor(kind,width=11,fn=target,chunk_size=131072):
    if kind not in ('b*t+u','b*u+t','D*b+u') or not 2 <= width <= 12 or chunk_size <= 0:
        raise ValueError('invalid family, width or chunk')
    mask=(1<<width)-1
    outputs=[fn(x)&mask for x in range(mask+1)]
    rows=[]
    started=time.monotonic()
    for name,indices in families():
        total=1<<(2*width) if name.startswith(('affine','mixed')) else 1<<width
        survivors=[]
        for start in range(0,total,chunk_size):
            ids=np.arange(start,min(start+chunk_size,total),dtype=np.int64)
            a,key=ids>>width,ids&mask
            at_zero=outputs[0] if kind=='b*u+t' else outputs[0]^key
            b=np.zeros_like(a)+((at_zero-operand(0,width,name,indices,a,0))&mask)
            if name=='product_live':
                b=np.zeros_like(a)  # This family has no adjustable additive constant.
            t0=operand(0,width,name,indices,a,b)&mask
            keep=(t0 if kind=='b*u+t' else t0^key)==outputs[0]
            a,b,key=[v[keep] for v in (a,b,key)]
            t1=operand(1,width,name,indices,a,b)&mask
            scale=(outputs[1]-(t1^key))&mask if kind=='D*b+u' else np.zeros_like(a)
            for x,y in enumerate(outputs):
                if not len(a):
                    break
                t=operand(x,width,name,indices,a,b)&mask
                u=t^key
                value=x*t+u if kind=='b*t+u' else x*u+t if kind=='b*u+t' else scale*x+u
                keep=(value&mask)==y
                a,b,key,scale=[v[keep] for v in (a,b,key,scale)]
            survivors.extend([[int(v) for v in row] for row in zip(a,b,key,scale)])
        rows.append({'family':name,'indices':indices,'tested':total,
                     'survivors_A_B_K_D':survivors})
    return {'kind':kind,'width':width,'parameter_cases':sum(r['tested'] for r in rows),
            'survivors':sum(len(r['survivors_A_B_K_D']) for r in rows),
            'excluded':all(not r['survivors_A_B_K_D'] for r in rows),'rows':rows,
            'seconds':round(time.monotonic()-started,3)}


def force_mask(bitwise,width,threshold,fn=target):
    """For F=D*b+(t OP K), determine mask bits forced neutral for every D."""
    if bitwise not in ('&','|') or not 2 <= width <= 12 or not 0 < threshold < width:
        raise ValueError('invalid mask forcing parameters')
    mask=(1<<width)-1
    ds=np.arange(mask+1,dtype=np.int64)
    forced=np.zeros_like(ds)
    needed=mask^((1<<threshold)-1)
    for x in range(mask+1):
        g=(fn(x)-ds*x)&mask
        forced |= g if bitwise=='&' else mask^g
        if np.all((forced&needed)==needed):
            break
    exceptions=ds[(forced&needed)!=needed].tolist()
    witnesses=[r for r in rectangles(width,fn) if r[2]-r[0]==1<<threshold and
               (fn(r[3])-fn(r[2])-fn(r[1])+fn(r[0]))&mask]
    witness=witnesses[0] if witnesses else None
    return {'bitwise':bitwise,'width':width,'threshold':threshold,
            'coefficients_tested':mask+1,'last_sample':x,'required_neutral_bits':needed,
            'exceptional_coefficients':exceptions,'rectangle':witness,
            'mixed_difference':((fn(witness[3])-fn(witness[2])-fn(witness[1])+fn(witness[0]))&mask)
                              if witness else None,
            'nonexceptional_coefficients_excluded':witness is not None}


def masked_first_mad(scale,width=11,fn=target,chunk_size=131072):
    """t AND K=G iff t AND OR_x(G(x))=G on the entire finite domain."""
    if not 2 <= width <= 12 or chunk_size <= 0:
        raise ValueError('invalid width or chunk')
    mask=(1<<width)-1
    outputs=[(fn(x)-scale*x)&mask for x in range(mask+1)]
    forced=0
    for y in outputs:
        forced |= y
    rows=[]
    for name,indices in families():
        total=parameter_count(width,name)
        survivors=[]
        for start in range(0,total,chunk_size):
            ids=np.arange(start,min(start+chunk_size,total),dtype=np.int64)
            a,b=parameters(ids,width,name)
            for x,y in enumerate(outputs):
                if not len(ids):
                    break
                keep=(operand(x,width,name,indices,a,b)&forced)==y
                ids,a,b=[v[keep] for v in (ids,a,b)]
            survivors.extend(ids.tolist())
        rows.append({'family':name,'indices':indices,'tested':total,
                     'survivor_parameter_ids':survivors})
    return {'width':width,'scale':scale,'forced_mask':forced,'rows':rows,
            'parameter_cases':sum(r['tested'] for r in rows),
            'excluded':all(not r['survivor_parameter_ids'] for r in rows)}


def scaled_xor_input_certificate():
    # G(b)=F(b)-b is 2 mod 8. D*u=G forces v2(D)=0 or 1 and
    # u, hence t=u XOR K, constant modulo 2**(3-v2(D)).
    g=lambda x:(target(x)-x)&2047
    assert {g(x)&7 for x in range(2048)}=={2}
    offsets=live(0,3)
    assert all(all((v-x-offset)&7==0 for v,offset in zip(live(x,3),offsets))
               for x in range(8))
    # Two live multiplicands give a monic quadratic modulo 4, whose second
    # difference is 2. It cannot be constant modulo 4. Thus t is linear in
    # its live operands and its total slope is divisible by 2**(3-v2(D)).
    xs=[8,264]
    words=[live(x,11) for x in xs]
    assert all((v-u)&2047==256 for u,v in zip(*words))
    difference=(g(xs[1])-g(xs[0]))&2047
    assert difference==1024
    return {'excluded':True,'width':11,'G_low_three_bits':2,
            'possible_v2_D':[0,1],'live_offsets_mod8':offsets,
            'monic_quadratic_second_difference_mod4':2,
            'inputs':xs,'live_values':words,'G_values':[g(x) for x in xs],
            'G_difference':difference,
            'scope':'F=D*(t XOR K)+b; t is any one MAD from b,a,prior,constants. '
                    'Linear t changes by a multiple of 2**(11-v2(D)); XOR preserves '
                    'that congruence, so D*u cannot change modulo 2048 at this pair.'}


def check_tonly(kind,name,width=11,timeout_ms=7000,fn=target):
    if kind not in ('D*t+u','D*u+t') or name not in ('affine_b','affine_a','mixed_ba','mixed_ab'):
        raise ValueError('invalid t-only family')
    if not 2 <= width <= 12 or timeout_ms <= 0:
        raise ValueError('invalid width or timeout')
    mask=(1<<width)-1
    outputs=[fn(x)&mask for x in range(mask+1)]
    if len(set(outputs)) != mask+1:
        raise ValueError('t-only injectivity cuts require a permutation')
    a,b,key,d=z3.BitVecs('a b key d',width)
    solver=z3.SolverFor('QF_BV')
    solver.set(timeout=timeout_ms)
    # Summing a permutation modulo 2**w gives 2**(w-1). t and t XOR K
    # are both permutations, so D must be even for either final expression.
    solver.add(a&1 == (0 if name.startswith('mixed') else 1), d&1 == 0)
    rng=random.Random(863)
    samples=list(dict.fromkeys(list(range(min(32,mask+1)))+[rng.randrange(mask+1) for _ in range(32)]))
    def add_sample(x):
        t=operand(x,width,name,(),a,b)
        u=t^key
        solver.add((d*t+u if kind=='D*t+u' else d*u+t)==outputs[x])
    for x in samples:
        add_sample(x)
    started=time.monotonic()
    result={'kind':kind,'family':name,'width':width,'timeout_ms':timeout_ms,
            'samples':len(samples),'first_injective_cut':True,'D_even':True}
    for iteration in range(8):
        status=solver.check()
        result.update(result=str(status),iterations=iteration+1)
        if status==z3.unknown:
            result['reason']=solver.reason_unknown()
            break
        if status==z3.unsat:
            break
        values=[solver.model().eval(v,model_completion=True).as_long() for v in (a,b,key,d)]
        av,bv,kv,dv=values
        bad=None
        for x,y in enumerate(outputs):
            t=operand(x,width,name,(),av,bv)&mask
            u=t^kv
            if ((dv*t+u if kind=='D*t+u' else dv*u+t)&mask)!=y:
                bad=x
                break
        result['constants_A_B_K_D']=values
        if bad is None:
            result['verification']='equivalent_on_reduced_domain'
            break
        add_sample(bad)
        result['verification']='counterexample'
        result['counterexample']=bad
    result['seconds']=round(time.monotonic()-started,3)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    enum=[enumerate_xor(kind) for kind in ('b*t+u','b*u+t','D*b+u')]
    and_force=force_mask('&',11,8)
    and_cases=[masked_first_mad(d) for d in and_force['exceptional_coefficients']]
    or_force=force_mask('|',12,9)
    tonly=[check_tonly(kind,name) for kind in ('D*t+u','D*u+t')
           for name in ('affine_a','mixed_ba','mixed_ab')]
    # These two exact affine-b roles were already independently excluded.
    prior_path=Path(__file__).with_name('middle_cross_stage863_results.json')
    prior=json.loads(prior_path.read_text())['experiments']['outer_roles_refined']['rows']
    prior_rows=[next(r for r in prior if r['role']==role and r['result']=='unsat')
                for role in ([1,3,2],[2,3,1])]
    scaled=scaled_xor_input_certificate()
    excluded=(all(r['excluded'] for r in enum) and
              and_force['nonexceptional_coefficients_excluded'] and all(r['excluded'] for r in and_cases) and
              or_force['nonexceptional_coefficients_excluded'] and not or_force['exceptional_coefficients'] and
              all(r['result']=='unsat' for r in tonly) and scaled['excluded'])
    report={'xor_enumerations':enum,'and_mask_forcing':and_force,'and_exceptions':and_cases,
            'or_mask_forcing':or_force,'t_only_smt':tonly,
            't_only_affine_b_prior':{'file':prior_path.name,'rows':prior_rows},
            'scaled_xor_plus_input':scaled,'all_eight_residual_classes_excluded':excluded}
    with open(args.output,'w') as output:
        json.dump(report,output,indent=2)
        output.write('\n')
    print(json.dumps({'excluded':excluded,'xor_cases':[r['parameter_cases'] for r in enum],
                      'and_exceptional_coefficients':and_force['exceptional_coefficients'],
                      'or_exceptional_coefficients':or_force['exceptional_coefficients'],
                      't_only_results':[r['result'] for r in tonly]}))


if __name__=='__main__':
    main()
