"""Complete the sole-result outer-MAD case with a constant XOR key.

Target F is a permutation. If the output is D*u+E with u=t XOR K, both D
and the first MAD t must be invertible. The four affine/mixed first-MAD
supersets suffice; product+constant has a top-bit collision and product+live
has constant parity. The affine-b case already has an exact enumerative
certificate; the other three are fixed-structure SMT, without selectors.
"""
import argparse
import json
import random
import time

import z3

from codex.screen_affine_xor_affine863 import search as affine_search
from codex.screen_live_middle_structure863 import operand
from codex.screen_quadratic_xor_affine863 import target


def check(name,width=11,timeout_ms=15000,fn=target):
    if name not in ('affine_b','affine_a','mixed_ba','mixed_ab') or not 2 <= width <= 12:
        raise ValueError('requires one of the four affine/mixed families and width 2..12')
    if timeout_ms <= 0:
        raise ValueError('timeout must be positive')
    mask=(1<<width)-1
    outputs=[fn(x)&mask for x in range(mask+1)]
    if len(set(outputs)) != mask+1:
        raise ValueError('injectivity cuts require a bijective low-bit target')
    a,b,key,d,e=z3.BitVecs('a b key d e',width)
    solver=z3.SolverFor('QF_BV')
    solver.set(timeout=timeout_ms)
    solver.add(a&1 == (0 if name.startswith('mixed') else 1), d&1 == 1)
    rng=random.Random(863)
    samples=list(dict.fromkeys(list(range(min(32,mask+1)))+
                              [rng.randrange(mask+1) for _ in range(32)]))
    def add_sample(x):
        solver.add(d*(operand(x,width,name,(),a,b)^key)+e == outputs[x])
    for x in samples:
        add_sample(x)
    started=time.monotonic()
    result={'family':name,'width':width,'timeout_ms':timeout_ms,
            'initial_samples':len(samples),'target_is_permutation':True}
    # CEGIS only strengthens the necessary sample constraints. Verification
    # enumerates the entire reduced domain, never claims uint32 equivalence.
    for iteration in range(8):
        status=solver.check()
        result.update(result=str(status),iterations=iteration+1)
        if status == z3.unknown:
            result['reason']=solver.reason_unknown()
            break
        if status == z3.unsat:
            break
        model=solver.model()
        av,bv,kv,dv,ev=[model.eval(v,model_completion=True).as_long() for v in (a,b,key,d,e)]
        bad=next((x for x in range(mask+1) if
                  (dv*((operand(x,width,name,(),av,bv)&mask)^kv)+ev)&mask != outputs[x]),None)
        result['constants']=[av,bv,kv,dv,ev]
        if bad is None:
            result['verification']='equivalent_on_reduced_domain'
            break
        result['verification']='counterexample'
        result['counterexample']=bad
        add_sample(bad)
    result['seconds']=round(time.monotonic()-started,3)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    # Fast recheck of the already established affine-b certificate; do not
    # extend the identical SMT query that had timed out for this family.
    original=affine_search()
    rows=[check(name) for name in ('affine_a','mixed_ba','mixed_ab')]
    result={'width':11,'affine_b_enumeration':original,'smt_rows':rows,
        'excluded':not original['survivors'] and all(r['result']=='unsat' for r in rows),
        'noninjective_first_families':{
            'product_key':'t(b+2**10)=t(b): every live input toggles its top bit and '
                          'both multiplicands have the same parity',
            'product_live':'t is always even because every live input has parity b'},
        'scope':'t=one MAD from b,a,prior,constants; u=t XOR K; output=D*u+E. '
                'Valid restricted full32 domain; no other final-MAD live operands.'}
    with open(args.output,'w') as output:
        json.dump(result,output,indent=2)
        output.write('\n')
    print(json.dumps({'excluded':result['excluded'],'smt_rows':rows,
                     'affine_b_survivors':len(original['survivors'])}))


if __name__ == '__main__':
    main()
