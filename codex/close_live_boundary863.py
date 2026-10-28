"""Exact certificates for the seven final full live-boundary UNKNOWNs.

No timeout is reclassified as solver UNSAT. These are separate polynomial
rectangle, image cardinality, parity/dataflow and permutation-sum proofs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import z3

from codex.search_live_boundary863 import inputs,target,lsb_conditions,primary_path_avoiding,MASK


NODE=0xA76E1971
def key(r):return tuple(r['order']),tuple(tuple(w) for w in r['wiring'])
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rectangle():
    bs=[0,1<<30,1<<31,3<<30]
    vv=[inputs(b,NODE,'arms',False) for b in bs]
    zs=[v[1] for v in vv];ys=[target(v[0],NODE) for v in vv]
    mixed=lambda a:(a[3]-a[1]-a[2]+a[0])&MASK
    assert mixed(bs)==mixed(zs)==0
    assert all((a-a0)%65536==0 for data in (bs,zs) for a0 in data[:1] for a in data)
    delta=mixed(ys);distance=min(delta,(-delta)&MASK)
    assert distance>2*65535
    return dict(primary_b=bs,z=zs,target=ys,mixed_target=delta,
                modular_distance=distance,maximum_mixed_shift_residual=2*65535,
                polynomial_mixed_difference=0,
                proof='All coordinate increments are multiples of 2^16, so every mixed polynomial term vanishes mod 2^32')


def image():
    values=[(target(e,NODE)-e)&MASK for e in range(1<<17)]
    count=len(set(values));assert count>65536
    return dict(e_range=[0,1<<17],distinct_residuals=count,
                maximum_constant_times_shr16_image=65536,
                values_sha256=hashlib.sha256(b''.join(v.to_bytes(4,'little') for v in values)).hexdigest())


def top_facts():
    b=z3.BitVec('top_primary',32);root=z3.BitVec('top_root',30)
    n=z3.ZeroExt(2,root)^0xB55A4F09
    a=inputs(b,n,'arms',False);c=inputs(b^(1<<31),n,'arms',False)
    facts=[]
    def check(name,bad,extra=()):
        s=z3.SolverFor('QF_BV');s.set(timeout=10000);s.add(*extra,bad)
        assert s.check()==z3.unsat,name
        facts.append(name)
    for i in range(5):
        check(f'initial_{i}_low16_ignores_primary31',z3.Extract(15,0,a[i])!=z3.Extract(15,0,c[i]))
    check('target_bit15_flips_on_primary31',z3.Extract(15,15,target(a[0],n))==z3.Extract(15,15,target(c[0],n)))
    k=z3.BitVec('even_coefficient',32)
    check('even_multiplier_loses_highest_bit',k*(1<<31)!=0,(k&1==0,))
    return facts


def permutation_sum_controls():
    count=0
    for width in (4,6,8):
        modulus=1<<width
        for shift in range(1,width):
            total=(1<<shift)*((1<<(width-shift))*((1<<(width-shift))-1)//2)
            assert total%modulus==modulus//2
            for k in range(modulus):
                vals={(x+k*(x>>shift))%modulus for x in range(modulus)}
                if len(vals)==modulus:assert not k&1
                count+=1
    total=65536*(65536*65535//2)
    assert total & MASK==1<<31
    k=z3.BitVec('permutation_sum_K',32)
    s=z3.SolverFor('QF_BV');s.add(k*(total&MASK)==0,k&1==1)
    assert s.check()==z3.unsat
    return dict(exhaustive_small_coefficient_cases=count,sum_shr16_mod32=total&MASK,
                proof='A permutation preserves the sum of all uint32 values; g(t)=t+K*(t>>16) therefore requires K even')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();evidence={}
    def read(path):
        data=Path(path).read_bytes();evidence[path]=hashlib.sha256(data).hexdigest()
        return json.loads(gzip.decompress(data) if path.endswith('.gz') else data)
    src='codex/live_boundary863_arms_complete.json.gz'
    original=read(src);refine=read('codex/live_boundary863_arms_complete_refinement.json')
    assert original['completed']==len(original['manifest'])==len(original['results'])==113265
    assert not original['sampled_not_exhaustive'] and original['live']=='arms' and not original['runtime_node']
    assert original['node']==refine['node']==NODE
    assert refine['source_sha256']==digest(src)
    remaining={key(r) for r in original['results'] if r['status']=='unknown'}
    assert remaining=={key(r) for r in refine['results']} and len(remaining)==207
    rr={key(r):r for r in original['results']};rr.update({key(r):r for r in refine['results']})
    assert len(rr)==113265 and set(rr)=={key(r) for r in original['manifest']}
    ez=read('codex/live_boundary863_ez.json');prior=read('codex/live_boundary863_audit.json')
    assert prior['ez_remaining']==0 and original['reused_ez_sha256']==digest('codex/live_boundary863_audit.json')
    assert ez['node']==NODE and len(ez['manifest'])==6192 and not ez['sampled_not_exhaustive']
    for path,sha in prior['evidence_sha256'].items():assert digest(path)==sha,path
    ezkeys={key(r) for r in ez['manifest']}
    counts=Counter();pending=[]
    for r in rr.values():
        status=r['status']
        if status=='unsat':counts['smt_unsat']+=1
        elif status=='structural_excluded':
            assert r['order'][r['cut']-5]=='shr16'
            assert not primary_path_avoiding(r,5,False,r['cut'])
            counts['shr16_wordcut']+=1
        elif status=='prior_excluded':
            assert not any(j in (2,3,4) for w in r['wiring'] for j in w)
            compact=dict(order=r['order'],wiring=[[j if j<2 else j-3 for j in w] for w in r['wiring']])
            assert key(compact) in ezkeys
            counts['complete_ez_reuse']+=1
        elif status=='dependency_excluded':
            ks=[z3.BitVecs(f'low_{i}a low_{i}b',32) for i in range(3)]
            s=z3.SolverFor('QF_BV');s.add(*lsb_conditions(r,ks,'arms',False,NODE))
            assert s.check()==z3.unsat;counts['lsb_dependency']+=1
        else:
            assert status=='unknown', 'Verified candidates must be integrated, not excluded'
            pending.append(r)
    assert len(pending)==7
    rect=rectangle();im=image();tops=top_facts();sums=permutation_sum_controls()
    closed=[]
    for r in pending:
        assert r['order']==['mad','shr16','mad'] and r['wiring'][1]==[5]
        first,last=r['wiring'][0],r['wiring'][2]
        if last[2]==6 and 6 not in last[:2]:
            cert='integer_polynomial_plus_bounded_shift_rectangle'
        elif 6 in last[:2] and any(v>=7 for v in last[:2]) and last[2]==0:
            cert='residual_T_minus_e_image_exceeds_65536'
        elif first==[2,5,6] and set(last[:2])=={5,6} and last[2]<5:
            ks=[z3.BitVecs(f'top_{i}a top_{i}b',32) for i in range(3)]
            s=z3.SolverFor('QF_BV');s.add(*lsb_conditions(r,ks,'arms',False,NODE),ks[0][0]&1==1)
            assert s.check()==z3.unsat
            cert='lsb_requires_even_A_but_top_input_dependency_requires_odd_A'
        elif last[2]==5 and 6 in last[:2] and any(v>=7 for v in last[:2]):
            assert not primary_path_avoiding(r,5,False,5)
            constant=next(v for v in last[:2] if v>=7)-7
            ks=[z3.BitVecs(f'sum_{i}a sum_{i}b',32) for i in range(3)]
            s=z3.SolverFor('QF_BV');s.add(*lsb_conditions(r,ks,'arms',False,NODE),ks[2][constant]&1==0)
            assert s.check()==z3.unsat
            cert='lsb_requires_odd_K_but_permutation_sum_requires_even_K'
        else:raise AssertionError(r)
        counts[cert]+=1;closed.append(dict(order=r['order'],wiring=r['wiring'],certificate=cert))
    assert sum(counts.values())==113265
    result=dict(node=NODE,complete_manifest=113265,remaining=0,counts=dict(counts),
                certificates=closed,rectangle=rect,residual_image=im,top_dependency_facts=tops,
                permutation_sum=sums,evidence_sha256=evidence,best_executable_cycles=864,new_executable=False,
                mandatory_mad=True,shift_count=16,
                scope='One SHR16, one MAD, and a third MAD/XOR/AND/OR operation, with the five stated live hash words and arbitrary prepared constants for one legal root. No other shift counts or global kernel bound.')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(result['counts'],'remaining',result['remaining'])


if __name__=='__main__':main()
