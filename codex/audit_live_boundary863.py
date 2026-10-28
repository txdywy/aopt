"""Combine live-value searches, prior certificates, and a legal-root witness."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import z3

from codex.search_live_boundary863 import inputs,expression,target,C0,C5,MASK,primary_path_avoiding
from tests.frozen_problem import Tree,Input,reference_kernel,myhash,myhash_traced


def key(r):return tuple(r['order']),tuple(tuple(w) for w in r['wiring'])
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def legal_sample(row):
    """Use a real valid root at which the remaining runtime-node graph fails."""
    raw=0x355A4F09;node=raw ^ C5;assert node==1<<31 and raw<1<<30
    ks=[z3.BitVecs(f'witness_{i}a witness_{i}b',32) for i in range(3)]
    values=inputs(z3.BitVecVal(0,32),z3.BitVecVal(node,32),'ez',True)
    s=z3.SolverFor('QF_BV');s.set(timeout=10000)
    s.add(expression(row,ks,values)==target(values[0],node))
    result=s.check();assert result==z3.unsat,result
    # Invert the triangular middle XOR to make z=0, then invert g19 and MAD0.
    b=0
    for bit in range(32):
        zz=((33*b+0xE9F8CC1D)&MASK)^((16896*b+0xACCF6200)&MASK)
        if (zz>>bit)&1:b ^= 1<<bit
    assert (((33*b+0xE9F8CC1D)&MASK)^((16896*b+0xACCF6200)&MASK))==0
    a=b ^ 0xC761C23C;a ^= a>>19
    p=((a-C0)*pow(4097,-1,1<<32))&MASK
    trace={};myhash_traced(p,trace,10,0)
    assert trace[(10,0,'hash_stage',3)]==0 and trace[(10,0,'hash_stage',4)]==0xFD7046C5
    witnesses={}
    for initial in range(64):
        value,index=initial,0
        for _ in range(10):
            value=myhash(value ^ (raw if index==0 else 0))
            index=2*index+(1 if value%2==0 else 2)
        witnesses.setdefault(value>>30,(initial,value,index))
    initial,before,index=witnesses[p>>30]
    leaf=p ^ before;assert leaf<1<<30
    nodes=[0]*2047;nodes[0]=raw;nodes[index]=leaf
    inp=Input([0],[initial],11);reference_kernel(Tree(10,nodes),inp)
    assert inp.values[0]==myhash(p) and inp.indices[0]==0
    trace={};myhash_traced(inp.values[0]^raw,trace,11,0)
    next_a=trace[(11,0,'hash_stage',0)]
    assert next_a==target(0xFD7046C5,node) and next_a&1
    return dict(order=row['order'],wiring=row['wiring'],status='unsat',width=32,
                primary_z=0,raw_root=raw,effective_node=node,
                all_private_constants_free=True,
                legal_tree=dict(height=10,root=raw,initial=initial,leaf_index=index,
                                leaf_value=leaf,other_nodes=0),
                leaf_hash_input=p,frozen_next_mad=next_a,
                certificate='One valid tree/input point is inconsistent for every private coefficient')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();evidence={}
    def read(path):
        evidence[path]=digest(path);return json.loads(Path(path).read_text())
    def combine(src,ref=None):
        a=read(src);assert a['completed']==len(a['manifest'])==len(a['results'])
        results={key(r):r for r in a['results']}
        assert len(results)==len(a['manifest'])
        if ref:
            b=read(ref);assert b['source_sha256']==digest(src)
            pending={k for k,r in results.items() if r['status'] not in ('unsat','structural_excluded','proved_equivalent')}
            assert pending=={key(r) for r in b['results']}
            results.update({key(r):r for r in b['results']})
        for r in results.values():
            n=(2 if r['live']=='ez' else 5)+int(r['runtime_node'])
            if r['status']=='structural_excluded':
                assert r['order'][r['cut']-n]=='shr16'
                assert not primary_path_avoiding(r,n,r['runtime_node'],r['cut'])
            if r.get('necessary_first_prefix_mod4_bijection'):
                assert r['order'][0]!='shr16' and not primary_path_avoiding(r,n,r['runtime_node'],n)
        return a,results
    ez,er=combine('codex/live_boundary863_ez.json','codex/live_boundary863_ez_refinement.json')
    old=read('codex/root_boundary863_scout.json')
    old_unsat={key(r) for r in old['results'] if r['status']=='unsat'}
    prior=[]
    for r in er.values():
        if r['status'] in ('unsat','structural_excluded'):continue
        assert r['status'] in ('unknown','verification_unknown'), 'A verified macro must be integrated'
        assert not r['runtime_node'] and r['live']=='ez' and r['fixed_node']==old['node']
        assert all(1 not in w for w in r['wiring']), 'Original z is a real bypass'
        compact=dict(order=r['order'],wiring=[[0 if j==0 else j-1 for j in w] for w in r['wiring']])
        assert key(compact) in old_unsat
        prior.append(dict(order=r['order'],wiring=r['wiring'],compact=compact,
                          certificate='Unused z: prior one-input e graph is UNSAT'))
    arms,ar=combine('codex/live_boundary863_arms_scout.json','codex/live_boundary863_arms_refinement.json')
    assert all(r['status'] in ('unsat','structural_excluded') for r in ar.values())
    runtime,rr=combine('codex/live_boundary863_runtime_node_scout.json')
    remaining=[r for r in rr.values() if r['status'] not in ('unsat','structural_excluded')]
    assert len(remaining)==1 and remaining[0]['status']=='unknown'
    witness=legal_sample(remaining[0])
    result=dict(ez_complete_manifest=len(er),ez_prior_certificates=prior,ez_remaining=0,
                arms_sampled_manifest=len(ar),arms_sampled_not_exhaustive=True,arms_remaining=0,
                runtime_node_sampled_manifest=len(rr),runtime_node_sampled_not_exhaustive=True,
                runtime_node_legal_sample=witness,runtime_node_remaining=0,
                best_executable_cycles=864,new_executable=False,evidence_sha256=evidence,
                scope='Only the stated SHR16 three-op live-word grammars/manifests. No other shift amounts or global kernel bound.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print({k:result[k] for k in ('ez_complete_manifest','ez_remaining','arms_sampled_manifest','runtime_node_sampled_manifest')})


if __name__=='__main__':main()
