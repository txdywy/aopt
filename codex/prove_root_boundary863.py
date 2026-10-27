"""Check the dependency facts behind the one-SHR boundary reduction.

For three instructions with one literal SHR and two MAD/XOR/AND/OR ops,
the shift must be 16: smaller shifts cannot bring input bit 16 to output
bit zero; larger shifts either lose input bit 31, or introduce a freely
toggling bit s into the sole shifted bit-zero argument. The final Boolean
function cannot ignore that nuisance bit while depending on input bit 16.
The written argument is in SUB863_ROOT_BOUNDARY_20261028.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import z3

from codex.search_root_boundary863 import C0,MASK,expression,target,manifest,joint_manifest
from codex.search_root_affine863 import body
from tests.frozen_problem import Machine,DebugInfo,Tree,Input,myhash,myhash_traced,reference_kernel


def universal_checks():
    x,n,a,b=z3.BitVecs('x n a b',32)
    facts=[]
    def check(name,bad,assumptions=()):
        s=z3.SolverFor('QF_BV');s.set(timeout=10000)
        s.add(*assumptions,bad)
        status=s.check()
        assert status==z3.unsat,(name,status,s.reason_unknown())
        facts.append(dict(name=name,status='unsat'))
    bit=lambda v,j:z3.Extract(j,j,v)
    check('target_bit0_flips_on_input_bit16',bit(target(x ^ 65536,n),0)!=bit(target(x,n),0)^1)
    check('target_bit15_flips_on_input_bit31',bit(target(x ^ (1<<31),n),15)!=bit(target(x,n),15)^1)
    check('target_bit12_flips_on_input_bit0',bit(target(x ^ 1,n),12)!=bit(target(x,n),12)^1)
    for j in range(17,32):
        check(f'target_bit0_ignores_input_bit{j}',bit(target(x ^ (1<<j),n),0)!=bit(target(x,n),0))
    high=1<<31
    delta=z3.simplify(a*(x+high)+b-(a*x+b),som=True)
    check('even_affine_loses_input_bit31',delta!=0,(a&1==0,))
    delta=z3.simplify((x+high)*(x+high)-x*x,som=True)
    check('square_loses_input_bit31',delta!=0)
    for j in range(1,32):
        step=1<<j
        delta=z3.simplify(a*(x+step)+b-(a*x+b)-step,som=True)
        check(f'odd_affine_diagonal_{j}',z3.Extract(j,0,delta)!=0,(a&1==1,))
        delta=z3.simplify((x+step)*(x+step)+(x+step)-x*x-x-step,som=True)
        check(f'square_plus_x_diagonal_{j}',z3.Extract(j,0,delta)!=0)
    fold=lambda v:v ^ z3.LShR(v,16)
    check('fold_is_an_involution',fold(fold(x))!=x)
    inverse=pow(4097,-1,1<<32)
    assert inverse*4097 & MASK==1
    # Once the shifted bit is independently toggled by a nuisance input,
    # an arbitrary last operation sees only (input_bit0, shifted_bit0).
    table=z3.Bools('f00 f01 f10 f11')
    node_bit=z3.Bool('node_bit')
    assertions=[]
    for input0 in (False,True):
        for input16 in (False,True):
            desired=z3.Xor(z3.Xor(z3.BoolVal(input0),z3.BoolVal(input16)),node_bit)
            for shifted0 in (False,True):
                assertions.append(table[2*int(input0)+int(shifted0)]==desired)
    s=z3.Solver();s.add(*assertions)
    assert s.check()==z3.unsat
    facts.append(dict(name='arbitrary_final_bit_function_cannot_discard_nuisance',status='unsat'))
    return dict(full32_facts=facts,odd_multiplier_inverse=inverse)


def small_width_controls():
    counts={}
    for width in (4,6):
        mask,high=(1<<width)-1,1<<(width-1)
        models=[]
        for a in range(mask+1):
            for b in range(mask+1):
                models.append(('poly',[(a*x+b)&mask for x in range(mask+1)]))
        for b in range(mask+1):
            models.append(('poly',[(x*x+b)&mask for x in range(mask+1)]))
        models.append(('poly',[(x*x+x)&mask for x in range(mask+1)]))
        for k in range(mask+1):
            for op in ('^','&','|'):
                models.append(('bit',[(x^k if op=='^' else x&k if op=='&' else x|k) for x in range(mask+1)]))
        for kind,values in models:
            assert all(values[x]&1==values[x&1]&1 for x in range(mask+1))
            top_lost=all(values[x]==values[x^high] for x in range(mask+1))
            for j in range(1,width):
                if kind=='poly' and not top_lost:
                    assert all(((values[x]^values[x^(1<<j)])>>j)&1 for x in range(mask+1))
                if kind=='bit':
                    assert all((values[x]>>j)&1==(values[x&(1<<j)]>>j)&1 for x in range(mask+1))
                prefix=(1<<(j+1))-1
                assert all(values[x]&prefix==values[x&prefix]&prefix for x in range(mask+1))
        counts[str(width)]=len(models)
    return dict(unary_first_programs=counts,complete_domains=True)


def frozen_program(order,wiring,constants,inputs):
    program=[];initial={};next_base=0
    def vec(values):
        nonlocal next_base
        base=next_base;next_base+=8
        initial[base]=list(values)
        return base
    x=vec(inputs);shift=vec([16]*8);values=[x]
    for op,w,kk in zip(order,wiring,constants):
        pool=values+[vec([k]*8) for k in kk]
        args=[pool[i] for i in w]
        out=vec([0]*8)
        if op=='shr16':slot=('>>',out,args[0],shift)
        else:slot=('multiply_add' if op=='mad' else op,out,*args)
        program.append({'valu':[slot]});values.append(out)
    machine=Machine([0]*32,program,DebugInfo({}))
    machine.enable_pause=False;machine.enable_debug=False
    for base,items in initial.items():machine.cores[0].scratch[base:base+8]=items
    machine.run()
    return machine.cores[0].scratch[values[-1]:values[-1]+8]


def frozen_controls():
    rng=random.Random(8642813)
    runs=0
    row=dict(order=['shr16','^','mad'],wiring=[[0],[0,1],[2,3,4]])
    for node in (0,1<<31):
        cc=[[0,0],[0,0],[4097,(C0+node)&MASK]]
        for _ in range(16):
            xs=[rng.getrandbits(32) for _ in range(8)]
            got=frozen_program(row['order'],row['wiring'],cc,xs)
            assert got==[target(x,node) for x in xs]
            runs+=1
    c=dict(A=9,B=0xfd7046c5,G=4097,H=C0)
    for family in ('fold_affine','raw_xor','separate_affine'):
        if family=='separate_affine':
            order=['shr16','mad','^','mad'];wiring=[[0],[0,2,3],[1,2],[3,4,5]]
            cc=[[0,0],[c['A'],c['B']],[0,0],[c['G'],c['H']]]
        else:
            order=['mad','shr16','^','mad'];wiring=[[0,1,2],[1],[0 if family=='raw_xor' else 1,2],[3,4,5]]
            cc=[[c['A'],c['B']],[0,0],[0,0],[c['G'],c['H']]]
        for _ in range(16):
            xs=[rng.getrandbits(32) for _ in range(8)]
            got=frozen_program(order,wiring,cc,xs)
            assert got==[body(x,c,family) for x in xs]
            assert got==[expression(order,wiring,cc,x) for x in xs]
            runs+=1
    for rows in (manifest(),joint_manifest()):
        for _ in range(300):
            row=rng.choice(rows)
            cc=[[rng.getrandbits(32),rng.getrandbits(32)] for _ in row['order']]
            xs=[rng.getrandbits(32) for _ in range(8)]
            got=frozen_program(row['order'],row['wiring'],cc,xs)
            assert got==[expression(row['order'],row['wiring'],cc,x) for x in xs]
            with np.errstate(over='ignore'):
                assert got==expression(row['order'],row['wiring'],cc,np.array(xs,dtype=np.uint64)).tolist()
            kk=[[z3.BitVecVal(v,32) for v in pair] for pair in cc]
            assert got==[z3.simplify(expression(row['order'],row['wiring'],kk,z3.BitVecVal(x,32))).as_long() for x in xs]
            runs+=1
    return dict(frozen_vector_runs=runs,full32_lanes=8*runs,known_programs=5,
                random_wirings=600,integer_numpy_z3_frozen_agree=True)


def reachable_boundary_controls():
    """Construct legal 30-bit trees/initial values reaching every input quarter.

    With a fixed root and all other nonleaves zero, four initial values
    give all four high-two-bit classes just before the leaf. A free 30-bit
    leaf then produces any desired uint32 hash input p in its class.
    The first five hash stages are a permutation, so their output e also
    covers all uint32 words. Reference traversal checks the construction.
    """
    root=0x12345678
    witnesses={}
    for initial in range(64):
        value,index=initial,0
        for _ in range(10):
            value=myhash(value ^ (root if index==0 else 0))
            index=2*index+(1 if value%2==0 else 2)
        witnesses.setdefault(value>>30,dict(initial=initial,before_leaf=value,leaf=index))
    assert set(witnesses)=={0,1,2,3}
    rng=random.Random(8642814)
    ps=[0,MASK,1<<30,1<<31]+[rng.getrandbits(32) for _ in range(256)]
    for p in ps:
        w=witnesses[p>>30]
        leaf=p ^ w['before_leaf']
        assert leaf < 1<<30 and 1023<=w['leaf']<2047
        nodes=[0]*2047;nodes[0]=root;nodes[w['leaf']]=leaf
        inp=Input([0],[w['initial']],11)
        reference_kernel(Tree(10,nodes),inp)
        assert inp.values[0]==myhash(p)
        trace={};out=myhash_traced(p,trace,10,0)
        e=trace[(10,0,'hash_stage',4)]
        assert out==(e ^ (e>>16) ^ 0xB55A4F09)
        assert target(e,root ^ 0xB55A4F09)==(4097*(out ^ root)+C0)&MASK
    return dict(fixed_raw_root=root,effective_node=root ^ 0xB55A4F09,
                quarter_witnesses=witnesses,constructed_reference_cases=len(ps),
                scalar_reference_embeds_by_repeating_lanes=True,
                first_five_stages_bijective_factors=[4097,33,9],
                remaining_factors='g19 XOR constant and (x+C3) XOR (x<<9), both triangular bijections')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result=dict(dependencies=universal_checks(),small_controls=small_width_controls(),
                frozen_controls=frozen_controls(),reachable=reachable_boundary_controls(),new_executable=False)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(len(result['dependencies']['full32_facts']),'universal facts',result['small_controls'],result['frozen_controls'])


if __name__=='__main__':main()
