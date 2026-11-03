"""Integrate live-value root-boundary macros into the real 864 compiler.

Only runtime-node macros are executable here. Fixed-root search results
must first be generalized; no benchmark input is evaluated by the host.
The built-in original four-op expression is a correctness/cost control.
Actual alternating u/v lane packing is checked in both orientations.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import pickle

import z3

from codex import research_flow_stage863 as flow
from codex import research_frontier863 as frontier
from codex import research_sub864 as research
from codex import stride4_family as family
from codex.search_live_boundary863 import inputs,expression,target,C0,C5,MASK


CONTROL=dict(live='ez',runtime_node=True,order=['shr16','^','^','mad'],
             wiring=[[0],[0,3],[4,2],[5,6,7]],
             constants=[[0,0],[0,0],[0,0],[4097,C0]])


def verify(macro,seconds=30):
    if not macro['runtime_node']:
        raise ValueError('A fixed-root formula must be generalized before compiler integration')
    assert macro['live'] in ('ez','arms')
    n=(2 if macro['live']=='ez' else 5)+1
    assert len(macro['order'])==len(macro['wiring'])==len(macro['constants'])
    for i,(op,w,cc) in enumerate(zip(macro['order'],macro['wiring'],macro['constants'])):
        assert op in ('mad','^','&','|','shr16')
        assert len(w)==(3 if op=='mad' else 1 if op=='shr16' else 2)
        assert len(cc)==2 and all(0<=j<n+i+2 for j in w)
    p=z3.BitVec('boundary_primary',32);raw=z3.BitVec('boundary_raw_root',30)
    node=z3.ZeroExt(2,raw)^C5
    cc=[[z3.BitVecVal(k&MASK,32) for k in pair] for pair in macro['constants']]
    for swapped in ((False,True) if macro['live']=='arms' else (False,)):
        vv=inputs(p,node,macro['live'],True)
        if swapped:vv[2],vv[3]=vv[3],vv[2]
        s=z3.SolverFor('QF_BV');s.set(timeout=int(seconds*1000))
        s.add(expression(macro,cc,vv)!=target(vv[0],node))
        status=s.check()
        if status!=z3.unsat:
            raise ValueError(f'Macro equivalence not proved for swapped={swapped}: {status}')


def build(macro,base=None):
    verify(macro)
    b=deepcopy(base if base is not None else flow.original())
    ops=list(b.ops);before=Counter(o.engine for o in ops)
    known,_=family.evaluate(b)
    writer={v:o for o in ops if o.engine!='store' for v in o.args[0]}
    users={}
    for o in ops:
        for kind,arg in zip(o.kinds,o.args):
            if kind!='imm' and not kind.endswith('w'):
                for v in arg:users.setdefault(v,set()).add(id(o))
    def uniform_is(arg,value):return all(known.get(v)==value for v in arg)
    matches=[]
    for nxt in ops:
        if nxt.tag[0]!='hash_mad' or nxt.tag[1][0]!=11:continue
        assert nxt.code=='multiply_add' and uniform_is(nxt.args[2],4097) and uniform_is(nxt.args[3],C0)
        mix=writer[nxt.args[1][0]];assert mix.code=='^'
        found=[]
        for i,arg in enumerate(mix.args[1:]):
            tail=writer.get(arg[0])
            if tail is None or tail.code!='^':continue
            for earg,sarg in (tail.args[1:],tail.args[1:][::-1]):
                e,shift=writer.get(earg[0]),writer.get(sarg[0])
                if e is None or shift is None or e.code!='multiply_add' or shift.code!='>>':continue
                if shift.args[1]!=e.args[0] or not uniform_is(shift.args[2],16):continue
                if not uniform_is(e.args[2],9) or not uniform_is(e.args[3],0xFD7046C5):continue
                found.append((e,shift,tail,mix.args[2-i]))
        assert len(found)==1,(nxt.tag,len(found))
        e,shift,tail,node=found[0]
        assert users[shift.args[0][0]]=={id(tail)}
        assert users[tail.args[0][0]]=={id(mix)}
        assert users[mix.args[0][0]]=={id(nxt)}
        z=writer[e.args[1][0]];assert z.code=='^'
        a,c=[writer[arg[0]] for arg in z.args[1:]]
        assert a.code==c.code=='multiply_add' and a.args[1]==c.args[1]
        for lane in range(8):
            coeffs=(known[a.args[2][lane]],known[c.args[2][lane]])
            assert set(coeffs)=={33,16896}
            for op,coef in zip((a,c),coeffs):
                assert known[op.args[3][lane]]==(0xE9F8CC1D if coef==33 else 0xACCF6200)
        values=[e.args[0],z.args[0]]
        if macro['live']=='arms':values += [a.args[0],c.args[0],a.args[1]]
        values.append(node)
        matches.append((nxt,mix,tail,shift,values))
    assert len(matches)==32,len(matches)
    live_uniforms={};live_scalars={}
    for v,value in known.items():live_scalars.setdefault(value,(v,))
    for o in ops:
        for kind,arg in zip(o.kinds,o.args):
            if kind!='imm' and len(arg)==8 and arg[0] in known:
                value=known[arg[0]]
                if uniform_is(arg,value):live_uniforms.setdefault(value,arg)
    setup=[];emitted=[]
    def uniform(value):
        value &= MASK
        if value==0:return b.zero
        if value in live_uniforms:return live_uniforms[value]
        if value in live_scalars:source=live_scalars[value]
        else:
            source=b.new(1);b.put('load','const',[source,value],100,('live_boundary_constant',(value,)))
            setup.append(b.ops.pop());live_scalars[value]=source
        out=b.new(8);b.put('valu','vbroadcast',[out,source],101,('live_boundary_uniform',(value,)))
        setup.append(b.ops.pop());live_uniforms[value]=out
        return out
    remove=set()
    for nxt,mix,tail,shift,initial in matches:
        remove.update(id(o) for o in (nxt,mix,tail,shift));values=list(initial)
        for i,(op,w,cc) in enumerate(zip(macro['order'],macro['wiring'],macro['constants'])):
            pool=values+[None,None]
            for j in set(w):
                if j>=len(values):pool[j]=uniform(cc[j-len(values)])
            args=[pool[j] for j in w]
            if op=='shr16':args.append(shift.args[2])
            out=nxt.args[0] if i==len(macro['order'])-1 else b.new(8)
            b.put('valu','multiply_add' if op=='mad' else '>>' if op=='shr16' else op,
                  [out,*args],nxt.time+i,('live_boundary',(nxt.tag[1][1],i)))
            emitted.append(b.ops.pop());values.append(out)
    b.ops=setup+[o for o in ops if id(o) not in remove]+emitted
    b.sink=b.literals=b.literal_flow=0
    frontier.logical_order(b);ir=research.clean(b)
    after=Counter(o.engine for o in ir.ops)
    return ir,dict(replaced_boundaries=len(matches),macro_instructions=len(macro['order']),
                   before=dict(before),after=dict(after),new_setup_operations=len(setup),
                   lane_work_before=before['alu']+8*before['valu'],
                   lane_work_after=after['alu']+8*after['valu'],
                   packed_arms_checked_both_orientations=macro['live']=='arms')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--macro',type=Path)
    ap.add_argument('--base-ir',type=Path)
    ap.add_argument('--seeds',type=int,default=3)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    macro=json.loads(args.macro.read_text()) if args.macro else CONTROL
    base=pickle.loads(args.base_ir.read_bytes()) if args.base_ir else None
    ir,report=build(macro,base)
    check=flow.serial_control(ir,args.seeds)
    report.update(macro=macro,full32_macro_verified=True,frozen_random_seeds=args.seeds,
                  scratch=check['scratch'],serial_cycles_not_a_score=check['verified'],
                  new_executable=False)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
