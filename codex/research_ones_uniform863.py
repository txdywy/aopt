"""Build an all-ones vector with five real scalar ALUs and one FLOW.

The selector also overwrites three zeros with zeros.  Consumers use their
new SSA versions; its remaining overlapping inputs are initially zero and
are live until the select.  Scratch allocation must enforce these lifetimes.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from codex import research_flow_stage863 as flow
from codex import research_frontier863 as frontier
from codex import research_stage864 as stage
from codex import research_sub864 as research
from codex.bound_boundary_no_solver863 import prefix,suffix
from codex.packet_budget_bound import stream_bounds
from codex.packet_modes import expand
from codex.research_triple_uniform863 import triple_pack
from codex.word_ssa import rebuild


def pack_ones(builder):
    old=builder.vectors[1]
    producer=next(op for op in builder.ops if op.engine=='valu' and op.args[0]==old)
    assert producer.code=='==' and producer.args[1]==producer.args[2]==builder.zero
    created=[]
    def emit(engine,code,args,time,tag):
        builder.put(engine,code,args,time,tag);created.append(builder.ops.pop())
    old_zero=builder.zero
    one=builder.new()
    emit('alu','==',[one,(old_zero[0],),(old_zero[0],)],0,('ones_root',()))
    one2=builder.new()
    emit('alu','|',[one2,one,(old_zero[0],)],1,('ones_condition_seed',()))
    seeds=[]
    for i in range(3):
        dest=builder.new()
        emit('alu','|',[dest,one,(old_zero[0],)],1,('ones_seed',(i,)))
        seeds.extend(dest)
    seeds=tuple(seeds);out=builder.new(8)
    zeros=builder.new(3)
    builder.initial.update({v:frozenset([0]) for v in zeros})
    cond=one+one2+old_zero[:6]
    yes=seeds+builder.new(5,undefined=True)
    no=builder.new(2,undefined=True)+seeds+zeros
    emit('flow','vselect',[out,cond,yes,no],2,('ones_pack',()))
    vector=seeds+out[:5]
    new_zero=out[5:]+old_zero[3:]
    mapping=dict(zip(old,vector))
    mapping.update(zip(old_zero,new_zero))
    kept=[op for op in builder.ops if op is not producer]
    for op in kept:
        for i,(kind,arg) in enumerate(zip(op.kinds,op.args)):
            if kind!='imm' and not kind.endswith('w'):
                # Seed zero remains one after the in-place select.
                op.args[i]=((seeds[0],) if len(arg)==1 and arg[0] in old else
                            tuple(mapping.get(v,v) for v in arg))
    insertion=next(i for i,op in enumerate(kept) if op.tag[0]=='hash_mad')
    builder.ops=kept[:insertion]+created+kept[insertion:]
    builder.vectors[1]=vector;builder.scalar[1]=(seeds[0],)
    builder.zero=new_zero;builder.scalar[0]=(new_zero[0],)
    builder.stats['ones_seed_lanes']=5;builder.stats['ones_flow_ops']=1
    return dict(old=list(old),root=one[0],condition_seed=one2[0],
                seeds=list(seeds),initial_zeros=list(zeros),output=list(out),
                vector=list(vector),new_zeros=list(new_zero))


def build(triple=None):
    builder=deepcopy(flow.original())
    layout=pack_ones(builder)
    if triple:
        layout['triple']=triple_pack(builder,tuple(triple),direct_scalars=True)
    frontier.logical_order(builder)
    builder.sink=builder.literals=builder.literal_flow=0
    for _ in range(3):
        stage.prune_select_inputs(builder);ir=research.clean(builder)
    return ir,builder,layout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--triple',help='optional names, e.g. root,c0,c1')
    parser.add_argument('--controls',action='store_true')
    args=parser.parse_args()
    ir,b,layout=build(args.triple.split(',') if args.triple else None)
    counts=Counter(op.engine for op in ir.ops)
    work=counts['alu']+8*counts['valu'];streams=stream_bounds(ir)
    expanded=expand(deepcopy(ir))
    expanded.views=[arg for _,operands,_,_ in expanded.groups for arg in operands]
    rebuild(expanded)
    head,tail=prefix(expanded),suffix(expanded)
    lower=max((work+head['waste_lower_bound']+tail['waste_lower_bound']+59)//60,
              *(v['cycle_lower_bound'] for v in streams.values()))
    result=dict(counts=dict(counts),arithmetic_lanes=work,streams=streams,prefix=head,suffix=tail,
        cycle_lower_bound=lower,layout=layout,stats=dict(b.stats),new_executable=False,
        scope='This concrete in-place ones pack and optional triple pack. Bound is necessary, not a score.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('arithmetic_lanes','cycle_lower_bound','prefix','suffix')}),flush=True)
    if args.controls:
        control=flow.serial_control(deepcopy(ir),2)
        result['serial_control']=dict(scratch=control['scratch'],
            serialized_cycles=control['verified'],seeds=2)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'serial_control':result['serial_control']}),flush=True)


if __name__=='__main__':main()
