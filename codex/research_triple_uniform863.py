"""Build three real eight-word uniforms with seven ALUs and two FLOWs.

Two intermediate B copies share their physical cells with later A outputs.
The vselect reads before writing, so the overwrite is legal only after every
other scalar B consumer. SSA rebuild and allocation must enforce that WAR.
No result is a contest score until a packed schedule passes frozen replay.
"""
from collections import Counter
from copy import deepcopy
import argparse
import json
from pathlib import Path

from codex import research_flow_stage863 as flow
from codex import research_frontier863 as frontier
from codex import research_stage864 as stage
from codex import research_sub864 as research
from codex.packet_budget_bound import stream_bounds
from codex.research_packet_870 import schedule,verify
from codex.word_ssa import allocate,condense


def triple_pack(builder,order=('root','c1','c0'),direct_scalars=False):
    if len(order)!=3 or len(set(order))!=3:
        raise ValueError('three distinct supported vectors are required')
    choices={'root':builder.rootraw,'c0':builder.vectors[0x7ed55d16],
             'c1':builder.vectors[0xc761c23c],
             '3':builder.vectors[3],'9':builder.vectors[9],
             '19':builder.vectors[19]}
    vectors=tuple(choices[name] for name in order)
    writer={v:op for op in builder.ops if op.engine!='store' for v in op.args[0]}
    producers=[writer[vector[0]] for vector in vectors]
    assert all(op.engine=='valu' and op.args[0]==vec
               for op,vec in zip(producers,vectors))
    created=[]
    def emit(engine,code,args,time,tag):
        builder.put(engine,code,args,time,tag)
        created.append(builder.ops.pop())
    roots_by_name={}
    for name,op in zip(order,producers):
        if name in ('root','c0','c1'):
            assert op.code=='vbroadcast'
            roots_by_name[name]=op.args[1]
    if '3' in order:
        root=builder.new();emit('alu','cdiv',[root,builder.scalar[16],builder.scalar[7]],1,('triple_root',('3',)))
        roots_by_name['3']=root
    if '9' in order:
        root=builder.new();emit('alu','-',[root,builder.scalar[10],builder.scalar[1]],1,('triple_root',('9',)))
        roots_by_name['9']=root
    if '19' in order:
        root=builder.new();three=roots_by_name.get('3',builder.scalar[3])
        emit('alu','+',[root,builder.scalar[16],three],2,('triple_root',('19',)))
        roots_by_name['19']=root
    roots=[roots_by_name[name] for name in order]
    assert all(len(root)==1 for root in roots)
    assert len({root[0] for root in roots})==3
    # Header lane 7 has seven other long-lived neighbors. If it is not the
    # first seed, detach it so the header cannot overlap the packing window.
    root_role=order.index('root') if 'root' in order else None
    if root_role is not None and root_role!=0:
        detached=builder.new()
        emit('alu','|',[detached,roots[root_role],(builder.zero[0],)],2,('triple_detach_root',()))
        roots[root_role]=detached
    def copies(root,count,label):
        result=list(root)
        for i in range(count):
            dest=builder.new()
            emit('alu','|',[dest,root,(builder.zero[0],)],2,('triple_seed',(label,i)))
            result.extend(dest)
        return tuple(result)
    a=copies(roots[0],3,'a')
    b=copies(roots[1],1,'b')
    c=copies(roots[2],3,'c')
    first=builder.new(8);second=builder.new(8)
    ones=builder.vectors[1]
    first_cond=ones[2:]+builder.zero[:2]
    second_cond=ones[4:]+builder.zero[:4]
    yes=a+b+builder.new(2,undefined=True)
    no=builder.new(2,undefined=True)+a+b
    emit('flow','vselect',[first,first_cond,yes,no],6,('triple_pack',(0,)))
    yes2=first[4:]+builder.new(4,undefined=True)
    no2=builder.new(4,undefined=True)+c
    emit('flow','vselect',[second,second_cond,yes2,no2],7,('triple_pack',(1,)))
    newvectors=(a+first[:4],first[4:]+second[:4],second[4:]+c)
    mapping={old:new for oldvec,newvec in zip(vectors,newvectors) for old,new in zip(oldvec,newvec)}
    scalar_mapping={old:roots[index][0] for index,(name,oldvec) in enumerate(zip(order,vectors))
                    if index!=1 or name=='9' for old in oldvec}
    removed={id(op) for op in producers}
    kept=[op for op in builder.ops if id(op) not in removed]
    # Replace individual uses as well as full views. Mapping B scalar uses
    # to its materialized vector avoids reading its overwritten root later.
    for op in kept:
        for i,(kind,arg) in enumerate(zip(op.kinds,op.args)):
            if kind!='imm' and not kind.endswith('w'):
                selected=scalar_mapping if direct_scalars and len(arg)==1 else mapping
                op.args[i]=tuple(selected.get(v,v) for v in arg)
    insertion=next(i for i,op in enumerate(kept) if op.tag[0]=='hash_mad')
    builder.ops=kept[:insertion]+created+kept[insertion:]
    for name,vector in zip(order,newvectors):
        if name=='root':builder.rootraw=vector
        else:
            key={'c0':0x7ed55d16,'c1':0xc761c23c,'3':3,'9':9,'19':19}[name]
            builder.vectors[key]=vector
            builder.scalar[key]=(vector[0],)
    builder.stats['triple_uniform_vectors']=3
    builder.stats['triple_uniform_seed_lanes']=7+int(root_role is not None and root_role!=0)
    builder.stats['triple_uniform_flow_ops']=2
    return dict(order=list(order),direct_scalars=direct_scalars,
        root_detached=root_role is not None and root_role!=0,
        roots=[r[0] for r in roots],seed_lengths=[len(a),len(b),len(c)],
        first_output=list(first),second_output=list(second),vectors=[list(v) for v in newvectors])


def build(order=('root','c1','c0'),direct_scalars=False):
    builder=deepcopy(flow.original())
    layout=triple_pack(builder,order,direct_scalars)
    frontier.logical_order(builder)
    builder.sink=builder.literals=builder.literal_flow=0
    for _ in range(3):
        stage.prune_select_inputs(builder)
        ir=research.clean(builder)
    return ir,builder,layout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,default=2)
    parser.add_argument('--schedule',action='store_true')
    parser.add_argument('--order',default='root,c1,c0')
    parser.add_argument('--direct-scalars',action='store_true')
    args=parser.parse_args()
    if args.seeds<1:parser.error('positive replay seeds required')
    ir,b,layout=build(tuple(args.order.split(',')),args.direct_scalars)
    counts=Counter(op.engine for op in ir.ops)
    report=dict(counts=dict(counts),arithmetic_lanes=counts['alu']+8*counts['valu'],
        streams=stream_bounds(ir),layout=layout,stats=dict(b.stats),new_executable=False,
        scope='One concrete triple-uniform graph. The serial replay below is correctness '
              'only; only a separately verified packed schedule can be a score.')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('counts','arithmetic_lanes','streams')}),flush=True)
    control=flow.serial_control(deepcopy(ir),args.seeds)
    report['serial_control']=dict(scratch=control['scratch'],
        serialized_cycles=control['verified'],seeds=args.seeds)
    print(json.dumps({'serial_control':report['serial_control']}),flush=True)
    if args.schedule:
        graph=condense(ir)
        resources=[(op.time,op.engine,(),op.tag) for op in ir.ops]
        _,cluster_times=schedule(resources,graph)
        times=[cluster_times[graph.owner[j]] for j in range(len(ir.ops))]
        program,words,_=allocate(ir,times)
        assert words<=1536
        cycles=verify(program,args.seeds)
        report['packed_schedule']=dict(cycles=cycles,scratch=words,seeds=args.seeds)
        # A frozen replay is necessary but is not the full official suite.
        report['candidate_sub864']=cycles<864
        print(json.dumps({'packed_schedule':report['packed_schedule']}),flush=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
