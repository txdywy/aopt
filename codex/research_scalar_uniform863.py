"""Eliminate optional uniform vectors by using their scalar roots directly.

This changes mode availability: a scalar operation with a repeated constant
cannot silently become a contiguous vector operand. Released constant reloads
can be spent on other real staged vectors. Every candidate needs scheduling.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from codex import research_flow_stage863 as flow
from codex import research_stage864 as stage
from codex import research_frontier863 as frontier
from codex import research_sub864 as research
from codex.word_ssa import kinds,observed
from codex.packet_budget_bound import boundary,lane_weight


def scalar_uniform(builder,name,partial=False,vector=None,source=None,head_groups=8):
    definitions={v:op for op in builder.ops if op.engine!='store'for v in op.args[0]}
    if vector is None:
        vector=(builder.rootraw if name=='root'else builder.cache[0]if name=='cache0'
                else builder.vectors[int(name)&0xffffffff])
    producer=definitions[vector[0]]
    extra=[]
    if source is None:
        if producer.code=='vbroadcast':source=producer.args[1]
        elif producer.tag[0]=='constant_reload':source=(producer.tag[1][0],)
        elif name in(19,-2):
            source=builder.new(1)
            args=([source,builder.scalar[16],builder.scalar[3]]if name==19 else
                  [source,(builder.zero[0],),builder.scalar[2]])
            extra.append(SimpleNamespace(engine='alu',code='+'if name==19 else'-',
                args=args,kinds=kinds('alu','+'if name==19 else'-'),time=2,
                tag=('scalar_uniform_root',(name,))))
        else:raise ValueError(('unsupported uniform producer',name,producer.code))
    assert len(source)==1 and source[0]not in vector
    hashes={op.args[0]:op.tag[1]for op in builder.ops if op.tag[0]=='hash_mad'}
    rewritten=[]
    converted=0
    for op in builder.ops:
        matches=[i for i,(kind,arg)in enumerate(zip(op.kinds,op.args))
                 if kind!='imm'and not kind.endswith('w')and arg==vector]
        if matches and (not partial or hashes.get(op.args[1],(-1,-1))[0]==0
                        and hashes[op.args[1]][1]<head_groups):
            assert op.engine=='valu'and op.code not in('multiply_add','vbroadcast'),(name,op.code)
            for lane in range(8):
                args=[source if i in matches else(arg[lane],)for i,arg in enumerate(op.args)]
                rewritten.append(SimpleNamespace(engine='alu',code=op.code,args=args,
                    kinds=kinds('alu',op.code),time=op.time,tag=('scalar_uniform_use',(str(name),*op.tag[1]))))
            converted+=1
        else:
            if not partial:
                # Standalone scalar users must not retain the deleted vector alias.
                for i,(kind,arg)in enumerate(zip(op.kinds,op.args)):
                    if kind!='imm'and not kind.endswith('w')and any(v in vector for v in arg):
                        assert len(arg)==1,(name,op.code,arg)
                        op.args[i]=source
            rewritten.append(op)
    if not partial:
        deleted={id(producer)}
        if producer.tag[0]=='constant_reload':
            tag=('constant_stage',producer.tag[1])
            stores=[op for op in rewritten if op.tag[0]==tag[0]and op.tag[1][0]==tag[1][0]]
            assert len(stores)==8,(name,len(stores))
            deleted.update(map(id,stores))
        rewritten=[op for op in rewritten if id(op)not in deleted]
        if isinstance(name,int):builder.scalar[name&0xffffffff]=source
    builder.ops=extra+rewritten
    return dict(name=name,partial=partial,converted_vectors=converted,scalar_source=source[0])


def flow_4097(builder):
    vector=next(op for op in builder.ops if op.args[0]==builder.vectors[4097])
    source=builder.new(1)
    builder.put('flow','add_imm',[source,(builder.zero[0],),4097],1,('scalar_4097',()))
    vector.code,vector.args,vector.kinds,vector.time=(
        'vbroadcast',[vector.args[0],source],kinds('valu','vbroadcast'),2)
    builder.scalar[4097]=source


def build(full=(),methods=None,fused=0,trades=0,prolog19=0,init4097=None,header_packs=()):
    if methods:methods={k if k=='root'else int(k):v for k,v in methods.items()}
    builder=deepcopy(flow.original())
    if init4097=='flow':flow_4097(builder)
    elif init4097=='load':flow.early_4097(builder)
    elif init4097 is not None:raise ValueError('flow or load 4097 initializer required')
    converted=[]
    for name in full:converted.append(scalar_uniform(builder,name))
    flow.fuse_digits(builder,list(range(fused)))
    # Newly independent scalar roots must be evaluated after their inputs.
    frontier.logical_order(builder)
    packs=flow.materialize_uniforms(builder,methods)if methods else[]
    for key,_,source,out in packs:
        if key!='root':
            builder.vectors[int(key)&0xffffffff]=out
            builder.scalar[int(key)&0xffffffff]=source
    if prolog19:
        pack=next((p for p in packs if p[0]==19),None)
        converted.append(scalar_uniform(builder,19,True,
            vector=pack[3]if pack else None,source=pack[2]if pack else None,head_groups=prolog19))
    flow.trade_loads(builder,trades)
    frontier.logical_order(builder)
    if header_packs:
        frontier.pack_header(builder,header_packs)
        frontier.logical_order(builder)
    builder.sink=builder.literals=builder.literal_flow=0
    for _ in range(3):
        stage.prune_select_inputs(builder)
        ir=research.clean(builder)
    return ir,builder,converted


def measure(config,seconds=1):
    ir,builder,converted=build(**config)
    row=flow.measure(ir,seconds)
    grouped={j for _,_,ids,_ in ir.groups for j in ids}
    mandatory_alu=sum(op.engine=='alu'and j not in grouped for j,op in enumerate(ir.ops))
    row.update(config=config,converted=converted,mandatory_alu=mandatory_alu,
        mandatory_alu_cycle_bound=(mandatory_alu+11)//12,
        cycle_lower_bound=max(row['cycle_lower_bound'],(mandatory_alu+11)//12))
    needed=observed(ir)
    productive=sum(lane_weight(ir,j,True,needed)for j in range(len(ir.ops)))
    if productive<row['arithmetic_lanes']:
        head=boundary(ir,6,False,seconds,2,productive=True)
        tail=boundary(ir,3,True,seconds,2,productive=True)
        bound=(productive+head['waste_lower_bound']+tail['waste_lower_bound']+59)//60
        row['productive_bound']=dict(arithmetic_lanes=productive,prefix=head,suffix=tail,
            cycle_lower_bound=bound,excluded_unobserved_lanes=row['arithmetic_lanes']-productive)
        row['cycle_lower_bound']=max(row['cycle_lower_bound'],bound)
    return row


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--group',choices=('single','joint'),default='single')
    ap.add_argument('--controls',action='store_true')
    ap.add_argument('--seeds',type=int,default=2)
    args=ap.parse_args()
    if args.seeds<1:ap.error('positive seeds required')
    core=('cache0',0xb55a4f09)
    if args.group=='single':
        configs=[{}]+[dict(full=[s])for s in('root','cache0',19,-2,0xc761c23c,0xb55a4f09)]
        configs += [dict(full=[*core,*extra])for extra in((),(-2,),('root',),(-2,0xc761c23c),('root',-2,0xc761c23c))]
        configs += [dict(init4097=mode)for mode in('flow','load')]
    else:
        configs=[]
        for extra in ((-2,0xc761c23c),('root',-2,0xc761c23c),(-2,),('root',-2)):
            for initializer in(None,'flow','load'):
                for more in ({},{3:'third'},{3:'third',2310:'third'}):
                    methods={9:'stage',19:'stage',**more}
                    configs.append(dict(full=[*core,*extra],methods=methods,init4097=initializer,
                        prolog19=8,fused=max(0,(len(more)+(initializer=='flow')-1)//2)))
    if args.controls:
        joint=dict(full=[*core,-2,0xc761c23c],methods={9:'stage',19:'stage'},
                   prolog19=8,init4097='flow')
        configs=[dict(full=[*core,-2,0xc761c23c]),dict(full=['root',19],init4097='flow'),
                 dict(joint,methods={9:'stage',19:'stage',3:'third'}),
                 dict(joint,header_packs=['+:1','>>:1','-:2'])]
    results=[]
    for config in configs:
        if args.controls:
            ir,_,converted=build(**config)
            row=dict(config=config,converted=converted,seeds=args.seeds,
                     serial_control=flow.serial_control(ir,args.seeds))
            results.append(row)
            args.output.write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps(row),flush=True)
            continue
        row=measure(config);results.append(row)
        args.output.write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(dict(id=len(results)-1,config=config,C=row['arithmetic_lanes'],
            head=row['prefix']['waste_lower_bound'],tail=row['suffix']['waste_lower_bound'],
            load=row['streams']['load']['cycle_lower_bound'],flow=row['streams']['flow']['cycle_lower_bound'],
            mandatory_alu=row['mandatory_alu'],bound=row['cycle_lower_bound'])),flush=True)


if __name__=='__main__':main()
