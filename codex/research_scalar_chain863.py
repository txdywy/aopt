"""Compose real scalar-uniform elimination with long FLOW uniform chains.

No frozen files or submission are changed.  Bounds are necessary conditions
for the generated graph; a serial correctness replay is not a contest score.
"""
from collections import Counter
from copy import deepcopy
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from codex import research_flow_stage863 as flow
from codex import research_frontier863 as frontier
from codex import research_scalar_uniform863 as scalar
from codex import research_stage864 as stage
from codex import research_sub864 as research
from codex.bound_boundary_no_solver863 import order_and_distance,prefix,suffix
from codex.packet_budget_bound import stream_bounds
from codex.packet_modes import expand
from codex.research_chain_uniform863 import pack
from codex.word_ssa import kinds,rebuild


def parse_values(text):
    return tuple(item if item in ('root','cache0') else int(item,0)
                 for item in text.split(',') if item)


def direct_constant(code,a,b):
    """The frozen scalar ALU on known unsigned words, or None if invalid."""
    if code=='+':v=a+b
    elif code=='-':v=a-b
    elif code=='*':v=a*b
    elif code=='^':v=a^b
    elif code=='&':v=a&b
    elif code=='|':v=a|b
    elif code=='<<':v=(a<<b) if b<32 else 0
    elif code=='>>':v=(a>>b) if b<32 else 0
    elif code=='//':v=a//b if b else None
    elif code=='cdiv':v=-(-a//b) if b else None
    elif code=='%':v=a%b if b else None
    else:raise ValueError(code)
    return None if v is None else v&0xffffffff


def rederive_early_scalars(builder,ir,extended=False,ready_limit=2):
    """Recompute known setup scalars from immutable early definitions.

    Header fields, initially zero words, the native ones producer and direct
    literal producers are the only sources; their definitions are never
    rewritten here.  No result is accepted without graph rebuild/replay.
    """
    _,distance=order_and_distance(ir.deps,ir.succ)
    stable=set(builder.header[:7])|set(builder.initial)
    stable.add(builder.vectors[1][0])
    for j,op in enumerate(ir.ops):
        if op.engine=='load' and op.code=='const':stable.update(op.args[0])
        elif (op.engine=='flow' and op.code=='add_imm' and
              op.args[1]==(builder.zero[0],)):stable.update(op.args[0])
        elif extended and distance[j]+1<=ready_limit and op.engine in ('alu','valu'):
            stable.update(v for v in op.args[0] if v in ir.known)
    pool={}
    for v in stable:
        if v not in ir.known:continue
        writer=ir.writer.get(v)
        ready=0 if writer is None else distance[writer]+1
        if ready>ready_limit:continue
        value=ir.known[v]
        if value not in pool or (ready,v)<pool[value]:pool[value]=(ready,v)
    # Never rewrite an ancestor of an adopted source. This protects the new
    # edges even when several setup expressions are changed in one pass.
    frozen=set();pending=[ir.writer[v] for _,v in pool.values() if v in ir.writer]
    while pending:
        j=pending.pop()
        if j in frozen:continue
        frozen.add(j);pending.extend(ir.deps[j])
    codes=('+','-','^','|','&','>>','<<','*','//','cdiv','%')
    alternatives={}
    for a,(ta,va) in pool.items():
        for b,(tb,vb) in pool.items():
            for rank,code in enumerate(codes):
                value=direct_constant(code,a,b)
                if value is None:continue
                choice=(max(ta,tb)+1,rank,va,vb,code,a,b)
                if value not in alternatives or choice<alternatives[value]:alternatives[value]=choice
    changed=[]
    for j,op in enumerate(ir.ops):
        if j in frozen or op.engine!='alu' or len(op.args[0])!=1:continue
        dest=op.args[0][0]
        if dest in stable or dest not in ir.known:continue
        choice=alternatives.get(ir.known[dest])
        if choice is None or choice[0]>=distance[j]+1:continue
        ready,_,a,b,code,av,bv=choice
        old=(op.code,op.args[1],op.args[2])
        op.code,op.args,op.kinds,op.time=(code,[(dest,),(a,),(b,)],
                                            kinds('alu',code),ready-1)
        changed.append(dict(value=ir.known[dest],old_code=old[0],new_code=code,
                            operands=[av,bv],old_earliest=distance[j]+1,new_earliest=ready,
                            tag=list(op.tag)))
    return dict(extended=extended,ready_limit=ready_limit,source_values=len(pool),frozen_ops=len(frozen),
                rewritten=len(changed),operations=changed)


def early_scalar_root(builder,count):
    """Bypass a late root uniform for the first few root-XOR groups."""
    if not 0<=count<=32:raise ValueError('head-root count must be 0..32')
    roots={op.args[1]:op.tag[1][1] for op in builder.ops
           if op.tag[0]=='hash_mad' and op.tag[1][0]==0}
    rows=[];converted=[]
    for op in builder.ops:
        group=roots.get(op.args[0]) if op.engine=='valu' and op.code=='^' else None
        if group is None or group>=count or op.args[2]!=builder.rootraw:
            rows.append(op);continue
        converted.append(group)
        for lane in range(8):
            rows.append(SimpleNamespace(engine='alu',code='^',
                args=[(op.args[0][lane],),(op.args[1][lane],),(builder.header[7],)],
                kinds=kinds('alu','^'),time=op.time,tag=('head_root_scalar',(group,lane))))
    assert len(converted)==count,(len(converted),count)
    builder.ops=rows
    return converted


def early_scalar_19(builder,count,vector=None,source=None):
    """Read the terminal chain seed for selected first-round shifts."""
    if not 0<=count<=32:raise ValueError('prolog19 count must be 0..32')
    hashes={op.args[0]:op.tag[1] for op in builder.ops if op.tag[0]=='hash_mad'}
    vector=vector if vector is not None else builder.vectors[19]
    source=source if source is not None else (vector[4],)
    rows=[];converted=[]
    for op in builder.ops:
        coords=hashes.get(op.args[1]) if op.engine=='valu' and op.code=='>>' else None
        if coords is None or coords[0]!=0 or coords[1]>=count or op.args[2]!=vector:
            rows.append(op);continue
        group=coords[1];converted.append(group)
        for lane in range(8):
            rows.append(SimpleNamespace(engine='alu',code='>>',
                args=[(op.args[0][lane],),(op.args[1][lane],),source],
                kinds=kinds('alu','>>'),time=op.time,tag=('head_shift19_scalar',(group,lane))))
    assert len(converted)==count,(len(converted),count)
    builder.ops=rows
    return converted


def pack_staged_cache_pairs(builder,indices):
    """Replace pairs of staged node uniforms with six ALUs and one FLOW."""
    if len(indices)%2 or len(set(indices))!=len(indices):
        raise ValueError('cache indices must be distinct pairs')
    writers={v:op for op in builder.ops if op.engine!='store' for v in op.args[0]}
    selected=[];removed=set()
    for index in indices:
        if not 0<=index<len(builder.cache):raise ValueError('invalid cache index')
        vec=builder.cache[index];op=writers[vec[0]]
        assert op.engine=='load' and op.code=='vload' and op.tag[0]=='constant_reload'
        source=op.tag[1][0];stores=[o for o in builder.ops if o.tag[0]=='constant_stage'
                                   and o.tag[1][0]==source]
        assert len(stores)==8,(index,len(stores))
        assert len(writers[source].args[0])==1
        selected.append((index,vec,(source,)))
        removed.add(id(op));removed.update(map(id,stores))
    created=[];mapping={};layouts=[]
    def emit(engine,code,args,time,tag):
        builder.put(engine,code,args,time,tag);created.append(builder.ops.pop())
    def seed(source,label):
        result=list(source)
        for i in range(3):
            out=builder.new()
            emit('alu','|',[out,source,(builder.zero[0],)],12,('cache_pair_seed',(label,i)))
            result.extend(out)
        return tuple(result)
    for part in range(0,len(selected),2):
        (i,old_a,src_a),(j,old_b,src_b)=selected[part:part+2]
        a,b=seed(src_a,i),seed(src_b,j)
        out=builder.new(8)
        cond=builder.vectors[1][4:]+builder.zero[:4]
        yes=a+builder.new(4,undefined=True)
        no=builder.new(4,undefined=True)+b
        emit('flow','vselect',[out,cond,yes,no],16,('cache_pair_pack',(i,j)))
        va,vb=a+out[:4],out[4:]+b
        mapping.update(zip(old_a,va));mapping.update(zip(old_b,vb))
        builder.cache[i],builder.cache[j]=va,vb
        layouts.append(dict(indices=[i,j],vectors=[list(va),list(vb)]))
    kept=[op for op in builder.ops if id(op) not in removed]
    for op in kept:
        for k,(kind,arg) in enumerate(zip(op.kinds,op.args)):
            if kind!='imm' and not kind.endswith('w'):
                op.args[k]=tuple(mapping.get(v,v) for v in arg)
    insertion=next(i for i,op in enumerate(kept) if op.tag[0]=='hash_mad')
    builder.ops=kept[:insertion]+created+kept[insertion:]
    return layouts


def stage_third_caches(builder,indices):
    """Trade existing three-seed cache FLOWs for real STORE/LOAD stages."""
    if len(set(indices))!=len(indices):raise ValueError('duplicate cache index')
    writers={v:op for op in builder.ops if op.engine!='store' for v in op.args[0]}
    selected=[];removed=set()
    for index in indices:
        if not 0<=index<len(builder.cache):raise ValueError('invalid cache index')
        old=builder.cache[index];source=(old[0],)
        seedops=[writers[old[i]] for i in (1,2)]
        selector=writers[old[3]]
        assert all(op.engine=='alu' and op.code=='|' and op.args[1]==source for op in seedops)
        assert selector.engine=='flow' and selector.code=='vselect'
        assert old[3:]==selector.args[0][:5]
        # No deleted seed or selector output may have an unrelated reader.
        for op in builder.ops:
            for kind,arg in zip(op.kinds,op.args):
                if kind=='imm' or kind.endswith('w'):continue
                if any(v in arg for v in old[1:]) and op is not selector:
                    assert arg==old,(index,op.tag,op.code,arg)
        removed.update(map(id,[*seedops,selector]))
        selected.append((index,old,source))
    created=[];mapping={};layouts=[]
    def emit(engine,code,args,time,tag):
        builder.put(engine,code,args,time,tag);created.append(builder.ops.pop())
    for index,old,source in selected:
        for lane in range(8):
            emit('store','store',[builder.bankptr[lane],source],12+lane//2,
                 ('constant_stage',(source[0],lane)))
        out=builder.new(8)
        emit('load','vload',[out,builder.bankptr[0]],16,('constant_reload',(source[0],)))
        mapping.update(zip(old,out));builder.cache[index]=out
        layouts.append(dict(index=index,source=source[0],vector=list(out)))
    kept=[op for op in builder.ops if id(op) not in removed]
    for op in kept:
        for k,(kind,arg) in enumerate(zip(op.kinds,op.args)):
            if kind!='imm' and not kind.endswith('w'):
                op.args[k]=tuple(mapping.get(v,v) for v in arg)
    insertion=next(i for i,op in enumerate(kept) if op.tag[0]=='hash_mad')
    builder.ops=kept[:insertion]+created+kept[insertion:]
    return layouts


def share_header_root_seed(builder,packs):
    """Use header lane 7 as the first seed of a root third-pack."""
    item=next((p for p in packs if p[0]=='root' and p[1]=='third'),None)
    if item is None:raise ValueError('share-root-seed needs a root third-pack')
    _,_,source,vector=item
    assert source==(builder.header[7],)
    writer={v:o for o in builder.ops if o.engine!='store' for v in o.args[0]}
    seed=vector[0];op=writer[seed]
    assert op.engine=='alu' and op.code=='|' and op.args[1]==source
    for current in builder.ops:
        if current is op:continue
        for k,(kind,arg) in enumerate(zip(current.kinds,current.args)):
            if kind!='imm' and not kind.endswith('w'):
                current.args[k]=tuple(source[0] if v==seed else v for v in arg)
    builder.ops=[current for current in builder.ops if current is not op]
    builder.rootraw=(source[0],)+vector[1:]
    return dict(removed_copy=seed,root=source[0],vector=list(builder.rootraw))


def build(full,chain,staged=(),thirds=(),fused=0,trades=0,init4097=None,head_root=0,
          prolog19=0,prolog_c1=0,cache_pairs=(),cache_stages=(),rederive_head=False,
          late_full=(),rederive_passes=1,rederive_ready=2,share_root_seed=False):
    if not 0<=fused<=32:raise ValueError('fused must be between 0 and 32')
    b=deepcopy(flow.original())
    if init4097=='load':flow.early_4097(b)
    elif init4097=='flow':scalar.flow_4097(b)
    elif init4097 is not None:raise ValueError('unknown 4097 initializer')
    converted=[scalar.scalar_uniform(b,name) for name in full]
    head_groups=early_scalar_root(b,head_root) if head_root else []
    staged_cache_layout=stage_third_caches(b,cache_stages) if cache_stages else []
    cache_layout=pack_staged_cache_pairs(b,cache_pairs) if cache_pairs else []
    flow.fuse_digits(b,list(range(fused)))
    layout=pack(b,chain) if chain else None
    frontier.logical_order(b)
    methods={key:'stage' for key in staged}
    if set(methods)&set(thirds):raise ValueError('duplicate uniform method')
    methods.update({key:'third' for key in thirds})
    packs=flow.materialize_uniforms(b,methods) if methods else []
    shared_root=share_header_root_seed(b,packs) if share_root_seed else None
    for key,_,source,out in packs:
        if key!='root':
            b.vectors[int(key)&0xffffffff]=out
            b.scalar[int(key)&0xffffffff]=source
    converted.extend(scalar.scalar_uniform(b,name) for name in late_full)
    if prolog19:
        staged19=next((p for p in packs if p[0]==19),None)
        if staged19:
            early19=early_scalar_19(b,prolog19,vector=staged19[3],source=staged19[2])
        elif chain and chain[-1]=='19':
            # The terminal seed survives the chain's FLOW writes.
            early19=early_scalar_19(b,prolog19)
        else:raise ValueError('prolog19 needs staged 19 or a terminal 19')
    else:early19=[]
    if prolog_c1:
        if not 0<=prolog_c1<=32:raise ValueError('prolog_c1 count must be 0..32')
        staged_c1=next((p for p in packs if p[0]==0xc761c23c),None)
        if staged_c1 is None:raise ValueError('prolog_c1 needs staged c1')
        converted.append(scalar.scalar_uniform(b,0xc761c23c,partial=True,
            source=staged_c1[2],vector=staged_c1[3],head_groups=prolog_c1))
    flow.trade_loads(b,trades,include_input=trades>=4)
    frontier.logical_order(b)
    b.sink=b.literals=b.literal_flow=0
    for _ in range(3):
        stage.prune_select_inputs(b)
        ir=research.clean(b)
    rederived=None
    if rederive_head:
        if not 1<=rederive_passes<=4 or not 1<=rederive_ready<=4:
            raise ValueError('rederive passes and source readiness must be 1..4')
        rederived=[]
        for step in range(rederive_passes):
            info=rederive_early_scalars(b,ir,extended=step>0,ready_limit=rederive_ready)
            rederived.append(info)
            frontier.logical_order(b)
            for _ in range(3):
                stage.prune_select_inputs(b)
                ir=research.clean(b)
            if not info['rewritten']:break
    return ir,b,dict(converted=converted,cache_pairs=cache_layout,cache_stages=staged_cache_layout,
                     head_root_groups=head_groups,rederived=rederived,shared_root=shared_root,
                     prolog19_groups=early19,chain=layout)


def measure(ir):
    counts=Counter(op.engine for op in ir.ops)
    work=counts['alu']+8*counts['valu']
    streams=stream_bounds(ir)
    expanded=expand(deepcopy(ir))
    expanded.views=[arg for _,operands,_,_ in expanded.groups for arg in operands]
    rebuild(expanded)
    head,tail=prefix(expanded),suffix(expanded)
    grouped={j for _,_,ids,_ in expanded.groups for j in ids}
    mandatory=sum(op.engine=='alu' and j not in grouped for j,op in enumerate(expanded.ops))
    lower=max((work+head['waste_lower_bound']+tail['waste_lower_bound']+59)//60,
              (mandatory+11)//12,*(v['cycle_lower_bound'] for v in streams.values()))
    return dict(counts=dict(counts),arithmetic_lanes=work,mandatory_alu=mandatory,
                prefix=head,suffix=tail,streams=streams,cycle_lower_bound=lower)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--full',default='cache0,0xb55a4f09')
    ap.add_argument('--chain',default='c1,19,9,3,2310')
    ap.add_argument('--staged',default='')
    ap.add_argument('--thirds',default='')
    ap.add_argument('--fused',type=int,default=1)
    ap.add_argument('--trades',type=int,default=0)
    ap.add_argument('--init4097',choices=('load','flow'))
    ap.add_argument('--head-root',type=int,default=0)
    ap.add_argument('--prolog19',type=int,default=0)
    ap.add_argument('--prolog-c1',type=int,default=0)
    ap.add_argument('--cache-pairs',default='',help='e.g. 3,4,5,6')
    ap.add_argument('--cache-stages',default='',help='e.g. 1,2')
    ap.add_argument('--rederive-head',action='store_true')
    ap.add_argument('--rederive-passes',type=int,default=1)
    ap.add_argument('--rederive-ready',type=int,default=2)
    ap.add_argument('--late-full',default='',help='uniforms scalarized after staging')
    ap.add_argument('--share-root-seed',action='store_true')
    ap.add_argument('--controls',action='store_true')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    cfg=dict(full=parse_values(args.full),chain=tuple(x for x in args.chain.split(',') if x),
             staged=parse_values(args.staged),thirds=parse_values(args.thirds),
             fused=args.fused,trades=args.trades,init4097=args.init4097,
             head_root=args.head_root,prolog19=args.prolog19,
             prolog_c1=args.prolog_c1,
             cache_pairs=tuple(int(x) for x in args.cache_pairs.split(',') if x),
             cache_stages=tuple(int(x) for x in args.cache_stages.split(',') if x),
             rederive_head=args.rederive_head,late_full=parse_values(args.late_full),
             rederive_passes=args.rederive_passes,rederive_ready=args.rederive_ready,
             share_root_seed=args.share_root_seed)
    ir,b,layout=build(**cfg)
    row=dict(config=cfg,layout=layout,**measure(ir),new_executable=False,
             scope='This explicit scalar-plus-chain graph. Bounds and serial replay are not scores.')
    args.output.write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps({k:row[k] for k in ('config','arithmetic_lanes','mandatory_alu',
                                         'cycle_lower_bound','prefix','streams')}),flush=True)
    if args.controls:
        control=flow.serial_control(deepcopy(ir),2)
        row['serial_control']=dict(scratch=control['scratch'],
            serialized_cycles=control['verified'],seeds=2)
        args.output.write_text(json.dumps(row,indent=2)+'\n')
        print(json.dumps({'serial_control':row['serial_control']}),flush=True)


if __name__=='__main__':main()
