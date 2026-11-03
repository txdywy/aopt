"""Build longer overlapping uniform chains with real FLOW and ALU instructions."""
from collections import Counter
from copy import deepcopy
import argparse
import json
from pathlib import Path

from codex import research_flow_stage863 as flow
from codex import research_frontier863 as frontier
from codex import research_stage864 as stage
from codex import research_sub864 as research
from codex.bound_boundary_no_solver863 import prefix,suffix
from codex.packet_budget_bound import stream_bounds
from codex.packet_modes import expand
from codex.word_ssa import rebuild

NAMES={'c0':0x7ed55d16,'c1':0xc761c23c,'2':2,'3':3,'9':9,'19':19,
       '16':16,'256':256,'2310':2310,'4097':4097,'-2':0xfffffffe}


def pack(builder, names):
    if len(names)<2 or len(names)!=len(set(names)) or any(n!='root' and n not in NAMES for n in names):
        raise ValueError('distinct supported uniform names are required')
    vectors=[builder.rootraw if n=='root' else builder.vectors[NAMES[n]] for n in names]
    writer={v:op for op in builder.ops if op.engine!='store' for v in op.args[0]}
    producers=[writer[v[0]] for v in vectors]
    assert all(op.engine=='valu' and op.args[0]==vec for op,vec in zip(producers,vectors))
    created=[]
    def emit(engine,code,args,time,tag):
        builder.put(engine,code,args,time,tag);created.append(builder.ops.pop())
    roots={}
    for name,op in zip(names,producers):
        if name in ('root','c0','c1','16','256','2310','4097'):
            assert op.code=='vbroadcast';roots[name]=op.args[1]
    if '2' in names:
        roots['2']=builder.new()
        emit('alu','//',[roots['2'],builder.scalar[16],builder.scalar[7]],1,('chain_root',('2',)))
    if '3' in names:
        roots['3']=builder.new()
        emit('alu','cdiv',[roots['3'],builder.scalar[16],builder.scalar[7]],1,('chain_root',('3',)))
    if '9' in names:
        roots['9']=builder.new()
        emit('alu','-',[roots['9'],builder.scalar[10],builder.scalar[1]],1,('chain_root',('9',)))
    if '19' in names:
        roots['19']=builder.new()
        emit('alu','+',[roots['19'],builder.scalar[16],roots.get('3',builder.scalar[3])],2,('chain_root',('19',)))
    if '-2' in names:
        roots['-2']=builder.new()
        emit('alu','-',[roots['-2'],builder.scalar[0],roots.get('2',builder.scalar[2])],2,('chain_root',('-2',)))
    # Header lanes before the root also have live following header lanes.
    # Give each selected header scalar an ordinary detached definition.
    for name in ('16','256','2310'):
        if name in names:
            detached=builder.new()
            emit('alu','|',[detached,roots[name],(builder.zero[0],)],2,('chain_detach_header',(name,)))
            roots[name]=detached
    if 'root' in names and names[0]!='root':
        detached=builder.new()
        emit('alu','|',[detached,roots['root'],(builder.zero[0],)],2,('chain_detach_root',()))
        roots['root']=detached
    def copies(root,count,label):
        words=list(root)
        for i in range(count):
            dest=builder.new()
            emit('alu','|',[dest,root,(builder.zero[0],)],2,('chain_seed',(label,i)))
            words.extend(dest)
        return tuple(words)
    previous=copies(roots[names[0]],3,names[0]);newvectors=[]
    for rank,name in enumerate(names[1:]):
        out=builder.new(8)
        if rank==len(names)-2:
            terminal=copies(roots[name],3,name)
            cond=builder.vectors[1][4:]+builder.zero[:4]
            yes=previous+builder.new(4,undefined=True)
            no=builder.new(4,undefined=True)+terminal
        else:
            seed=copies(roots[name],1,name)
            cond=builder.vectors[1][2:]+builder.zero[:2]
            yes=previous+seed+builder.new(2,undefined=True)
            no=builder.new(2,undefined=True)+previous+seed
        emit('flow','vselect',[out,cond,yes,no],6+rank,('chain_pack',(rank,)))
        newvectors.append(previous+out[:4]);previous=out[4:]
    newvectors.append(previous+terminal)
    mapping={a:b for old,new in zip(vectors,newvectors) for a,b in zip(old,new)}
    kept=[op for op in builder.ops if all(op is not p for p in producers)]
    for op in kept:
        for i,(kind,arg) in enumerate(zip(op.kinds,op.args)):
            if kind!='imm' and not kind.endswith('w'):
                op.args[i]=tuple(mapping.get(v,v) for v in arg)
    insertion=next(i for i,op in enumerate(kept) if op.tag[0]=='hash_mad')
    builder.ops=kept[:insertion]+created+kept[insertion:]
    for name,vector in zip(names,newvectors):
        if name=='root':builder.rootraw=vector
        else:builder.vectors[NAMES[name]]=vector;builder.scalar[NAMES[name]]=(vector[0],)
    builder.stats['uniform_chain_vectors']=len(names)
    return dict(names=list(names),vectors=[list(v) for v in newvectors],
                root_detached='root' in names and names[0]!='root')


def build(names,fused=0,staged=(),trades=0,load4097=False):
    if not 0<=fused<=32:raise ValueError('fused must be between 0 and 32')
    b=deepcopy(flow.original())
    if load4097:flow.early_4097(b)
    flow.fuse_digits(b,list(range(fused)))
    layout=pack(b,tuple(names));frontier.logical_order(b)
    if staged:flow.materialize_uniforms(b,{value:'stage' for value in staged})
    flow.trade_loads(b,trades,include_input=trades>=4)
    frontier.logical_order(b)
    b.sink=b.literals=b.literal_flow=0
    for _ in range(3):
        stage.prune_select_inputs(b);ir=research.clean(b)
    return ir,b,layout


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--names',default='root,c0,c1,9,19')
    ap.add_argument('--fused',type=int,default=1)
    ap.add_argument('--staged',default='')
    ap.add_argument('--trades',type=int,default=0)
    ap.add_argument('--load4097',action='store_true')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--controls',action='store_true')
    args=ap.parse_args()
    names=tuple(args.names.split(','));staged=tuple(int(x,0) for x in args.staged.split(',') if x)
    ir,b,layout=build(names,args.fused,staged,args.trades,args.load4097)
    counts=Counter(op.engine for op in ir.ops);work=counts['alu']+8*counts['valu']
    streams=stream_bounds(ir);expanded=expand(deepcopy(ir))
    expanded.views=[a for _,operands,_,_ in expanded.groups for a in operands]
    rebuild(expanded);head,tail=prefix(expanded),suffix(expanded)
    lower=max((work+head['waste_lower_bound']+tail['waste_lower_bound']+59)//60,
              *(v['cycle_lower_bound'] for v in streams.values()))
    result=dict(config=dict(names=names,fused=args.fused,staged=staged,trades=args.trades,load4097=args.load4097),
                layout=layout,counts=dict(counts),arithmetic_lanes=work,prefix=head,suffix=tail,
                streams=streams,cycle_lower_bound=lower,new_executable=False,
                scope='This concrete chain; necessary bounds and serial replay are not scores.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('arithmetic_lanes','cycle_lower_bound','prefix','streams')}),flush=True)
    if args.controls:
        control=flow.serial_control(deepcopy(ir),2)
        result['serial_control']=dict(scratch=control['scratch'],
            serialized_cycles=control['verified'],seeds=2)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'serial_control':result['serial_control']}),flush=True)


if __name__=='__main__':main()
