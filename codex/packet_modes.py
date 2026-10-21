"""Export a word-SSA graph for native-vector / scalar-lane mode search."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from copy import deepcopy
import pickle
from pathlib import Path
from types import SimpleNamespace

from codex.research_packet_870 import capture,bilinear,verify
from codex.word_ssa import lift,fold,rebuild,condense,allocate,kinds,preselect_tail

ENGINES=('alu','valu','load','store','flow')


def expand(ir):
    old=ir.ops;oldrows=ir.rows;ops=[];parts=[];groups=[];views=[];source_order=[]
    oldmem=ir.memory_events;events={}
    for j,op in enumerate(old):
        parts.append([])
        if op.engine=='valu' and op.code not in ('multiply_add','vbroadcast'):
            ids=[]
            for lane in range(8):
                ids.append(len(ops));parts[-1].append(len(ops))
                args=[(a[lane],) for a in op.args]
                ops.append(SimpleNamespace(engine='alu',code=op.code,args=args,kinds=kinds('alu',op.code),tag=op.tag,time=op.time))
                source_order.append((oldrows[j][0] if j<len(oldrows) else op.time)*16+lane)
            groups.append((op.code,op.args,ids,1));views.extend(op.args)
        else:
            parts[-1].append(len(ops))
            if j in oldmem:events[len(ops)]=oldmem[j]
            ops.append(op)
            source_order.append((oldrows[j][0] if j<len(oldrows) else op.time)*16)
    ir.ops=ops;ir.memory_events=events
    # graph(..., event_override=...) needs only engine and slot arity.
    rows=[]
    for index,op in enumerate(ops):
        slot=(op.code,*(0 for _ in op.kinds))
        # Source execution order defines memory epochs. Scheduling priorities
        # can intentionally put a gather earlier than its table-build hint.
        rows.append((source_order[index],op.engine,slot,op.tag))
    ir.rows=rows;ir.views=views
    rebuild(ir)
    # Additional original ALU vectors can be merged only when all operands
    # already have the corresponding contiguous views.
    covered={j for code,args,ids,mode in groups for j in ids}
    signatures=defaultdict(dict)
    for j,op in enumerate(ops):
        if j in covered or op.engine!='alu':continue
        d,a,b=(ir.uf.find(arg[0]) for arg in op.args)
        key=(op.code,d[0],a[0],b[0],a[1]-d[1],b[1]-d[1])
        signatures[key][d[1]]=j
    for key,values in signatures.items():
        for start in sorted(values):
            ids=[values.get(start+l) for l in range(8)]
            if any(j is None or j in covered for j in ids):continue
            args=[tuple(ops[j].args[k][0] for j in ids) for k in range(3)]
            groups.append((key[0],args,ids,0));covered.update(ids)
    ir.groups=groups
    return ir


def materialize(ir,times,modes):
    covered={j:g for g,(code,args,ids,mode) in enumerate(ir.groups) if modes[g] for j in ids}
    keep=[]
    for j,op in enumerate(ir.ops):
        if j in covered:
            group=covered[j];code,args,ids,_=ir.groups[group]
            if j!=min(ids):continue
            assert len({times[i] for i in ids})==1
            keep.append((times[j],SimpleNamespace(engine='valu',code=code,args=args,kinds=kinds('valu',code),tag=op.tag,time=times[j])))
        else:keep.append((times[j],op))
    # Rebuild only the view/coloring model; no memory/data edges are needed to
    # audit a supplied already validated calendar.
    result=SimpleNamespace(origin=ir.origin,known=ir.known,undef=getattr(ir,'undef',set()),ops=[op for t,op in keep])
    from codex.word_ssa import Offsets,observed
    uf=Offsets()
    for _ in ir.origin:uf.add()
    writer={};uses=defaultdict(set);needed=observed(result)
    for j,op in enumerate(result.ops):
        for k,arg in zip(op.kinds,op.args):
            if k=='imm':continue
            for lane,v in enumerate(arg):uf.join(arg[0],v,lane)
            if k.endswith('w'):
                for v in arg:writer[v]=j
            else:
                for v in arg:
                    if v in needed:uses[v].add(j)
    result.uf=uf;result.writer=writer;result.uses=uses
    return allocate(result,[t for t,op in keep])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('export','check'))
    ap.add_argument('--prefix',type=Path,default=Path('/tmp/aopt-modes'))
    ap.add_argument('--bilinear',action='store_true');ap.add_argument('--fold',type=int,default=0)
    ap.add_argument('--preselect-tail',action='store_true')
    ap.add_argument('--ir',type=Path)
    args=ap.parse_args();prefix=str(args.prefix)
    if args.mode=='export':
        if args.ir:
            with args.ir.open('rb') as f:ir=pickle.load(f)
        else:
            rows,emit=capture()
            if args.bilinear:rows=bilinear(rows,emit)
            ir=lift(rows)
        if args.fold:fold(ir,args.fold)
        if args.preselect_tail:preselect_tail(ir)
        ir=expand(ir)
        # Native merging also needs these complete operand views in the IR.
        ir.views=[a for code,aa,ids,md in ir.groups for a in aa]
        rebuild(ir);g=condense(ir)
        assert g.n==len(ir.ops),('nontrivial coissue components',g.n,len(ir.ops))
        edges=[(d,j,lag) for j,dd in enumerate(ir.deps) for d,lag in dd.items()]
        with open(prefix+'.txt','w') as f:
            f.write(f'{len(ir.ops)} {len(edges)} {len(ir.groups)}\n')
            for op in ir.ops:f.write(f'{ENGINES.index(op.engine)} {op.time}\n')
            for d,j,lag in edges:f.write(f'{d} {j} {lag}\n')
            for code,aa,ids,md in ir.groups:f.write(str(md)+' '+' '.join(map(str,ids))+'\n')
        with open(prefix+'.pkl','wb') as f:pickle.dump(ir,f)
        print({'nodes':len(ir.ops),'edges':len(edges),'groups':len(ir.groups)},flush=True)
    else:
        with open(prefix+'.pkl','rb') as f:ir=pickle.load(f)
        values=list(map(int,Path(prefix+'.sol').read_text().split()));n,ng=values[:2]
        assert (n,ng)==(len(ir.ops),len(ir.groups))
        times=values[2:2+n];modes=values[2+n:]
        for j,dd in enumerate(ir.deps):
            for d,lag in dd.items():assert times[j]>=times[d]+lag
        program,words,shapes=materialize(ir,times,modes)
        assert words<=1536,words
        cycles=verify(program,20)
        import json
        Path(prefix+'.json').write_text(json.dumps(program))
        print({'cycles':cycles,'scratch':words,'verified':20},flush=True)


if __name__=='__main__':main()
