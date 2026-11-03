"""Word-level SSA and contiguous-view analysis for packet-kernel research."""
from __future__ import annotations

from collections import Counter,defaultdict
from types import SimpleNamespace

from codex.research_packet_870 import capture,graph,schedule,verify,LIMITS,analyze,bilinear


class Offsets:
    def __init__(self):self.parent=[];self.delta=[]
    def add(self):
        j=len(self.parent);self.parent.append(j);self.delta.append(0);return j
    def find(self,j):
        if self.parent[j]!=j:
            root,off=self.find(self.parent[j]);self.delta[j]+=off;self.parent[j]=root
        return self.parent[j],self.delta[j]
    def join(self,a,b,d):
        # position(b) - position(a) = d
        ra,oa=self.find(a);rb,ob=self.find(b)
        if ra==rb:assert ob-oa==d
        else:self.parent[rb]=ra;self.delta[rb]=d+oa-ob


def kinds(engine,op):
    if engine=='alu':return ('w','r','r')
    if engine=='valu':
        if op=='vbroadcast':return ('vw','r')
        return ('vw','vr','vr','vr') if op=='multiply_add' else ('vw','vr','vr')
    if engine=='load':return ('w','imm') if op=='const' else ('vw','r') if op=='vload' else ('w','r')
    if engine=='store':return ('r','vr' if op=='vstore' else 'r')
    if engine=='flow':
        if op=='add_imm':return ('w','r','imm')
        return ('vw','vr','vr','vr') if op=='vselect' else ('w','r','r','r')
    raise ValueError((engine,op))


def lift(rows):
    uf=Offsets();origin=[];writer={};uses=defaultdict(set)
    def new(old):
        j=uf.add();origin.append(old);return j
    current=[new(i) for i in range(1536)];ops=[];known={i:0 for i in range(1536)}
    definitions={};analyze(rows,definitions)
    bytime=defaultdict(list)
    for j,row in enumerate(rows):bytime[row[0]].append(j)
    for t,ids in sorted(bytime.items()):
        pending={}
        for j in ids:
            _,engine,slot,tag=rows[j];code,*a=slot
            ks=kinds(engine,code);args=[];reads=[];writes=[]
            for k,p in zip(ks,a,strict=True):
                if k=='imm':args.append(p);continue
                width=8 if k.startswith('v') else 1
                if k.endswith('w'):
                    value=tuple(new(p+l) for l in range(width))
                    for l,v in enumerate(value):
                        writer[v]=j;pending[p+l]=v
                        abstract=definitions[j].get(p+l)
                        if abstract is not None and len(abstract)==1:known[v]=next(iter(abstract))
                    writes.extend(value)
                else:
                    value=tuple(current[p+l] for l in range(width));reads.extend(value)
                    for v in value:uses[v].add(j)
                for l,v in enumerate(value):uf.join(value[0],v,l)
                args.append(value)
            ops.append(SimpleNamespace(engine=engine,code=code,args=args,kinds=ks,reads=reads,writes=writes,tag=tag,time=t))
        for p,v in pending.items():current[p]=v
    assert len(ops)==len(rows)
    mem=graph(rows,scratch=False,condense=False)
    deps=[dict(d) for d in mem.deps]
    def edge(d,j,lag):
        if d!=j:deps[j][d]=max(deps[j].get(d,0),lag)
    for j,op in enumerate(ops):
        for v in op.reads:
            if v in writer:edge(writer[v],j,1)
    # Only values connected by an ISA contiguous view have to reuse positions.
    cells=defaultdict(list)
    for v in set(writer)|set(uses):cells[uf.find(v)].append(v)
    for values in cells.values():
        values.sort(key=lambda v:rows[writer[v]][0] if v in writer else -1)
        for old,newv in zip(values,values[1:]):
            if newv not in writer:continue
            q=writer[newv]
            if old in writer:edge(writer[old],q,1)
            for reader in uses[old]:edge(reader,q,0)
    succ=[[] for _ in ops]
    for j,dd in enumerate(deps):
        for d,lag in dd.items():succ[d].append((j,lag))
    return SimpleNamespace(uf=uf,ops=ops,writer=writer,uses=uses,origin=origin,known=known,
                           deps=deps,succ=succ,rows=rows,explicit=mem.deps,memory_events=analyze(rows))


def observed(ir):
    writer={v:j for j,op in enumerate(ir.ops) for v in op.args[0]
            if op.engine not in ('store',) and op.kinds[0].endswith('w')}
    needed=set();pending=[]
    def need(v):
        if v in getattr(ir,'undef',()):return
        if v not in needed:needed.add(v);pending.append(v)
    for op in ir.ops:
        if op.engine=='store':
            for a in op.args:
                for v in a:need(v)
        if op.engine=='load' and op.code!='const':
            for v in op.args[1]:need(v)
        if op.code=='<<':
            for a in op.args[1:]:
                for v in a:need(v)
    while pending:
        v=pending.pop()
        if v not in writer:continue
        op=ir.ops[writer[v]];lane=op.args[0].index(v)
        if op.engine=='load':continue
        if op.code=='vbroadcast':need(op.args[1][0]);continue
        if op.code in ('vselect','select'):
            cond=op.args[1][lane];need(cond);value=ir.known.get(cond)
            for index in ((3,) if value==0 else (2,) if value is not None else (2,3)):
                need(op.args[index][lane])
            continue
        for kind,arg in zip(op.kinds[1:],op.args[1:]):
            if kind!='imm':need(arg[lane if kind.startswith('v') else 0])
    return needed


def rebuild(ir):
    uf=Offsets()
    for _ in ir.origin:uf.add()
    for view in getattr(ir,'views',[]):
        for lane,v in enumerate(view):uf.join(view[0],v,lane)
    writer={};uses=defaultdict(set);needed=observed(ir)
    for j,op in enumerate(ir.ops):
        op.reads=[];op.writes=[]
        for kind,arg in zip(op.kinds,op.args):
            if kind=='imm':continue
            for lane,v in enumerate(arg):uf.join(arg[0],v,lane)
            if kind.endswith('w'):
                op.writes.extend(arg)
                for v in arg:assert v not in writer;writer[v]=j
            else:
                op.reads.extend(v for v in arg if v in needed)
                for v in arg:
                    if v in needed:uses[v].add(j)
    # A gather's unused padding still executes, but its loaded value cannot
    # create a dependency on a store. The pointer remains fully observed.
    events={}
    for j,(engine,bases,locations) in ir.memory_events.items():
        if engine=='load':
            live=[lane for lane,v in enumerate(ir.ops[j].args[0]) if v in needed]
            locations={b+lane for b in bases for lane in live}
        events[j]=(engine,bases,locations)
    mem=graph(ir.rows,scratch=False,condense=False,event_override=events)
    deps=[dict(d) for d in mem.deps]
    while len(deps)<len(ir.ops):deps.append({})
    def edge(d,j,lag):
        if d!=j:deps[j][d]=max(deps[j].get(d,0),lag)
    for j,op in enumerate(ir.ops):
        for v in op.reads:
            if v in writer:edge(writer[v],j,1)
    cells=defaultdict(list)
    for v in set(writer)|set(uses):cells[uf.find(v)].append(v)
    for values in cells.values():
        values.sort(key=lambda v:ir.ops[writer[v]].time if v in writer else -1)
        for old,newv in zip(values,values[1:]):
            if newv not in writer:continue
            q=writer[newv]
            if old in writer:edge(writer[old],q,1)
            for reader in uses[old]:edge(reader,q,0)
    succ=[[] for _ in ir.ops]
    for j,dd in enumerate(deps):
        for d,lag in dd.items():succ[d].append((j,lag))
    ir.uf,ir.writer,ir.uses,ir.deps,ir.succ=uf,writer,uses,deps,succ
    return ir


def fold(ir,count=27):
    uniform={}
    for op in ir.ops:
        for kind,arg in zip(op.kinds,op.args):
            if kind!='imm' and len(arg)==8 and all(v in ir.known for v in arg):
                vals={ir.known[v] for v in arg}
                if len(vals)==1:uniform.setdefault(next(iter(vals)),arg)
    stages={};choices={}
    for j,op in enumerate(ir.ops):
        name,coord=op.tag
        if name in ('boundary_even','boundary_parent_offset'):stages[coord[0]]=j
        if name in ('boundary_offset','packet_boundary_choice'):choices[coord[0]]=j
    def fresh():
        out=tuple(range(len(ir.origin),len(ir.origin)+8));ir.origin.extend([-1]*8);return out
    for batch in sorted(choices,key=lambda b:ir.ops[choices[b]].time,reverse=True)[:count]:
        i,j=stages[batch],choices[batch];first,second=ir.ops[i],ir.ops[j]
        assert second.code=='multiply_add'
        if first.tag[0]=='boundary_even':b4,b3=first.args[1],second.args[1]
        else:b3,b4=first.args[1],second.args[1]
        even,odd=fresh(),fresh()
        first.args=[even,b3,uniform[2863311502],uniform[2863311488]]
        ir.ops.append(SimpleNamespace(engine='flow',code='vselect',args=[odd,b3,uniform[2863311509],uniform[2863311495]],
                                      kinds=kinds('flow','vselect'),tag=('ssa_odd',(batch,)),time=first.time))
        second.engine='flow';second.code='vselect';second.kinds=kinds('flow','vselect')
        second.args=[second.args[0],b4,odd,even]
    return rebuild(ir)


def preselect_tail(ir):
    """Distribute the root XOR before the last eight scalar root selects."""
    roots={};final=None
    writer={v:j for j,op in enumerate(ir.ops) for v in op.args[0]
            if op.engine!='store' and op.kinds[0].endswith('w')}
    for j,op in enumerate(ir.ops):
        name,coord=op.tag
        if name=='sibling_selected_root':roots[coord[0]]=j
        if name=='packet_xor_6':final=j
    assert len(roots)==8 and final is not None
    out=ir.ops[final];state=out.args[1];left=out.args[2];cond=[]
    right=tuple(range(len(ir.origin),len(ir.origin)+8));ir.origin.extend([-1]*8)
    for lane in range(8):
        op=ir.ops[roots[lane]];dest,b,a,c=op.args
        assert dest==(left[lane],)
        cond.append(b[0])
        ready=max(ir.ops[writer[state[lane]]].time,ir.ops[writer[a[0]]].time,ir.ops[writer[c[0]]].time)+1
        op.engine='alu';op.code='^';op.kinds=kinds('alu','^');op.args=[dest,(state[lane],),a];op.time=ready
        ir.ops.append(SimpleNamespace(engine='alu',code='^',args=[(right[lane],),(state[lane],),c],
                                      kinds=kinds('alu','^'),tag=('tail_other',(lane,)),time=ready))
    out.engine='flow';out.code='vselect';out.kinds=kinds('flow','vselect')
    out.args=[out.args[0],tuple(cond),left,right]
    out.time=max(ir.ops[roots[lane]].time for lane in range(8))+1
    return rebuild(ir)


def condense(ir):
    deps,succ=ir.deps,ir.succ;n=len(ir.ops);seen=set();post=[]
    import sys
    sys.setrecursionlimit(100000)
    def dfs(j):
        seen.add(j)
        for q,lag in succ[j]:
            if q not in seen:dfs(q)
        post.append(j)
    for j in range(n):
        if j not in seen:dfs(j)
    owner={};clusters=[]
    def back(j,c):
        owner[j]=c;clusters[c].append(j)
        for d in deps[j]:
            if d not in owner:back(d,c)
    for j in reversed(post):
        if j not in owner:clusters.append([]);back(j,len(clusters)-1)
    dd=[{} for _ in clusters]
    for j,ds in enumerate(deps):
        for d,lag in ds.items():
            a,b=owner[d],owner[j]
            if a==b:assert lag==0,('positive cycle',ir.ops[j].tag,ir.ops[d].tag)
            else:dd[b][a]=max(dd[b].get(a,0),lag)
    ss=[[] for _ in clusters]
    for j,ds in enumerate(dd):
        for d,lag in ds.items():ss[d].append((j,lag))
    demands=[Counter(ir.ops[j].engine for j in ids) for ids in clusters]
    for i,demand in enumerate(demands):
        assert all(demand[e]<=LIMITS[e] for e in demand),('impossible cluster',demand,[ir.ops[j].tag for j in clusters[i]])
    return SimpleNamespace(n=len(clusters),deps=dd,succ=ss,clusters=clusters,demands=demands,
                           original=[max(ir.ops[j].time for j in ids) for ids in clusters],owner=owner)


def allocate(ir,times):
    uf=ir.uf;roots=defaultdict(list)
    active=set(ir.writer)|set(ir.uses)
    for op in ir.ops:
        for kind,arg in zip(op.kinds,op.args):
            if kind!='imm':active.update(arg)
    for v in active:
        r,o=uf.find(v);roots[r].append((v,o))
    shapes={};offsets={};births={};areas={}
    for r,words in roots.items():
        low=min(o for v,o in words);high=max(o for v,o in words)
        shape=[0]*(high-low+1);born=9999
        for v,off in words:
            if v not in ir.writer and v not in ir.uses:continue
            first=2*times[ir.writer[v]]+1 if v in ir.writer else 0
            last=max([2*times[j] for j in ir.uses[v]] or [first])
            last=max(first,last);born=min(born,first)
            bits=((1<<(last-first+1))-1)<<first
            assert not shape[off-low]&bits,('self conflict',r,off,v)
            shape[off-low]|=bits
        shapes[r]=shape;offsets[r]=low;births[r]=born;areas[r]=sum(m.bit_count() for m in shape)
    mask=[];physical={}
    for r in sorted(shapes,key=lambda r:(-areas[r],-len(shapes[r]),births[r])):
        shape=shapes[r];base=0
        while not all(not mask[base+l]&bits for l,bits in enumerate(shape) if base+l<len(mask)):base+=1
        if base+len(shape)>len(mask):mask.extend([0]*(base+len(shape)-len(mask)))
        physical[r]=base
        for l,bits in enumerate(shape):mask[base+l]|=bits
    def address(v):
        r,o=uf.find(v);return physical[r]+o-offsets[r]
    p=[{} for _ in range(max(times)+1)]
    for j,op in enumerate(ir.ops):
        a=[]
        for k,arg in zip(op.kinds,op.args):
            if k=='imm':a.append(arg)
            else:
                addresses=[address(v) for v in arg]
                assert addresses==list(range(addresses[0],addresses[0]+len(arg)))
                a.append(addresses[0])
        p[times[j]].setdefault(op.engine,[]).append((op.code,*a))
    return p,len(mask),Counter(map(len,shapes.values()))


def main():
    import argparse,json
    from pathlib import Path
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bilinear',action='store_true');ap.add_argument('--fold',type=int,default=0)
    ap.add_argument('--passes',type=int,default=0);ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    rows,emit=capture()
    if args.bilinear:rows=bilinear(rows,emit)
    ir=lift(rows)
    if args.fold:ir=fold(ir,args.fold)
    if args.passes:
        g=condense(ir);resource=[(op.time,op.engine,(),op.tag) for op in ir.ops];best=None
        counts=Counter(op.engine for op in ir.ops)
        print({'counts':dict(counts),'lane_work':counts['alu']+8*counts['valu'],'clusters':g.n},flush=True)
        for iteration in range(args.passes):
            _,cluster_times=schedule(resource,g,reverse=bool(iteration%2))
            g.original=cluster_times
            times=[cluster_times[g.owner[j]] for j in range(len(ir.ops))]
            if best is None or max(times)+1<best:
                try:
                    p,words,shapes=allocate(ir,times)
                    assert words<=1536,words
                    cycles=verify(p,5)
                except AssertionError as error:
                    print({'pass':iteration,'cycles':max(times)+1,'rejected':str(error)[:180]},flush=True);continue
                best=cycles
                print({'pass':iteration,'cycles':cycles,'scratch':words,'verified':5},flush=True)
                if args.output:args.output.write_text(json.dumps(p))
        return
    times=[row[0] for row in rows]
    program,words,shapes=allocate(ir,times)
    print({'ops':len(ir.ops),'words':words,'shapes':dict(shapes)},flush=True)
    assert words<=1536
    print({'verified_cycles':verify(program,20)},flush=True)


if __name__=='__main__':main()
