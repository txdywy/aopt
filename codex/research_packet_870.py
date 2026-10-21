"""Offline experiments on the verified 880-cycle kernel; never edits submission.

Uses the original ISA, including all eight writes of vector instructions.
The bilinear end-pointer construction is from Josu San Martin's public log.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import heapq
import itertools
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from codex import perf_takehome_880 as source
from codex.verify_compact import scratch_accesses
from frozen_problem import Machine, Tree, Input, build_mem_image, reference_kernel2

LIMITS = dict(alu=12, valu=6, load=2, store=2, flow=1)
MASK = (1 << 32) - 1


def capture():
    """Retain the emitter's semantic coordinates for every original slot."""
    tags = []
    current = None
    rows = source._issue_rows

    def tagged_rows(name):
        nonlocal current
        for row in rows(name):
            current = (name, row[:-1])
            yield row

    class Emitter(source._Emitter):
        def put(self, cycle, engine, operation, *operands):
            tags.append((cycle, engine, (operation, *operands), current))
            super().put(cycle, engine, operation, *operands)

    source._issue_rows = tagged_rows
    try:
        emit = Emitter()
        source._emit_final_computations(emit)
    finally:
        source._issue_rows = rows
    tags.sort(key=lambda row: row[0])
    return tags, emit


def calc(op, args):
    if any(a is None for a in args):
        if op in ("&", "|") and len(args) == 2:
            known = args[1] if args[0] is None else args[0]
            if known is not None:
                masks = [m if op == "&" else m ^ MASK for m in known]
                if max(m.bit_count() for m in masks) <= 10:
                    out = set()
                    for base, bits in zip(known, masks):
                        sub = bits
                        while True:
                            out.add(sub if op == "&" else base | sub)
                            if sub == 0:
                                break
                            sub = (sub - 1) & bits
                    return frozenset(out)
        return None
    if __import__("math").prod(map(len, args)) > 200000:
        return None
    def apply(a):
        if op == "+": return a[0] + a[1]
        if op == "-": return a[0] - a[1]
        if op == "*": return a[0] * a[1]
        if op == "//": return a[0] // a[1]
        if op == "%": return a[0] % a[1]
        if op == "cdiv": return (a[0] + a[1] - 1) // a[1]
        if op == "<": return int(a[0] < a[1])
        if op == "^": return a[0] ^ a[1]
        if op == "&": return a[0] & a[1]
        if op == "|": return a[0] | a[1]
        if op == ">>": return a[0] >> a[1]
        if op == "<<": return a[0] << a[1]
        if op == "==": return int(a[0] == a[1])
        if op == "multiply_add": return a[0] * a[1] + a[2]
        raise ValueError(op)
    return frozenset(apply(a) & MASK for a in itertools.product(*args))


def join(*values):
    return None if any(v is None for v in values) else frozenset().union(*values)


def analyze(rows,definitions=None):
    """Prove possible addresses from ISA constants, never from runtime input."""
    values = [frozenset([0])] * 1536
    memory = {i: frozenset([v]) for i, v in enumerate((16, 2047, 256, 10, 7, 2054, 2310))}
    memory.update({i: frozenset([0]) for i in range(2054, 2310)})
    events = {}
    groups = defaultdict(list)
    for j, row in enumerate(rows): groups[row[0]].append(j)
    for t, ids in sorted(groups.items()):
        writes, stores = {}, {}
        for j in ids:
            _, engine, s, _ = rows[j]
            op, *a = s
            if engine in ("alu", "valu"):
                for lane in range(8 if engine == "valu" else 1):
                    writes[a[0] + lane] = values[a[1]] if op == "vbroadcast" else calc(op, [values[p + lane] for p in a[1:]])
            elif engine == "flow":
                if op == "add_imm":
                    writes[a[0]] = calc("+", [values[a[1]], frozenset([a[2] & MASK])])
                else:
                    assert op in ("select", "vselect")
                    for lane in range(8 if op == "vselect" else 1):
                        cond, yes, no = (values[p + lane] for p in a[1:])
                        writes[a[0] + lane] = no if cond == frozenset([0]) else yes if cond is not None and 0 not in cond else join(yes, no)
            elif engine == "load" and op == "const":
                writes[a[0]] = frozenset([a[1] & MASK])
            else:
                ptr = a[1] + (a[2] if op == "load_offset" else 0) if engine == "load" else a[0]
                bases = values[ptr]
                assert bases, ("unproved address", j, t, engine, s)
                width = 8 if op in ("vload", "vstore") else 1
                locations = {b + lane for b in bases for lane in range(width)}
                assert all(0 <= x < 2566 for x in locations), (j, locations)
                events[j] = (engine, bases, locations)
                if engine == "load":
                    for lane in range(width):
                        dest = a[0] + (a[2] if op == "load_offset" else lane)
                        writes[dest] = join(*(memory.get(b + lane) for b in bases))
                else:
                    for b in bases:
                        for lane in range(width):
                            addr = b + lane
                            value = values[a[1] + lane]
                            if len(bases) > 1:
                                value = join(stores.get(addr, memory.get(addr)), value)
                            stores[addr] = value
            if definitions is not None:
                definitions[j]={a:writes[a] for a in scratch_accesses(engine,s)[1] if a in writes}
        for address, value in writes.items(): values[address] = value
        memory.update(stores)
    return events


def bilinear(rows, emit, groups=31):
    """A14=select(b11,16,3)*(h13|~1)+select(b12,2278,2272)."""
    rows = list(rows)
    events = analyze(rows)
    image = 1437
    mask, w0, w1 = image + 32, image + 40, image + 48
    ptr0, ptr1 = image + 56, image + 57
    t32, t6, t38, t26 = image + 58, image + 59, image + 60, image + 61
    pointers = list(range(image + 62, image + 70))
    assert max(pointers) < 1536
    R = emit.reg
    def add(t, e, *s): rows.append((t, e, tuple(s), ("new", ())))
    # Preserve the existing three tree-image producer groups in fresh scratch.
    # No additional LOAD or copies are needed for the private table.
    stores = [(j, next(iter(ev[1]))) for j, ev in events.items()
              if ev[0] == "store" and len(ev[1]) == 1 and next(iter(ev[1])) in (8, 16, 24)
              and rows[j][2][0] == "vstore"]
    assert len(stores) == 3, stores
    for store, address in stores:
        cycle, engine, slot, tag = rows[store]
        old = slot[2]
        for lane in range(8):
            defs = [j for j, row in enumerate(rows[:store]) if row[0] < cycle
                    and old + lane in scratch_accesses(row[1], row[2])[1]]
            producer = defs[-1]
            pt, pe, ps, ptag = rows[producer]
            assert pe == "alu" and ps[0] == "^", (producer, rows[producer])
            # This value's sole reader before its next overwrite is the vstore.
            assert not any(old + lane in scratch_accesses(row[1], row[2])[0]
                           for row in rows[producer + 1:store])
            rows[producer] = pt, pe, (ps[0], image + address - 8 + lane, *ps[2:]), ptag
        rows[store] = cycle, engine, (slot[0], slot[1], image + address - 8), tag
    header = R("header")
    add(1, "alu", "+", t32, header, header)
    add(1, "alu", "-", t6, header, header + 3)
    add(1, "alu", "+", t26, header, header + 3)
    add(2, "alu", "+", t38, t32, t6)
    add(3, "alu", "-", ptr0, header + 6, t38)
    add(3, "alu", "-", ptr1, header + 6, t32)
    add(102, "valu", "-", mask, R("constant_0"), R("constant_2"))
    add(123, "valu", "vbroadcast", w0, ptr0)
    add(156, "valu", "vbroadcast", w1, ptr1)
    addresses = (2266, 2269, 2272, 2275, 2240, 2256, 2246, 2262)
    offsets = {2240:t32, 2246:t26, 2256:header, 2262:header+3, 2266:t6, 2269:R("constant_3"), 2272:None, 2275:R("constant_3")}
    # Capture header 16/10 into permanent words before the original header dies.
    t16, t10 = image + 70, image + 71
    add(1, "alu", "+", t16, header, R("constant_0"))
    add(1, "alu", "+", t10, header+3, R("constant_0"))
    offsets[2256], offsets[2262] = t16, t10
    for number, address in enumerate(sorted(addresses)):
        pointer = pointers[number]
        if address == 2272: add(20, "alu", "+", pointer, ptr0, R("constant_0"))
        else: add(20, "alu", "+" if address > 2272 else "-", pointer, ptr0, offsets[address])
        q = addresses.index(address)
        add(160 + number, "store", "vstore", pointer, image + 3*q)
    address_jobs = {}
    for t,e,s,(name,coord) in rows:
        if name in ("packet_left_base", "packet_right_base", "packet_partial_address", "packet_address") and coord[0] == 13 and coord[1] < groups:
            address_jobs.setdefault(coord[1],{})[name]=(t,e,s)
    changed = []
    for t,e,s,tag in rows:
        name, coord = tag
        if name in ("packet_left_base", "packet_right_base", "packet_partial_address", "packet_address") and coord[0] == 13 and coord[1] < groups:
            _, batch = coord
            if name != "packet_address": continue
            # Placeholder; allocate the two inputs below, including their
            # longer lifetimes. The old operands die at the old partial select.
            s = ("multiply_add",s[1],-1,R("cache_bit",13,batch),-1)
        elif name in ("cache_bit", "cache_bit_2") and coord[0] == 13 and coord[1] < groups:
            lane = coord[2] if len(coord) == 3 else 0
            s = ("|", s[1], s[2], mask + lane)
        changed.append((t,e,s,tag))
    for batch,job in address_jobs.items():
        changed.append((job['packet_left_base'][0],'flow',
                        ('vselect',-1,R('cache_bit',12,batch),w1,w0),
                        ('bilinear_left',(batch,))))
        changed.append((job['packet_right_base'][0],'flow',
                        ('vselect',-1,R('cache_bit',11,batch),R('constant_16'),R('constant_3')),
                        ('bilinear_right',(batch,))))
    changed.sort(key=lambda row: row[0])
    # Fixed-calendar interval coloring for the new temporary vectors. Include
    # every original architectural write, even writes to dead vector padding.
    busy=[0]*1536;born={};last={}
    bytime=defaultdict(list)
    for row in changed:bytime[row[0]].append(row)
    def reserve(word,lo,hi):
        if word>=0 and hi>=lo:busy[word] |= ((1<<(hi-lo+1))-1)<<lo
    for t,packet in sorted(bytime.items()):
        for _,engine,slot,tag in packet:
            rr,ww=scratch_accesses(engine,slot)
            for word in rr:
                if word>=0:last[word]=2*t
        for _,engine,slot,tag in packet:
            rr,ww=scratch_accesses(engine,slot)
            for word in ww:
                if word in born:reserve(word,born[word],max(born[word],last.get(word,-1)))
                elif word in last:reserve(word,0,last[word])
                born[word]=2*t+1;last.pop(word,None)
    for word,lo in born.items():reserve(word,lo,max(lo,last.get(word,-1)))
    for word in range(8):busy[word]=(1<<1800)-1
    allocated={}
    jobs=[]
    for batch,job in address_jobs.items():
        end=job['packet_address'][0]
        for name in ('packet_left_base','packet_right_base'):
            start=job[name][0]
            jobs.append((-(end-start),start,end,batch,name))
    for _,start,end,batch,name in sorted(jobs):
        maskbits=((1<<(2*end-(2*start+1)+1))-1)<<(2*start+1)
        base=next((b for b in range(8,1529) if all(not busy[b+l]&maskbits for l in range(8))),None)
        assert base is not None,('new vector allocation failed',batch,name,start,end)
        for lane in range(8):busy[base+lane]|=maskbits
        allocated[batch,name]=base
    final=[]
    for t,e,s,tag in changed:
        name,coord=tag
        if name=='packet_address' and coord[0]==13 and coord[1]<groups:
            batch=coord[1]
            s=(s[0],s[1],allocated[batch,'packet_right_base'],s[3],allocated[batch,'packet_left_base'])
        elif name in ('bilinear_left','bilinear_right'):
            key='packet_left_base' if name=='bilinear_left' else 'packet_right_base'
            s=(s[0],allocated[coord[0],key],*s[2:])
        final.append((t,e,s,tag))
    final.sort(key=lambda row:row[0])
    return final


def color_extra(rows,jobs):
    """Place additional whole vectors without hiding any padding writes."""
    busy=[0]*1536;born={};last={};bytime=defaultdict(list)
    for row in rows:bytime[row[0]].append(row)
    def reserve(word,lo,hi):
        if word>=0 and hi>=lo:busy[word]|=((1<<(hi-lo+1))-1)<<lo
    for t,packet in sorted(bytime.items()):
        for _,e,s,tag in packet:
            for word in scratch_accesses(e,s)[0]:
                if word>=0:last[word]=2*t+1
        for _,e,s,tag in packet:
            for word in scratch_accesses(e,s)[1]:
                if word in born:reserve(word,born[word],max(born[word],last.get(word,-1)))
                elif word in last:reserve(word,0,last[word])
                born[word]=2*t+1;last.pop(word,None)
    for word,lo in born.items():reserve(word,lo,max(lo,last.get(word,-1)))
    for word in range(8):busy[word]=(1<<2000)-1
    allocated={}
    for key,start,end in sorted(jobs,key=lambda x:-(x[2]-x[1])):
        bits=((1<<(2*end-2*start+1))-1)<<(2*start+1)
        base=next((b for b in range(8,1529) if all(not busy[b+l]&bits for l in range(8))),None)
        assert base is not None,('temporary allocation',key,start,end)
        for lane in range(8):busy[base+lane]|=bits
        allocated[key]=base
    return allocated


def fold_boundaries(rows,emit,count=27):
    """Two MAD + one select -> one MAD + three selects; saves eight lanes."""
    R=emit.reg
    choices={};stages={}
    for j,(t,e,s,(name,coord)) in enumerate(rows):
        if name in ('boundary_even','boundary_parent_offset'):
            stages[coord[0]]=(j,t,s)
        if name in ('boundary_offset','packet_boundary_choice'):
            assert e=='valu' and s[0]=='multiply_add'
            choices[coord[0]]=(j,t,s,name)
    targets=sorted(choices,key=lambda b:choices[b][1],reverse=True)[:count]
    changed=list(rows);jobs=[];remove=set();consumers={};extra=[]
    for batch in targets:
        first,t0,a=stages[batch];second,t1,b,name=choices[batch]
        assert t1>t0,(batch,t0,t1)
        remove.update((first,second))
        uses=[j for j,(t,e,s,(n,c)) in enumerate(rows)
              if n in ('next_packet_address','next_packet_address_2') and c==(batch,) and s[4]==b[1]]
        assert len(uses)==1,(batch,uses)
        user=uses[0];t2=rows[user][0]
        consumers[user]=batch
        # Form both b3 arms before b4's old consumer. This keeps b4 inside its
        # original lifetime; no extra parity extraction or vector copy is used.
        extra.append((t0-2,'flow',('vselect',-1,R('branch_bit',3,batch),R('constant_2863311502'),R('constant_2863311488')),('fold_even',(batch,))))
        extra.append((t0-1,'flow',('vselect',-1,R('branch_bit',3,batch),R('constant_2863311509'),R('constant_2863311495')),('fold_odd',(batch,))))
        extra.append((t0,'flow',('vselect',-1,R('branch_bit',4,batch),-1,-1),('fold_final',(batch,))))
        jobs.extend([(('even',batch),t0-2,t0),(('odd',batch),t0-1,t0),(('final',batch),t0,t2)])
    changed=[]
    for j,(t,e,s,tag) in enumerate(rows):
        if j in remove:continue
        if j in consumers:
            batch=consumers[j]
            s=(*s[:4],-1);tag=('fold_use',(batch,))
        changed.append((t,e,s,tag))
    changed.extend(extra)
    bases=color_extra(changed,jobs)
    for j,(t,e,s,(name,coord)) in enumerate(changed):
        if name=='fold_even':s=(s[0],bases['even',coord[0]],*s[2:])
        elif name=='fold_odd':s=(s[0],bases['odd',coord[0]],*s[2:])
        elif name=='fold_final':s=(s[0],bases['final',coord[0]],s[2],bases['odd',coord[0]],bases['even',coord[0]])
        elif name=='fold_use':s=(*s[:4],bases['final',coord[0]])
        changed[j]=t,e,s,(name,coord)
    changed.sort(key=lambda row:row[0])
    return changed


def graph(rows,scratch=True,condense=True,event_override=None):
    events = analyze(rows) if event_override is None else event_override
    n = len(rows)
    deps = [dict() for _ in rows]
    writers, readers = {}, defaultdict(set)
    mw, mr = defaultdict(set), defaultdict(set)
    groups = defaultdict(list)
    for j, row in enumerate(rows): groups[row[0]].append(j)
    zero_reads = sorted(rows[j][0] for j,e in events.items() if e[0]=='load' and 0 in e[2])
    zero_defs = sorted(rows[j][0] for j,e in events.items() if e[0]=='store' and e[1]==frozenset([0]))
    live_zero = {}
    for j,e in events.items():
        if e[0]=='store' and 0 in e[2]:
            t=rows[j][0]
            nr=next((x for x in zero_reads if x>t),99999)
            nd=next((x for x in zero_defs if x>t),99998)
            live_zero[j]=nr<=nd
    def edge(a,b,lag):
        if a != b: deps[b][a] = max(deps[b].get(a, 0), lag)
    for t, ids in sorted(groups.items()):
        for j in ids:
            r, w = scratch_accesses(rows[j][1], rows[j][2])
            for a in r if scratch else ():
                if a in writers: edge(writers[a],j,1)
                readers[a].add(j)
            if j in events and events[j][0] == "load":
                for a in events[j][2]:
                    for q in mw[a]: edge(q,j,1)
                    mr[a].add(j)
        for j in ids:
            r, w = scratch_accesses(rows[j][1], rows[j][2])
            for a in w if scratch else ():
                if a in writers: edge(writers[a],j,1)
                for q in readers[a]: edge(q,j,0)
                readers[a].clear(); writers[a] = j
            if j in events and events[j][0] == "store":
                _, bases, locations = events[j]
                for a in locations:
                    for q in mr[a]: edge(q,j,0)
                    # Unknown conditional writes to the dead zero sink do not
                    # need a serial WAW chain. Definite writes still close it.
                    if a != 0 or len(bases) == 1 or live_zero.get(j,False):
                        for q in mw[a]: edge(q,j,1)
                    # A load in this packet reads before BOTH store slots.
                    # Keep it until every same-cycle write has acquired WAR.
                    if a != 0:
                        mr[a] = {q for q in mr[a] if rows[q][0] == t}
                    if len(bases) == 1: mw[a].clear()
                    mw[a].add(j)
    successors = [[] for _ in rows]
    for j, dd in enumerate(deps):
        for d, lag in dd.items(): successors[d].append((j,lag))
    if not condense:
        return SimpleNamespace(n=n,deps=deps,succ=successors,original=[row[0] for row in rows])
    # Physical register reuse can force operations to coissue (zero-lag cycles).
    sys.setrecursionlimit(100000)
    seen=set();post=[]
    def dfs(j):
        seen.add(j)
        for q,lag in successors[j]:
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
    cd=[{} for _ in clusters]
    for j,dd in enumerate(deps):
        for d,lag in dd.items():
            a,b=owner[d],owner[j]
            if a==b:assert lag==0,("positive cycle",d,j,lag)
            else:cd[b][a]=max(cd[b].get(a,0),lag)
    cs=[[] for _ in clusters]
    for j,dd in enumerate(cd):
        for d,lag in dd.items():cs[d].append((j,lag))
    demands=[Counter(rows[j][1] for j in ids) for ids in clusters]
    for index,demand in enumerate(demands):
        assert all(demand[e]<=LIMITS[e] for e in demand),(demand,[rows[j] for j in clusters[index]])
    return SimpleNamespace(n=len(clusters),deps=cd,succ=cs,
                           original=[max(rows[j][0] for j in ids) for ids in clusters],
                           clusters=clusters,demands=demands)


def schedule(rows,g,reverse=False,noise=0,seed=0):
    rng=random.Random(seed)
    ds=[dict(s) for s in g.succ] if reverse else g.deps
    nxt=[list(d.items()) for d in g.deps] if reverse else g.succ
    horizon=max(g.original)
    pr=[(horizon-t if reverse else t)+noise*rng.random() for t in g.original]
    degree=list(map(len,ds));earliest=[0]*g.n;times=[-1]*g.n
    ready=[(pr[j],j) for j,d in enumerate(degree) if not d];heapq.heapify(ready)
    future=[];t=0;done=0
    while done<g.n:
        while future and future[0][0]<=t:
            _,j=heapq.heappop(future);heapq.heappush(ready,(pr[j],j))
        used=Counter();blocked=[]
        while ready:
            item=heapq.heappop(ready);_,j=item
            if any(used[e]+count>LIMITS[e] for e,count in g.demands[j].items()):
                blocked.append(item);continue
            used.update(g.demands[j]);times[j]=t;done+=1
            for q,lag in nxt[j]:
                earliest[q]=max(earliest[q],t+lag);degree[q]-=1
                if not degree[q]:
                    if earliest[q]<=t:heapq.heappush(ready,(pr[q],q))
                    else:heapq.heappush(future,(earliest[q],q))
        ready=blocked;heapq.heapify(ready);t+=1
        assert t<3000,('cycle or bad graph',done,g.n)
    if reverse: times=[t-1-v for v in times]
    for j,dd in enumerate(g.deps):
        for d,lag in dd.items(): assert times[j]>=times[d]+lag
    program=[{} for _ in range(t)]
    row_time={row:times[j] for j,ids in enumerate(g.clusters) for row in ids}
    for row,(_,e,s,_) in enumerate(rows):
        program[row_time[row]].setdefault(e,[]).append(s)
    return program,times


def verify(program,seeds=3):
    for seed in range(seeds):
        rng=random.Random(seed)
        tree=Tree(10,[rng.getrandbits(32) for _ in range(2047)])
        inp=Input([0]*256,[rng.getrandbits(32) for _ in range(256)],16)
        mem=build_mem_image(tree,inp);ref=mem.copy()
        for _ in reference_kernel2(ref):pass
        machine=Machine(mem,program,source.KernelBuilder().debug_info())
        machine.enable_debug=machine.enable_pause=False;machine.run()
        assert machine.mem[2310:2566]==ref[2310:2566],seed
    return machine.cycle


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bilinear',action='store_true')
    ap.add_argument('--fold',type=int,default=0)
    ap.add_argument('--passes',type=int,default=8)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    rows,emit=capture()
    if args.bilinear:rows=bilinear(rows,emit)
    if args.fold:rows=fold_boundaries(rows,emit,args.fold)
    g=graph(rows)
    counts=Counter(e for t,e,s,tag in rows)
    print({'operations':dict(counts),'lane_work':counts['alu']+8*counts['valu'],'edges':sum(map(len,g.deps))},flush=True)
    best=None
    for step in range(args.passes):
        program,times=schedule(rows,g,reverse=bool(step%2),noise=0)
        g.original=times
        if best is None or len(program)<len(best):
            best=program
            print({'pass':step,'cycles':len(best),'verified':verify(best)},flush=True)
            if args.output:args.output.write_text(json.dumps(best))


if __name__=='__main__':main()
