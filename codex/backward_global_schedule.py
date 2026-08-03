"""Reverse topological list scheduler with explicit latest-time bounds."""

from __future__ import annotations

from collections import Counter
import heapq
import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops, validate
from problem import SLOT_LIMITS


def build_ops():
    configure_target(); kernel.SCHEDULE_EXACT_CYCLES = None
    b = kernel.KernelBuilder()
    try:
        b.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        if not hasattr(b, "dag_ops"):
            raise
    return real_tail_ops(b.dag_ops)


def main() -> None:
    ops = build_ops(); n = len(ops)
    children = [[] for _ in ops]
    child_count = [0] * n
    for child, op in enumerate(ops):
        child_count[child] = 0
        for parent, lag in op.parents.items():
            children[parent].append((child, int(lag)))
            child_count[parent] += 1
    # A forward topological walk gives exact source depth and reverse reach.
    indeg = [len(op.parents) for op in ops]
    q = [i for i, d in enumerate(indeg) if not d]; heapq.heapify(q); topo=[]
    while q:
        p=heapq.heappop(q); topo.append(p)
        for c,_ in children[p]:
            indeg[c]-=1
            if indeg[c]==0: heapq.heappush(q,c)
    depth=[0]*n; reach=[0]*n
    for p in topo:
        for c,lag in children[p]:
            depth[c]=max(depth[c], depth[p]+lag+1)
    for p in reversed(topo):
        reach[p]=min(1_000_000,len(children[p])+sum(reach[c] for c,_ in children[p]))
    horizon=int(os.environ.get("TARGET","959"))
    latest=[horizon-1]*n
    scheduled=[-1]*n
    heaps={e:[] for e in SLOT_LIMITS}
    rng_seed=int(os.environ.get("SEED","1"))
    hint={}
    for raw in os.environ.get("HINTS","").split(","):
        if not raw: continue
        d=json.loads(Path(raw).read_text()); c=d.get("cycles",{})
        hint.update({i:int(v) for i,v in enumerate(c)} if isinstance(c,list) else {int(i):int(v) for i,v in c.items()})
    hint_weight=int(os.environ.get("HINT_WEIGHT","0"))
    def key(i):
        # Among operations that can still finish at the current cycle, keep
        # the deepest/highest-fanout cone on the machine first.  A smaller
        # latest bound is more urgent; source hints break equal deadlines.
        return (latest[i], -depth[i], -reach[i]//16, hint_weight*hint.get(i,horizon), i)
    def push(i): heapq.heappush(heaps[ops[i].engine],(key(i),i))
    for i,d in enumerate(child_count):
        if d==0: push(i)
    order=tuple(e for e in os.environ.get("ENGINE_ORDER","store,flow,load,valu,alu,debug").split(",") if e)
    done=0; cycle=horizon-1
    while done<n:
        used={e:0 for e in heaps}; progress=False
        # Refresh each heap's priorities because latest bounds tighten while
        # children are assigned in this same reverse bundle.
        for e in heaps:
            items=[(key(i),i) for _,i in heaps[e]]
            heapq.heapify(items); heaps[e]=items
        while True:
            inner=False
            for e in order:
                if e not in heaps or used[e]>=SLOT_LIMITS[e]: continue
                heap=heaps[e]; chosen=None; deferred=[]
                while heap:
                    _,i=heapq.heappop(heap)
                    if latest[i]>=cycle:
                        chosen=i; break
                    # An operation whose latest is earlier moves the global
                    # frontier; keep it queued for the next earlier cycle.
                    deferred.append((key(i),i))
                for x in deferred: heapq.heappush(heap,x)
                if chosen is None: continue
                scheduled[chosen]=cycle; used[e]+=1; done+=1; progress=inner=True
                for parent,lag in ops[chosen].parents.items():
                    latest[parent]=min(latest[parent],cycle+(-int(lag)))
                    child_count[parent]-=1
                    if child_count[parent]==0: push(parent)
            if not inner: break
        if not progress:
            candidates=[latest[i] for e in heaps for _,i in heaps[e] if latest[i]<cycle]
            cycle=min(candidates,default=cycle-1)
        else:
            cycle-=1
        if cycle<0 and done<n: raise RuntimeError(f"backward schedule underflow done={done}/{n}")
    low=min(scheduled); shifted=[x-low for x in scheduled] if low<0 else scheduled
    validate(ops,shifted); makespan=max(shifted)+1
    print(f"makespan={makespan} jobs={n} low={low}",flush=True)
    out=Path(os.environ.get("OUT",".search959/backward-global.json"));out.write_text(json.dumps({"makespan":makespan,"cycles":shifted}));print(f"output={out}")


if __name__=="__main__": main()
