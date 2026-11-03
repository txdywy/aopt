"""Repair a near-optimal calendar with time-indexed CP-SAT constraints.

Vector/scalar modes stay fixed. Each real instruction may move by a small
number of packets; all data/memory edges and per-cycle engine limits hold.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import pickle

from ortools.sat.python import cp_model


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--target',type=int,required=True)
    ap.add_argument('--radius',type=int,default=1)
    ap.add_argument('--seconds',type=float,default=60)
    ap.add_argument('--seed',type=int,default=1)
    ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();prefix=str(args.prefix)
    ir=pickle.loads(Path(prefix+'.pkl').read_bytes())
    data=list(map(int,Path(prefix+'.sol').read_text().split()))
    n,ng=data[:2];old=data[2:2+n];modes=data[2+n:]
    assert (n,ng)==(len(ir.ops),len(ir.groups))
    rep=list(range(n));native=set()
    for g,(_,_,ids,_) in enumerate(ir.groups):
        if modes[g]:
            root=ids[0];native.add(root)
            for j in ids:rep[j]=root
    real=sorted(set(rep));model=cp_model.CpModel();times={};choices={}
    slots=defaultdict(list);engines={'alu':0,'valu':1,'load':2,'store':3,'flow':4}
    for j in real:
        lo=max(0,old[j]-args.radius);hi=min(args.target-1,old[j]+args.radius)
        if lo>hi:
            print({'status':'DOMAIN_INFEASIBLE','instruction':j},flush=True);return
        var=model.new_int_var(lo,hi,f't{j}');times[j]=var
        options=[];port=1 if j in native else engines[ir.ops[j].engine]
        for t in range(lo,hi+1):
            flag=model.new_bool_var(f'x{j}_{t}');options.append((t,flag))
            slots[port,t].append(flag)
            model.add_hint(flag,int(t==min(hi,max(lo,old[j]))))
        choices[j]=options
        model.add_exactly_one(flag for _,flag in options)
        model.add(var==sum(t*flag for t,flag in options))
        model.add_hint(var,min(hi,max(lo,old[j])))
    edges={}
    for j,dd in enumerate(ir.deps):
        b=rep[j]
        for p,lag in dd.items():
            a=rep[p]
            if a==b:
                assert lag==0;continue
            edges[a,b]=max(edges.get((a,b),0),lag)
    for (a,b),lag in edges.items():
        # Omit edges already implied by the allowed movement domains.
        if old[b]-old[a]>=lag+2*args.radius:continue
        model.add(times[b]>=times[a]+lag)
    for (port,t),flags in slots.items():
        model.add(sum(flags)<=(12,6,2,2,1)[port])
    solver=cp_model.CpSolver()
    solver.parameters.max_time_in_seconds=args.seconds
    solver.parameters.num_search_workers=args.workers
    solver.parameters.random_seed=args.seed
    solver.parameters.stop_after_first_solution=True
    solver.parameters.repair_hint=True
    solver.parameters.hint_conflict_limit=100000
    print({'real_instructions':len(real),'variables':sum(map(len,choices.values())),
           'target':args.target,'radius':args.radius},flush=True)
    status=solver.solve(model)
    print({'status':solver.status_name(status),'seconds':solver.wall_time},flush=True)
    if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):return
    out=[solver.value(times[rep[j]]) for j in range(n)]
    for j,dd in enumerate(ir.deps):
        for p,lag in dd.items():assert out[j]>=out[p]+lag
    usage=defaultdict(int)
    for j in real:
        port=1 if j in native else engines[ir.ops[j].engine]
        usage[port,out[j]]+=1
    for (port,t),count in usage.items():assert count<=(12,6,2,2,1)[port]
    args.output.write_text(f'{n} {ng}\n'+' '.join(map(str,out))+'\n'+' '.join(map(str,modes))+'\n')
    print({'cycles':max(out)+1,'moved':sum(out[j]!=old[j] for j in real),
           'output':str(args.output)},flush=True)


if __name__=='__main__':main()
