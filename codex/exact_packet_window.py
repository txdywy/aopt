"""CP-SAT rescheduling of a bounded packet window with flexible ALU modes.

Install OR-Tools outside the submission and add it to PYTHONPATH when using
this research utility. All frozen operations and cross-window edges remain.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import pickle

from ortools.sat.python import cp_model


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--lo',type=int,required=True)
    ap.add_argument('--hi',type=int)
    ap.add_argument('--target',type=int)
    ap.add_argument('--seconds',type=float,default=45)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--seed',type=int,default=1)
    ap.add_argument('--global-memory',action='store_true',
                    help='allow all LOAD/STORE/FLOW operations to cross the window boundary')
    ap.add_argument('--pack-before',type=int)
    ap.add_argument('--slack',type=int,default=52)
    ap.add_argument('--radius',type=int,
                    help='reschedule the whole graph within this distance of the old calendar')
    ap.add_argument('--fixed-modes',action='store_true')
    args=ap.parse_args();prefix=str(args.prefix)
    ir=pickle.loads(Path(prefix+'.pkl').read_bytes())
    old=list(map(int,Path(prefix+'.sol').read_text().split()))
    n,ng=old[:2];ts=old[2:2+n];md=old[2+n:]
    assert (n,ng)==(len(ir.ops),len(ir.groups))
    oldh=max(ts)+1;hi=args.hi or oldh;limit=args.target or hi
    globalops={j for j,op in enumerate(ir.ops)
               if args.global_memory and op.engine in ('load','store','flow')}
    chosen=(set(range(n)) if args.radius is not None else
            {j for j,t in enumerate(ts) if args.lo<=t<hi}|globalops)
    if args.target is not None:
        assert args.lo<limit<=hi
        # Compress an interior/head window as well as a tail window. Later
        # frozen packets shift by the same amount, including all cross edges.
        delta=limit-hi
        for j,t in enumerate(ts):
            if t>=hi and j not in chosen:ts[j]=t+delta
    members={j:g for g,(_,_,ids,_) in enumerate(ir.groups) for j in ids}
    model=cp_model.CpModel();times={};modes={};intervals=[[] for _ in range(5)]
    engine={'alu':0,'valu':1,'load':2,'store':3,'flow':4}
    for j in chosen:
        earliest=0 if j in globalops else args.lo
        latest=(oldh+limit-hi-1) if j in globalops else limit-1
        if args.radius is not None:
            earliest=max(0,ts[j]-args.radius)
            latest=min(limit-1,ts[j]+args.radius)
        for p,lag in ir.deps[j].items():
            if p not in chosen:earliest=max(earliest,ts[p]+lag)
        for q,lag in ir.succ[j]:
            if q not in chosen:latest=min(latest,ts[q]-lag)
        if earliest>latest:
            print({'status':'CROSS_BOUNDARY_INFEASIBLE','node':j,'bounds':(earliest,latest)})
            return
        times[j]=model.new_int_var(earliest,latest,f't{j}')
        model.add_hint(times[j],min(latest,max(earliest,ts[j])))
    freegroups={g for g,(_,_,ids,_) in enumerate(ir.groups) if set(ids)<=chosen}
    for g in freegroups:
        _,_,ids,_=ir.groups[g]
        mode=model.new_bool_var(f'm{g}');modes[g]=mode
        if args.fixed_modes:model.add(mode==md[g])
        model.add_hint(mode,md[g])
        root=times[ids[0]]
        for j in ids[1:]:model.add(times[j]==root).only_enforce_if(mode)
        intervals[1].append(model.new_optional_fixed_size_interval_var(root,1,mode,f'v{g}'))
    for j in chosen:
        op=ir.ops[j];g=members.get(j)
        if g in freegroups:
            interval=model.new_optional_fixed_size_interval_var(times[j],1,modes[g].Not(),f'a{j}')
            intervals[0].append(interval)
        else:
            # A native group is wholly at one time and hence wholly selected.
            assert g is None or md[g]==0
            intervals[engine[op.engine]].append(model.new_fixed_size_interval_var(times[j],1,f'i{j}'))
        for p,lag in ir.deps[j].items():
            if p in chosen:model.add(times[j]>=times[p]+lag)
    for e,cap in enumerate((12,6,2,2,1)):
        model.add_cumulative(intervals[e],[1]*len(intervals[e]),cap)
    if args.pack_before is not None:
        early=[]
        for j in chosen:
            op=ir.ops[j]
            if op.engine not in ('alu','valu'):continue
            flag=model.new_bool_var(f'early{j}')
            model.add(times[j]<args.pack_before).only_enforce_if(flag)
            model.add(times[j]>=args.pack_before).only_enforce_if(flag.Not())
            early.append(flag*(8 if op.engine=='valu' else 1))
        model.add(sum(early)>=60*args.pack_before-args.slack)
    elif args.target is None:
        # Packing arithmetic earlier improves head utilization and gives the
        # global bidirectional scheduler a substantially different warm start.
        objective=[]
        for j in chosen:
            op=ir.ops[j]
            if op.engine in ('alu','valu'):
                objective.append(times[j]*(8 if op.engine=='valu' else 1))
        model.minimize(sum(objective))
    solver=cp_model.CpSolver()
    solver.parameters.max_time_in_seconds=args.seconds
    solver.parameters.num_search_workers=args.workers
    solver.parameters.random_seed=args.seed
    print({'window':[args.lo,hi],'limit':limit,'nodes':len(chosen),
           'flexible_groups':len(freegroups)},flush=True)
    status=solver.solve(model)
    print({'status':solver.status_name(status),'seconds':solver.wall_time,
           'objective':solver.objective_value,'bound':solver.best_objective_bound},flush=True)
    if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):return
    for j,v in times.items():ts[j]=solver.value(v)
    for g,v in modes.items():md[g]=solver.value(v)
    by=[Counter() for _ in range(max(ts)+1)]
    for j,op in enumerate(ir.ops):
        for p,lag in ir.deps[j].items():assert ts[j]>=ts[p]+lag,(p,j)
        g=members.get(j)
        if g is not None and md[g]:
            ids=ir.groups[g][2];assert len({ts[k] for k in ids})==1
            if j!=ids[0]:continue
            e=1
        else:e=engine[op.engine]
        by[ts[j]][e]+=1
    for t,c in enumerate(by):
        for e,cap in enumerate((12,6,2,2,1)):assert c[e]<=cap,(t,e,c[e])
    args.output.write_text(f'{n} {ng}\n'+' '.join(map(str,ts))+'\n'+' '.join(map(str,md))+'\n')
    print({'cycles':max(ts)+1,'head20_slack':sum(60-c[0]-8*c[1] for c in by[:20]),
           'output':str(args.output)},flush=True)


if __name__=='__main__':main()
