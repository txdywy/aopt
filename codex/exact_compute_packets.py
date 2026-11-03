"""Pack a compute window exactly around a supplied fixed memory calendar.

Uses time-indexed Boolean choices for individual ALU lanes or complete native
vectors. Every original SSA, padding-write and memory-order dependency remains.
The input calendar may contain compute violations; its fixed operations must
already meet their own capacities. Requires the optional research OR-Tools.
"""
from __future__ import annotations

import argparse
from collections import Counter
import heapq
from pathlib import Path
import pickle

from ortools.sat.python import cp_model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--lo', type=int, default=800)
    ap.add_argument('--horizon', type=int, required=True)
    ap.add_argument('--radius', type=int, default=4)
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--log-search', action='store_true',
                    help='Print CP-SAT search progress for diagnosing a hard window')
    ap.add_argument('--repair-hint', action='store_true',
                    help='Use optional CP hint repair (some OR-Tools builds crash with it)')
    ap.add_argument('--hi', type=int,
                    help='Freeze packets at or after this interior boundary')
    ap.add_argument('--move-streams', action='store_true',
                    help='Jointly move LOAD/STORE/FLOW inside the same window')
    ap.add_argument('--pack-before', type=int,
                    help='Require near-full arithmetic utilization before this cycle')
    ap.add_argument('--max-waste', type=int, default=0,
                    help='Allowed arithmetic waste between lo and pack-before')
    ap.add_argument('--maximize-prefix',action='store_true',
                    help='Optimize prefix density instead of requiring the full density target')
    ap.add_argument('--total-valu', type=int,
                    help='Optional exact count of native VALU instructions in the full schedule')
    ap.add_argument('--tail-capacities', type=str,
                    help='Proved per-cycle arithmetic caps for the last cycles, e.g. 48,16,0')
    args = ap.parse_args()
    ir = pickle.loads(Path(str(args.prefix) + '.pkl').read_bytes())
    data = list(map(int, Path(str(args.prefix) + '.sol').read_text().split()))
    n, ng = data[:2]
    assert (n, ng) == (len(ir.ops), len(ir.groups))
    old = data[2:2 + n]
    modes = data[2 + n:]
    hi=args.hi or args.horizon
    assert args.lo<hi<=args.horizon
    tail_caps = ([int(value) for value in args.tail_capacities.split(',')]
                 if args.tail_capacities else [])
    assert all(0 <= cap <= 60 for cap in tail_caps)
    assert len(tail_caps) <= args.horizon
    chosen = {j for j, op in enumerate(ir.ops)
              if (args.move_streams or op.engine in ('alu', 'valu'))
              and args.lo<=old[j]<(args.hi or max(old)+1)}
    outside=[j for j,t in enumerate(old) if j not in chosen and not 0<=t<args.horizon]
    if outside:
        print({'status':'FIXED_OUTSIDE_HORIZON','nodes':outside[:10]},flush=True)
        return
    member = {j: g for g, (_, _, ids, _) in enumerate(ir.groups) for j in ids}
    freegroups = {g for g, (_, _, ids, _) in enumerate(ir.groups) if set(ids) <= chosen}
    low = [old[j] if j not in chosen else max(args.lo, old[j] - args.radius)
           for j in range(n)]
    high = [old[j] if j not in chosen else min(hi - 1, old[j] + args.radius)
            for j in range(n)]
    degrees = list(map(len, ir.deps))
    ready = [j for j, degree in enumerate(degrees) if not degree]
    heapq.heapify(ready)
    order = []
    while ready:
        j = heapq.heappop(ready)
        order.append(j)
        for q, lag in ir.succ[j]:
            low[q] = max(low[q], low[j] + lag)
            degrees[q] -= 1
            if not degrees[q]:
                heapq.heappush(ready, q)
    assert len(order) == n
    for j in reversed(order):
        for q, lag in ir.succ[j]:
            high[j] = min(high[j], high[q] - lag)
    bad = [j for j in range(n) if low[j] > high[j]]
    if bad:
        print({'status': 'WINDOW_INFEASIBLE', 'nodes': bad[:10]}, flush=True)
        return
    model = cp_model.CpModel()
    single = {(j, t): model.new_bool_var(f's{j}_{t}') for j in chosen
              for t in range(low[j], high[j] + 1)}
    native = {}
    for g in freegroups:
        ids = ir.groups[g][2]
        first, last = max(low[j] for j in ids), min(high[j] for j in ids)
        for t in range(first, last + 1):
            native[g, t] = model.new_bool_var(f'v{g}_{t}')
        model.add(sum(native.get((g, t), 0) for t in range(first, last + 1)) <= 1)

    def at(j, t):
        return single.get((j, t), 0) + native.get((member.get(j, -1), t), 0)

    for j in chosen:
        model.add(sum(at(j, t) for t in range(low[j], high[j] + 1)) == 1)
        for d, lag in ir.deps[j].items():
            if d not in chosen:
                continue
            for t in range(low[j], high[j] + 1):
                model.add(at(j, t) <= sum(at(d, s)
                    for s in range(low[d], min(high[d], t - lag) + 1)))
    fixed_use = Counter()
    for j, op in enumerate(ir.ops):
        if j in chosen:
            continue
        g = member.get(j)
        if g is not None and modes[g]:
            ids = ir.groups[g][2]
            assert not set(ids) & chosen
            if j != ids[0]:
                continue
            engine = 'valu'
        else:
            engine = op.engine
        fixed_use[engine, old[j]] += 1
    singles_by_time = {}
    for (j, t), var in single.items():
        singles_by_time.setdefault((ir.ops[j].engine, t), []).append(var)
        g = member.get(j)
        hint = int(old[j] == t and (g not in freegroups or not modes[g]))
        model.add_hint(var, hint)
    native_by_time = {}
    for (g, t), var in native.items():
        native_by_time.setdefault(t, []).append(var)
        model.add_hint(var, int(bool(modes[g]) and old[ir.groups[g][2][0]] == t))
    if args.total_valu is not None:
        fixed_valu = sum(amount for (engine, _), amount in fixed_use.items()
                         if engine == 'valu')
        model.add(sum(var for (j, _), var in single.items() if ir.ops[j].engine == 'valu')
                  + sum(native.values()) + fixed_valu == args.total_valu)
    for t in range(args.lo, hi):
        alu=sum(singles_by_time.get(('alu', t), ()))
        valu=sum(singles_by_time.get(('valu', t), ()))+sum(native_by_time.get(t, ()))
        model.add(alu<=12-fixed_use['alu',t])
        model.add(valu<=6-fixed_use['valu',t])
        if tail_caps and t >= args.horizon - len(tail_caps):
            cap = tail_caps[t - (args.horizon - len(tail_caps))]
            model.add(alu + 8 * valu + fixed_use['alu', t]
                      + 8 * fixed_use['valu', t] <= cap)
        if args.pack_before is not None and t<args.pack_before and not args.maximize_prefix:
            # The aggregate lane budget implies these equalities. Stating
            # them per packet gives much stronger Boolean propagation.
            if args.max_waste<8:
                model.add(valu==6-fixed_use['valu',t])
                model.add(alu>=12-args.max_waste-fixed_use['alu',t])
            if args.max_waste==0:
                model.add(alu==12-fixed_use['alu',t])
        for engine,capacity in (('load',2),('store',2),('flow',1)):
            model.add(sum(singles_by_time.get((engine,t),()))<=capacity-fixed_use[engine,t])
    if args.pack_before is not None:
        assert args.lo<args.pack_before<=hi
        terms=[]
        for (j,t),var in single.items():
            if t<args.pack_before and ir.ops[j].engine in ('alu','valu'):
                terms.append(var*(8 if ir.ops[j].engine=='valu' else 1))
        terms.extend(var*8 for (g,t),var in native.items() if t<args.pack_before)
        fixed_work=sum(8 if op.engine=='valu' else 1 for j,op in enumerate(ir.ops)
                       if j not in chosen and op.engine in ('alu','valu')
                       and args.lo<=old[j]<args.pack_before)
        if args.maximize_prefix:
            model.maximize(sum(terms)+fixed_work)
        else:
            model.add(sum(terms)+fixed_work>=60*(args.pack_before-args.lo)-args.max_waste)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.seconds
    solver.parameters.num_search_workers = args.workers
    solver.parameters.random_seed = args.seed
    solver.parameters.log_search_progress = args.log_search
    solver.parameters.repair_hint = args.repair_hint
    solver.parameters.hint_conflict_limit = 100000
    print({'nodes': len(chosen), 'groups': len(freegroups),
           'booleans': len(single) + len(native), 'horizon': args.horizon}, flush=True)
    status = solver.solve(model)
    print({'status': solver.status_name(status), 'seconds': solver.wall_time}, flush=True)
    if args.maximize_prefix:
        print({'prefix_lanes_incumbent':solver.objective_value,
               'prefix_lanes_upper_bound':solver.best_objective_bound,
               'window_capacity':60*(args.pack_before-args.lo),
               'scope':'specified joint window, movement radius and frozen outside'},flush=True)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return
    result = list(old)
    for j in chosen:
        result[j] = next(t for t in range(low[j], high[j] + 1) if solver.value(at(j, t)))
    for g in freegroups:
        modes[g] = int(any(solver.value(var) for (gg, _), var in native.items() if gg == g))
    assert max(result)<args.horizon
    use = Counter()
    for j, op in enumerate(ir.ops):
        for d, lag in ir.deps[j].items():
            assert result[j] >= result[d] + lag, (d, j)
        g = member.get(j)
        if g is not None and modes[g]:
            ids = ir.groups[g][2]
            assert len({result[k] for k in ids}) == 1
            if j != ids[0]:
                continue
            engine = 'valu'
        else:
            engine = op.engine
        use[engine, result[j]] += 1
    for (engine, _), amount in use.items():
        assert amount <= {'alu': 12, 'valu': 6, 'load': 2, 'store': 2, 'flow': 1}[engine]
    if args.total_valu is not None:
        assert sum(amount for (engine, _), amount in use.items()
                   if engine == 'valu') == args.total_valu
    for offset, cap in enumerate(tail_caps):
        cycle = args.horizon - len(tail_caps) + offset
        assert use['alu', cycle] + 8 * use['valu', cycle] <= cap
    assert max(result) < args.horizon
    args.output.write_text(f'{n} {ng}\n' + ' '.join(map(str, result))
                          + '\n' + ' '.join(map(str, modes)) + '\n')
    print({'cycles': max(result) + 1, 'output': str(args.output)}, flush=True)


if __name__ == '__main__':
    main()
