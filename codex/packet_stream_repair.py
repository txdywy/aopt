"""Repair LOAD/STORE/FLOW order after eliminating unconstrained compute.

This emits a relaxed calendar, never an executable score. A separate compute
packing and frozen-machine replay are mandatory. Arithmetic before the chosen
boundary stays fixed; every original dependency remains in the projection.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from pathlib import Path
import pickle
import json

from ortools.sat.python import cp_model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--horizon', type=int, required=True)
    ap.add_argument('--compute-before', type=int, default=600)
    ap.add_argument('--stream-before', type=int, default=700)
    ap.add_argument('--radius', type=int, default=6)
    ap.add_argument('--seconds', type=float, default=30)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--feasibility-only', action='store_true',
                    help='Find any projected calendar without a movement objective')
    ap.add_argument('--repair-hint', action='store_true',
                    help='Enable optional CP hint repair')
    ap.add_argument('--export-model', type=Path)
    ap.add_argument('--energy-cuts', type=Path)
    ap.add_argument('--global-energy', action='store_true',
                    help='count all movable arithmetic, avoiding exchanges into a constrained window')
    ap.add_argument('--free-outputs', action='store_true',
                    help='Allow final output stores to wait through the full drain')
    ap.add_argument('--soft-energy',action='store_true',
                    help='Minimize energetic shortfalls before stream movement; still a relaxation')
    ap.add_argument('--hint-calendar',type=Path,
                    help='Use an independently valid projected LF calendar as a solver hint')
    args = ap.parse_args()
    ir = pickle.loads(Path(str(args.prefix) + '.pkl').read_bytes())
    data = list(map(int, Path(str(args.prefix) + '.sol').read_text().split()))
    n, ng = data[:2]
    assert (n, ng) == (len(ir.ops), len(ir.groups))
    old = data[2:2 + n]
    hint_times=list(old)
    if args.hint_calendar:
        hint=list(map(int,args.hint_calendar.read_text().split()))
        assert hint[:2]==[n,ng]
        hint_times=hint[2:2+n]
    model = cp_model.CpModel()
    variables = {}
    firsts, lasts, projected_edges = {}, {}, []
    for j, op in enumerate(ir.ops):
        if op.engine in ('load', 'store', 'flow') and old[j] >= args.stream_before:
            first = max(0, old[j] - args.radius)
            last = min(args.horizon - 1, old[j] + args.radius)
            if args.free_outputs and op.tag[0]=='output':
                last=args.horizon-1
            variables[j] = model.new_int_var(first, last, f't{j}')
            firsts[j], lasts[j] = first, last
            model.add_hint(variables[j], min(last, max(first, hint_times[j])))
    degree = list(map(len, ir.deps))
    ready = deque(j for j, d in enumerate(degree) if not d)
    sources = [None] * n
    constants = [0] * n
    projected = 0
    order = []
    while ready:
        j = ready.popleft()
        order.append(j)
        incoming = {}
        absolute = 0
        for d, lag in ir.deps[j].items():
            absolute = max(absolute, constants[d] + lag)
            for root, distance in sources[d].items():
                incoming[root] = max(incoming.get(root, -1), distance + lag)
        op = ir.ops[j]
        fixed = j not in variables and (op.engine in ('load', 'store', 'flow')
                                       or old[j] < args.compute_before)
        if j in variables:
            model.add(variables[j] >= absolute)
            firsts[j] = max(firsts[j], absolute)
            for root, distance in incoming.items():
                model.add(variables[j] >= variables[root] + distance)
                projected_edges.append((root, j, distance))
                projected += 1
            constants[j], sources[j] = 0, {j: 0}
        elif fixed:
            if absolute > old[j]:
                model.add(False)
            for root, distance in incoming.items():
                model.add(variables[root] + distance <= old[j])
                lasts[root] = min(lasts[root], old[j] - distance)
                projected += 1
            constants[j], sources[j] = old[j], {}
        else:
            constants[j], sources[j] = max(args.compute_before, absolute), incoming
            if constants[j] >= args.horizon:
                model.add(False)
            for root, distance in incoming.items():
                model.add(variables[root] + distance < args.horizon)
                lasts[root] = min(lasts[root], args.horizon - 1 - distance)
        for q, _ in ir.succ[j]:
            degree[q] -= 1
            if not degree[q]:
                ready.append(q)
    assert all(source is not None for source in sources)
    energy_groups = []
    shortfalls=[]
    if args.energy_cuts:
        cut_data = json.loads(args.energy_cuts.read_text())
        assert cut_data['freeze_before'] == args.compute_before
        assert cut_data['horizon'] == args.horizon
        right = [None] * n
        upper = [args.horizon - 1] * n
        for j in reversed(order):
            op = ir.ops[j]
            fixed = j not in variables and (op.engine in ('load', 'store', 'flow')
                                           or old[j] < args.compute_before)
            if j in variables:
                right[j] = {j: 0}
            elif fixed:
                upper[j], right[j] = old[j], {}
            else:
                values = {}
                for q, lag in ir.succ[j]:
                    upper[j] = min(upper[j], upper[q] - lag)
                    for root, distance in right[q].items():
                        values[root] = max(values.get(root, -1), distance + lag)
                right[j] = values
        for cut_rank, cut in enumerate(cut_data['cuts']):
            profiles = Counter()
            cut_jobs = ([j for j, op in enumerate(ir.ops) if op.engine in ('alu', 'valu')
                         and old[j] >= args.compute_before] if args.global_energy else cut['jobs'])
            need = (sum(8 if ir.ops[j].engine == 'valu' else 1 for j in cut_jobs)
                    - cut['capacity']) if args.global_energy else cut['excess']
            for j in cut_jobs:
                assert ir.ops[j].engine in ('alu', 'valu')
                before, after = None, None
                if constants[j] < cut['a']:
                    pairs = tuple(sorted((root, cut['a'] - 1 - distance)
                                         for root, distance in sources[j].items()))
                    if all(firsts[root] <= limit for root, limit in pairs):
                        before = tuple((r, lim) for r, lim in pairs if lasts[r] > lim)
                if upper[j] >= cut['b']:
                    pairs = tuple(sorted((root, cut['b'] + distance)
                                         for root, distance in right[j].items()))
                    if all(lasts[root] >= limit for root, limit in pairs):
                        after = tuple((r, lim) for r, lim in pairs if firsts[r] < lim)
                if before is not None or after is not None:
                    profiles[before, after] += 8 if ir.ops[j].engine == 'valu' else 1
            terms = []
            exported = []
            hinted_escape=0
            for rank, ((before, after), weight) in enumerate(profiles.items()):
                flags = []
                chose=False
                for kind, pairs in (('before', before), ('after', after)):
                    if pairs is None:
                        continue
                    flag = model.new_bool_var(f'escape{cut_rank}_{rank}_{kind}')
                    for root, limit in pairs:
                        comparison = variables[root] <= limit if kind == 'before' else variables[root] >= limit
                        model.add(comparison).only_enforce_if(flag)
                    good=all(hint_times[root]<=limit if kind=='before' else hint_times[root]>=limit
                             for root,limit in pairs)
                    chosen=bool(good and not chose)
                    model.add_hint(flag,int(chosen))
                    if chosen:
                        hinted_escape+=weight
                        chose=True
                    flags.append(flag)
                model.add(sum(flags) <= 1)
                terms.append(weight * sum(flags))
                exported.append((weight, before, after))
            if args.soft_energy:
                shortfall=model.new_int_var(0,max(0,need),f'energy_shortfall{cut_rank}')
                model.add_hint(shortfall,max(0,need-hinted_escape))
                model.add(sum(terms)+shortfall>=need)
                shortfalls.append(shortfall)
            else:
                model.add(sum(terms) >= need)
            energy_groups.append((need, exported))
        print({'energy_cuts': len(energy_groups),
               'escape_profiles': sum(len(groups) for _, groups in energy_groups)}, flush=True)
    if args.export_model:
        ids = sorted(variables)
        remap = {j: k for k, j in enumerate(ids)}
        fixed_rows = [(op.engine, old[j]) for j, op in enumerate(ir.ops)
                      if op.engine in ('load', 'store', 'flow') and j not in variables]
        engine = {'load': 0, 'store': 1, 'flow': 2}
        with args.export_model.open('w') as stream:
            stream.write(f'{len(ids)} {len(projected_edges)} {len(fixed_rows)} {args.horizon}\n')
            for j in ids:
                stream.write(f'{engine[ir.ops[j].engine]} {old[j]} {firsts[j]} {lasts[j]} {j}\n')
            for d, j, lag in projected_edges:
                stream.write(f'{remap[d]} {remap[j]} {lag}\n')
            for e, t in fixed_rows:
                stream.write(f'{engine[e]} {t}\n')
            stream.write(str(len(energy_groups)) + '\n')
            for need, groups in energy_groups:
                stream.write(f'{need} {len(groups)}\n')
                for weight, before, after in groups:
                    stream.write(f'{weight} {len(before) if before is not None else -1} {len(after) if after is not None else -1}\n')
                    for pairs in (before, after):
                        if pairs is not None:
                            for root, limit in pairs:
                                stream.write(f'{remap[root]} {limit}\n')
    intervals = {e: [] for e in ('load', 'store', 'flow')}
    objective = []
    for j, op in enumerate(ir.ops):
        if op.engine not in intervals:
            continue
        value = variables.get(j, old[j])
        intervals[op.engine].append(model.new_fixed_size_interval_var(value, 1, f'i{j}'))
        if j in variables and not args.feasibility_only:
            difference = model.new_int_var(0, max(abs(firsts[j]-old[j]),abs(lasts[j]-old[j])), f'd{j}')
            model.add_abs_equality(difference, value - old[j])
            objective.append(difference * {'load': 4, 'flow': 2, 'store': 1}[op.engine])
    for engine, capacity in (('load', 2), ('store', 2), ('flow', 1)):
        model.add_cumulative(intervals[engine], [1] * len(intervals[engine]), capacity)
    if not args.feasibility_only:
        model.minimize(sum(objective)+1000000*sum(shortfalls))
    else:
        assert not args.soft_energy, 'Soft energy needs its optimization objective'
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.seconds
    solver.parameters.num_search_workers = args.workers
    solver.parameters.repair_hint = args.repair_hint
    solver.parameters.hint_conflict_limit = 100000
    print({'stream_variables': len(variables), 'projected_dependencies': projected,
           'horizon': args.horizon}, flush=True)
    status = solver.solve(model)
    feasible=status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    print({'status': solver.status_name(status), 'seconds': solver.wall_time,
           'movement_cost':solver.objective_value if feasible and not shortfalls and not args.feasibility_only else None,
           'objective':solver.objective_value if feasible and not args.feasibility_only else None}, flush=True)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return
    if shortfalls:
        print({'relaxed_energy_shortfall':sum(solver.value(v) for v in shortfalls),
               'scope':'selected necessary energy windows; not an executable score'},flush=True)
    result = list(old)
    for j, value in variables.items():
        result[j] = solver.value(value)
    use = Counter((op.engine, result[j]) for j, op in enumerate(ir.ops)
                  if op.engine in intervals)
    assert all(amount <= {'load': 2, 'store': 2, 'flow': 1}[engine]
               for (engine, _), amount in use.items())
    from codex.packet_stream_windows import bounds
    first,last,_,_=bounds(ir,result,args.horizon,args.compute_before)
    assert all(a<=b for a,b in zip(first,last)), 'Projected precedence audit failed'
    args.output.write_text(f'{n} {ng}\n' + ' '.join(map(str, result))
                          + '\n' + ' '.join(map(str, data[2 + n:])) + '\n')
    print({'moved_stream_ops': sum(result[j] != old[j] for j in variables),
           'relaxed_calendar_only': True, 'output': str(args.output)}, flush=True)


if __name__ == '__main__':
    main()
