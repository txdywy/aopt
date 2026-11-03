"""Jointly pack a compute prefix and repair the remaining LF calendar.

Later ALU/VALU resources are relaxed. Propagated release/deadline variables
and optional global energy cuts protect the suffix. Output is not a score.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import json
from pathlib import Path
import pickle

from ortools.sat.python import cp_model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--lo', type=int, required=True)
    ap.add_argument('--hi', type=int, required=True)
    ap.add_argument('--horizon', type=int, required=True)
    ap.add_argument('--radius', type=int, default=8)
    ap.add_argument('--stream-before', type=int,
                    help='Allow LF changes before lo while the compute prefix stays fixed')
    ap.add_argument('--max-waste', type=int, default=0)
    ap.add_argument('--energy-cuts', type=Path)
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--stream-encoding', choices=('interval', 'indexed'), default='interval',
                    help='Use per-cycle stream choices for tighter short-window capacity propagation')
    ap.add_argument('--movement-objective', action='store_true')
    ap.add_argument('--maximize', action='store_true',
                    help='Optimize prefix work before LF movement instead of imposing full density')
    args = ap.parse_args()
    assert 0 <= args.lo < args.hi < args.horizon
    ir = pickle.loads(Path(str(args.prefix) + '.pkl').read_bytes())
    data = list(map(int, Path(str(args.prefix) + '.sol').read_text().split()))
    n, ng = data[:2]
    assert (n, ng) == (len(ir.ops), len(ir.groups))
    old, modes = data[2:2 + n], data[2 + n:]
    compute = {j for j, op in enumerate(ir.ops) if op.engine in ('alu', 'valu')
               and old[j] >= args.lo}
    stream_before = args.lo if args.stream_before is None else args.stream_before
    streams = {j for j, op in enumerate(ir.ops) if op.engine in ('load', 'store', 'flow')
               and old[j] >= stream_before}
    first, last = list(old), list(old)
    for j in compute:
        first[j], last[j] = args.lo, args.horizon - 1
    for j in streams:
        first[j] = max(0, old[j] - args.radius)
        last[j] = (args.horizon - 1 if ir.ops[j].tag[0] == 'output'
                   else min(args.horizon - 1, old[j] + args.radius))
    degree = list(map(len, ir.deps))
    ready = deque(j for j, count in enumerate(degree) if not count)
    order = []
    while ready:
        j = ready.popleft()
        order.append(j)
        for q, lag in ir.succ[j]:
            first[q] = max(first[q], first[j] + lag)
            degree[q] -= 1
            if not degree[q]:
                ready.append(q)
    assert len(order) == n
    for j in reversed(order):
        for q, lag in ir.succ[j]:
            last[j] = min(last[j], last[q] - lag)
    bad = [j for j in range(n) if first[j] > last[j]]
    if bad:
        print({'status': 'PROJECTED_WINDOW_INFEASIBLE', 'nodes': bad[:10]}, flush=True)
        return
    model = cp_model.CpModel()
    early, late = list(old), list(old)
    for j in compute:
        early[j] = model.new_int_var(first[j], last[j], f'e{j}')
        late[j] = model.new_int_var(first[j], last[j], f'l{j}')
        model.add(early[j] <= late[j])
    for j in streams:
        early[j] = late[j] = model.new_int_var(first[j], last[j], f't{j}')
        model.add_hint(early[j], min(last[j], max(first[j], old[j])))
    for j in range(n):
        for d, lag in ir.deps[j].items():
            if j in compute or j in streams or d in compute or d in streams:
                model.add(early[j] >= early[d] + lag)
                model.add(late[j] >= late[d] + lag)
    chosen = {j for j in compute if first[j] < args.hi}
    member = {j: g for g, (_, _, ids, _) in enumerate(ir.groups) for j in ids}
    groups = {g for g, (_, _, ids, _) in enumerate(ir.groups) if set(ids) <= chosen}
    single = {(j, t): model.new_bool_var(f's{j}_{t}') for j in chosen
              for t in range(first[j], min(last[j], args.hi - 1) + 1)}
    native = {}
    for g in groups:
        ids = ir.groups[g][2]
        a, b = max(first[j] for j in ids), min(last[j] for j in ids)
        for t in range(a, min(b, args.hi - 1) + 1):
            native[g, t] = model.new_bool_var(f'v{g}_{t}')
        model.add(sum(native.get((g, t), 0) for t in range(a, min(b, args.hi - 1) + 1)) <= 1)

    def at(j, t):
        return single.get((j, t), 0) + native.get((member.get(j, -1), t), 0)

    for j in compute:
        if j not in chosen:
            model.add(early[j] >= args.hi)
            continue
        selected = model.new_bool_var(f'selected{j}')
        model.add(selected == sum(at(j, t)
            for t in range(first[j], min(last[j], args.hi - 1) + 1)))
        position = sum(t * at(j, t)
            for t in range(first[j], min(last[j], args.hi - 1) + 1))
        model.add(early[j] == position).only_enforce_if(selected)
        model.add(late[j] == position).only_enforce_if(selected)
        model.add(early[j] >= args.hi).only_enforce_if(selected.Not())
    by_time, vectors, work = {}, {}, []
    for (j, t), var in single.items():
        by_time.setdefault((ir.ops[j].engine, t), []).append(var)
        work.append(var * (8 if ir.ops[j].engine == 'valu' else 1))
        model.add_hint(var, int(old[j] == t and (member.get(j) not in groups
                                               or not modes[member[j]])))
    for (g, t), var in native.items():
        vectors.setdefault(t, []).append(var)
        work.append(8 * var)
        model.add_hint(var, int(bool(modes[g]) and old[ir.groups[g][2][0]] == t))
    for t in range(args.lo, args.hi):
        alu = sum(by_time.get(('alu', t), ()))
        valu = sum(by_time.get(('valu', t), ())) + sum(vectors.get(t, ()))
        model.add(alu <= 12)
        model.add(valu <= 6)
        if args.max_waste < 8 and not args.maximize:
            model.add(valu == 6)
            model.add(alu >= 12 - args.max_waste)
    if not args.maximize:
        model.add(sum(work) >= 60 * (args.hi - args.lo) - args.max_waste)
    if args.stream_encoding == 'indexed':
        fixed_stream_use = Counter((op.engine, old[j]) for j, op in enumerate(ir.ops)
                                   if op.engine in ('load', 'store', 'flow') and j not in streams)
        stream_choices = {}
        for j in streams:
            choices = {t: model.new_bool_var(f'm{j}_{t}') for t in range(first[j], last[j] + 1)}
            model.add(sum(choices.values()) == 1)
            model.add(early[j] == sum(t * var for t, var in choices.items()))
            for t, var in choices.items():
                stream_choices.setdefault((ir.ops[j].engine, t), []).append(var)
                model.add_hint(var, int(t == old[j]))
        for engine, capacity in (('load', 2), ('store', 2), ('flow', 1)):
            for t in range(args.horizon):
                model.add(sum(stream_choices.get((engine, t), ()))
                          <= capacity - fixed_stream_use[engine, t])
    else:
        for engine, capacity in (('load', 2), ('store', 2), ('flow', 1)):
            intervals = [model.new_fixed_size_interval_var(early[j], 1, f'i{j}')
                         for j, op in enumerate(ir.ops) if op.engine == engine]
            model.add_cumulative(intervals, [1] * len(intervals), capacity)
    if args.energy_cuts:
        cuts = json.loads(args.energy_cuts.read_text())
        assert cuts['horizon'] == args.horizon and cuts['freeze_before'] == args.hi
        total = sum(8 if ir.ops[j].engine == 'valu' else 1 for j in compute)
        for rank, cut in enumerate(cuts['cuts']):
            assert args.hi <= cut['a'] < cut['b'] <= args.horizon
            terms = []
            for j in compute:
                flags = []
                if first[j] < cut['a']:
                    before = model.new_bool_var(f'b{rank}_{j}')
                    model.add(early[j] < cut['a']).only_enforce_if(before)
                    flags.append(before)
                if last[j] >= cut['b']:
                    after = model.new_bool_var(f'a{rank}_{j}')
                    model.add(late[j] >= cut['b']).only_enforce_if(after)
                    flags.append(after)
                model.add(sum(flags) <= 1)
                terms.append(sum(flags) * (8 if ir.ops[j].engine == 'valu' else 1))
            model.add(sum(terms) >= total - cut['capacity'])
    movement = []
    if args.movement_objective:
        for j in streams:
            delta = model.new_int_var(0, args.horizon + max(old), f'd{j}')
            model.add_abs_equality(delta, early[j] - old[j])
            movement.append(delta)
    if args.maximize:
        model.maximize(1000000 * sum(work) - sum(movement))
    elif movement:
        model.minimize(sum(movement))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.seconds
    solver.parameters.num_search_workers = args.workers
    print({'prefix_window': [args.lo, args.hi], 'compute_choices': len(chosen),
           'stream_variables': len(streams), 'compute_booleans': len(single) + len(native),
           'stream_encoding': args.stream_encoding}, flush=True)
    status = solver.solve(model)
    print({'status': solver.status_name(status), 'seconds': solver.wall_time}, flush=True)
    if args.maximize:
        print({'weighted_objective_bound': solver.best_objective_bound,
               'scope': 'prefix work optimization with relaxed later compute'}, flush=True)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return
    result = [max(t, args.hi) if j in compute else t for j, t in enumerate(old)]
    scheduled = set()
    for j in streams:
        result[j] = solver.value(early[j])
    for j in chosen:
        for t in range(first[j], min(last[j], args.hi - 1) + 1):
            if solver.value(at(j, t)):
                result[j] = t
                scheduled.add(j)
                break
    for g, (_, _, ids, _) in enumerate(ir.groups):
        if scheduled.intersection(ids):
            modes[g] = int(any(solver.value(native[g, t])
                for t in range(args.lo, args.hi) if (g, t) in native))
    use = Counter()
    for j, op in enumerate(ir.ops):
        if op.engine in ('alu', 'valu') and result[j] >= args.hi:
            continue
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
    assert all(amount <= {'alu': 12, 'valu': 6, 'load': 2, 'store': 2, 'flow': 1}[engine]
               for (engine, _), amount in use.items())
    from codex.packet_stream_windows import bounds
    aa, bb, _, _ = bounds(ir, result, args.horizon, args.hi)
    assert all(a <= b for a, b in zip(aa, bb))
    args.output.write_text(f'{n} {ng}\n' + ' '.join(map(str, result))
                          + '\n' + ' '.join(map(str, modes)) + '\n')
    print({'output': str(args.output), 'compute_packed_before': args.hi,
           'prefix_lanes': solver.value(sum(work)),
           'relaxed_calendar_only': True}, flush=True)


if __name__ == '__main__':
    main()
