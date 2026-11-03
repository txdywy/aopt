"""Pack a short compute prefix while leaving the later compute unscheduled.

LOAD/STORE/FLOW and the previous compute prefix are fixed. Later computation
is constrained by projected release/deadline bounds, but its resources are
relaxed. A saved calendar is therefore NOT an executable score.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle

from ortools.sat.python import cp_model

from codex.packet_stream_windows import bounds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--lo', type=int, required=True)
    ap.add_argument('--hi', type=int, required=True)
    ap.add_argument('--horizon', type=int, required=True)
    ap.add_argument('--max-waste', type=int, default=0)
    ap.add_argument('--seconds', type=float, default=30)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--backend', choices=('cp-sat', 'scip'), default='cp-sat')
    ap.add_argument('--maximize', action='store_true')
    ap.add_argument('--energy-cuts', type=Path,
                    help='Future-window cuts audited at a boundary between lo and hi')
    args = ap.parse_args()
    assert 0 <= args.lo < args.hi <= args.horizon
    ir = pickle.loads(Path(str(args.prefix) + '.pkl').read_bytes())
    data = list(map(int, Path(str(args.prefix) + '.sol').read_text().split()))
    n, ng = data[:2]
    assert (n, ng) == (len(ir.ops), len(ir.groups))
    old, modes = data[2:2 + n], data[2 + n:]
    first, last, fixed, _ = bounds(ir, old, args.horizon, args.lo)
    assert all(a <= b for a, b in zip(first, last))
    chosen = {j for j in range(n) if j not in fixed and first[j] < args.hi}
    member = {j: g for g, (_, _, ids, _) in enumerate(ir.groups) for j in ids}
    groups = {g for g, (_, _, ids, _) in enumerate(ir.groups) if set(ids) <= chosen}
    model = cp_model.CpModel()
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

    for j in chosen:
        issued = sum(at(j, t) for t in range(first[j], min(last[j], args.hi - 1) + 1))
        model.add(issued == 1 if last[j] < args.hi else issued <= 1)
        for d, lag in ir.deps[j].items():
            if d in fixed:
                continue
            for t in range(first[j], min(last[j], args.hi - 1) + 1):
                model.add(at(j, t) <= sum(at(d, s)
                    for s in range(first[d], min(last[d], t - lag) + 1)))
    by_time = {}
    objective = []
    for (j, t), var in single.items():
        by_time.setdefault((ir.ops[j].engine, t), []).append(var)
        objective.append(var * (8 if ir.ops[j].engine == 'valu' else 1))
        model.add_hint(var, int(old[j] == t and (member.get(j) not in groups
                                               or not modes[member[j]])))
    vectors = {}
    for (g, t), var in native.items():
        vectors.setdefault(t, []).append(var)
        objective.append(8 * var)
        model.add_hint(var, int(bool(modes[g]) and old[ir.groups[g][2][0]] == t))
    for t in range(args.lo, args.hi):
        alu = sum(by_time.get(('alu', t), ()))
        valu = sum(by_time.get(('valu', t), ())) + sum(vectors.get(t, ()))
        model.add(alu <= 12)
        model.add(valu <= 6)
        if not args.maximize and args.max_waste < 8:
            model.add(valu == 6)
            model.add(alu >= 12 - args.max_waste)
    work = sum(objective)
    if args.maximize:
        model.maximize(work)
    else:
        model.add(work >= 60 * (args.hi - args.lo) - args.max_waste)
    cuts = json.loads(args.energy_cuts.read_text()) if args.energy_cuts else None
    if cuts is not None:
        assert cuts['horizon'] == args.horizon
        assert args.lo <= cuts['freeze_before'] <= args.hi
    if cuts and cuts['cuts']:
        movable = set(range(n)) - fixed
        release = {j: model.new_int_var(first[j], last[j], f'release{j}') for j in movable}
        for j in movable:
            if j in chosen:
                selected = model.new_bool_var(f'selected{j}')
                model.add(selected == sum(at(j, t)
                    for t in range(first[j], min(last[j], args.hi - 1) + 1)))
                model.add(release[j] == sum(t * at(j, t)
                    for t in range(first[j], min(last[j], args.hi - 1) + 1))).only_enforce_if(selected)
                model.add(release[j] >= args.hi).only_enforce_if(selected.Not())
            else:
                model.add(release[j] >= args.hi)
            for d, lag in ir.deps[j].items():
                if d in movable:
                    model.add(release[j] >= release[d] + lag)
        for rank, cut in enumerate(cuts['cuts']):
            assert cuts['freeze_before'] <= cut['a'] < cut['b'] <= args.horizon
            due = [j for j in movable if last[j] < cut['b']]
            weight = {j: 8 if ir.ops[j].engine == 'valu' else 1 for j in due}
            need = sum(weight.values()) - cut['capacity']
            terms = []
            for j in due:
                if first[j] >= cut['a']:
                    continue
                before = model.new_bool_var(f'escape{rank}_{j}')
                model.add(release[j] < cut['a']).only_enforce_if(before)
                terms.append(weight[j] * before)
            model.add(sum(terms) >= need)
    if args.backend == 'scip':
        from codex.linear_scip import LinearScipSolver
        solver = LinearScipSolver(args.seconds, args.workers)
    else:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = args.seconds
        solver.parameters.num_search_workers = args.workers
    print({'prefix_window': [args.lo, args.hi], 'eligible_ops': len(chosen),
           'booleans': len(single) + len(native), 'backend': args.backend}, flush=True)
    status = solver.solve(model)
    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    print({'status': solver.status_name(status), 'seconds': solver.wall_time,
           'prefix_lanes': solver.value(work) if feasible else None,
           'prefix_upper_bound': solver.best_objective_bound if args.maximize else None}, flush=True)
    if not feasible:
        return
    result = [max(t, args.hi) if j not in fixed else t for j, t in enumerate(old)]
    scheduled = set()
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
    for j in scheduled:
        for d, lag in ir.deps[j].items():
            assert d in fixed or d in scheduled
            assert result[j] >= result[d] + lag
        g = member.get(j)
        if g is not None and modes[g]:
            ids = ir.groups[g][2]
            assert set(ids) <= scheduled and len({result[k] for k in ids}) == 1
            if j != ids[0]:
                continue
            engine = 'valu'
        else:
            engine = ir.ops[j].engine
        use[engine, result[j]] += 1
    assert all(amount <= {'alu': 12, 'valu': 6}[engine]
               for (engine, _), amount in use.items())
    after_first, after_last, _, _ = bounds(ir, result, args.horizon, args.hi)
    assert all(a <= b for a, b in zip(after_first, after_last))
    args.output.write_text(f'{n} {ng}\n' + ' '.join(map(str, result))
                          + '\n' + ' '.join(map(str, modes)) + '\n')
    print({'output': str(args.output), 'relaxed_calendar_only': True,
           'compute_packed_before': args.hi}, flush=True)


if __name__ == '__main__':
    main()
