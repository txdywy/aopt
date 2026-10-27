"""Roll a zero-slack 863 calendar window by window with CP-SAT.

The 51,612-lane scalar-chain graph meets every necessary 863 bound with no
slack: arithmetic must be 60 lanes in cycles 1..859, every LOAD slot in
0..841 and every FLOW slot in 0..852 must be used. This tool enforces those
densities exactly inside each window, plus per-node deadlines from dependency
tails, while moving operations within a radius of a template calendar.

The output calendar is research data until packet_modes check allocates it
and replays it on the frozen machine. Requires the research OR-Tools.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import pickle
import time

from ortools.sat.python import cp_model

CAPS = {'alu': 12, 'valu': 6, 'load': 2, 'store': 2, 'flow': 1}


class Model:
    def __init__(self, prefix, horizon):
        self.m = pickle.loads(Path(str(prefix) + '.pkl').read_bytes())
        m = self.m
        self.n = len(m.ops)
        self.engine = [op.engine for op in m.ops]
        self.member = {j: g for g, (_, _, ids, _) in enumerate(m.groups) for j in ids}
        self.horizon = horizon
        indeg = [len(d) for d in m.deps]
        queue = deque(i for i in range(self.n) if not indeg[i])
        topo = []
        while queue:
            i = queue.popleft()
            topo.append(i)
            for s, _ in m.succ[i]:
                indeg[s] -= 1
                if not indeg[s]:
                    queue.append(s)
        assert len(topo) == self.n
        self.topo = topo
        tail = [1] * self.n
        for i in reversed(topo):
            for s, lag in m.succ[i]:
                tail[i] = max(tail[i], lag + tail[s])
        est = [0] * self.n
        for i in topo:
            for d, lag in m.deps[i].items():
                est[i] = max(est[i], est[d] + lag)
        self.tail, self.est = tail, est
        self.deadline = [horizon - x for x in tail]
        self.load_last = max(self.deadline[j] for j in range(self.n)
                             if self.engine[j] == 'load' and self.tail[j] >= 22)
        self.flow_last = max(self.deadline[j] for j in range(self.n)
                             if self.engine[j] == 'flow')

    def usage(self, times, modes):
        use = Counter()
        for j in range(self.n):
            g = self.member.get(j)
            if g is not None and modes[g]:
                if j != self.m.groups[g][2][0]:
                    continue
                use['valu', times[j]] += 1
            else:
                use[self.engine[j], times[j]] += 1
        return use

    def audit(self, times, modes, upto=None):
        """Return dependency and capacity violations (optionally before a cycle)."""
        m = self.m
        bad_dep = 0
        for j in range(self.n):
            for d, lag in m.deps[j].items():
                if times[j] < times[d] + lag and (upto is None or min(times[j], times[d]) < upto):
                    bad_dep += 1
        use = self.usage(times, modes)
        over = Counter()
        for (e, c), v in use.items():
            if v > CAPS[e] and (upto is None or c < upto):
                over[e] += v - CAPS[e]
        native_split = 0
        for g, (_, _, ids, _) in enumerate(m.groups):
            if modes[g] and len({times[j] for j in ids}) != 1:
                native_split += 1
        return dict(dep=bad_dep, over=dict(over), native_split=native_split)

    def lanes(self, times, modes):
        use = self.usage(times, modes)
        out = Counter()
        for (e, c), v in use.items():
            if e == 'alu':
                out[c] += v
            elif e == 'valu':
                out[c] += 8 * v
        return out


def solve_window(M, times, modes, lo, target, hi, early, late, seconds, workers, seed,
                 load_spare, objective, log=False, dense=('alu', 'load', 'flow'),
                 future_ub=False):
    m = M.m
    n = M.n
    movable = [j for j in range(n) if lo <= times[j] < hi]
    mset = set(movable)
    # Native modes only for groups wholly inside the window.
    free_groups = []
    for g, (_, _, ids, _) in enumerate(m.groups):
        inside = [j in mset for j in ids]
        if all(inside):
            free_groups.append(g)
        elif any(inside) and modes[g]:
            raise RuntimeError(('native group straddles window', g))
    lowb, highb = {}, {}
    relaxed = 0
    for j in movable:
        lb = max(lo, M.est[j], min(times[j] - early, M.deadline[j] - early))
        ub = min(times[j] + late, M.deadline[j])
        for d, lag in m.deps[j].items():
            if d not in mset:
                lb = max(lb, times[d] + lag)
        lowb[j], highb[j] = lb, ub
    # Propagate bounds through movable dependencies (both directions).
    order = [j for j in M.topo if j in mset]
    for j in order:
        for d, lag in m.deps[j].items():
            if d in mset:
                lowb[j] = max(lowb[j], lowb[d] + lag)
    for j in reversed(order):
        for q, lag in m.succ[j]:
            if q in mset:
                highb[j] = min(highb[j], highb[q] - lag)
            elif times[q] >= hi and future_ub:
                limit = times[q] - lag
                if limit >= lowb[j]:
                    highb[j] = min(highb[j], limit)
                else:
                    relaxed += 1
    bad = [j for j in movable if lowb[j] > highb[j]]
    if bad:
        return dict(status='BOUNDS', bad=len(bad), sample=[(j, m.ops[j].engine, m.ops[j].tag,
                    times[j], lowb[j], highb[j], M.deadline[j]) for j in bad[:5]])
    model = cp_model.CpModel()
    x = {}
    for j in movable:
        for t in range(lowb[j], highb[j] + 1):
            x[j, t] = model.new_bool_var('')
    y = {}
    for g in free_groups:
        ids = m.groups[g][2]
        a, b = max(lowb[j] for j in ids), min(highb[j] for j in ids)
        for t in range(a, b + 1):
            y[g, t] = model.new_bool_var('')
    group_times = defaultdict(list)
    for (g, t) in y:
        group_times[g].append(t)

    def at_terms(j):
        terms = [(t, x[j, t]) for t in range(lowb[j], highb[j] + 1)]
        g = M.member.get(j)
        if g is not None and g in group_times:
            terms += [(t, y[g, t]) for t in group_times[g]]
        return terms

    T = {}
    for j in movable:
        terms = at_terms(j)
        model.add_exactly_one(v for _, v in terms)
        T[j] = model.new_int_var(lowb[j], highb[j], '')
        model.add(T[j] == sum(t * v for t, v in terms))
    for j in movable:
        for d, lag in m.deps[j].items():
            if d in mset:
                model.add(T[j] >= T[d] + lag)
    fixed = Counter()
    for j in range(n):
        if j in mset:
            continue
        g = M.member.get(j)
        if g is not None and modes[g]:
            if j != m.groups[g][2][0]:
                continue
            fixed['valu', times[j]] += 1
        else:
            fixed[M.engine[j], times[j]] += 1
    by = defaultdict(list)
    for (j, t), v in x.items():
        by[M.engine[j], t].append(v)
    for (g, t), v in y.items():
        by['valu', t].append(v)
    cycles = sorted({t for (_, t) in x} | {t for (_, t) in y})
    dense_set = set(dense)
    spare_vars = []
    for t in cycles:
        for e in CAPS:
            expr = sum(by.get((e, t), ()))
            cap = CAPS[e] - fixed[e, t]
            if isinstance(expr, int) and expr == 0:
                if cap < 0 and t < hi:
                    return dict(status='FIXED_OVER', cycle=t, engine=e)
                continue
            if t >= hi:
                model.add(expr <= max(0, cap))
                continue
            dense = False
            if lo <= t < target:
                if e in ('alu', 'valu') and 1 <= t <= M.horizon - 4 and 'alu' in dense_set:
                    dense = True
                if e == 'flow' and t <= M.flow_last and 'flow' in dense_set:
                    dense = True
            if e == 'load' and lo <= t < target and t <= M.load_last and 'load' in dense_set:
                if load_spare > 0:
                    s = model.new_int_var(0, 2, '')
                    model.add(expr + s == cap)
                    spare_vars.append(s)
                else:
                    model.add(expr == cap)
            elif dense:
                model.add(expr == cap)
            else:
                model.add(expr <= cap)
        if t == 0 and lo == 0 < target and 'alu' in dense_set:
            lanes0 = sum(by.get(('alu', 0), ())) + 8 * sum(by.get(('valu', 0), ()))
            model.add(lanes0 + fixed['alu', 0] + 8 * fixed['valu', 0] == 8)
    if spare_vars:
        model.add(sum(spare_vars) <= load_spare)
    for (j, t), v in x.items():
        g = M.member.get(j)
        model.add_hint(v, int(times[j] == t and not (g is not None and modes[g])))
    for (g, t), v in y.items():
        model.add_hint(v, int(bool(modes[g]) and times[m.groups[g][2][0]] == t))
    if objective == 'tail':
        model.minimize(sum(M.tail[j] * T[j] for j in movable))
    elif objective == 'deadline':
        # Pull the least-slack work earliest.
        model.minimize(sum(max(1, 64 - (M.deadline[j] - lo)) * T[j] for j in movable))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.log_search_progress = log
    status = solver.solve(model)
    name = solver.status_name(status)
    info = dict(status=name, seconds=round(solver.wall_time, 2), movable=len(movable),
                bools=len(x) + len(y), relaxed=relaxed)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return info
    new_times = list(times)
    new_modes = list(modes)
    for j in movable:
        new_times[j] = solver.value(T[j])
    for g in free_groups:
        new_modes[g] = int(any(solver.value(y[g, t]) for t in group_times.get(g, ())))
    used_spare = sum(solver.value(s) for s in spare_vars)
    pushed = 0
    for j in M.topo:
        if j in mset or new_times[j] < lo:
            continue
        need = max([new_times[j]] + [new_times[d] + lag for d, lag in m.deps[j].items()])
        if need != new_times[j]:
            new_times[j] = need
            pushed += 1
    for g, (_, _, ids, _) in enumerate(m.groups):
        if new_modes[g] and len({new_times[j] for j in ids}) != 1:
            new_modes[g] = 0
    info.update(spare_used=used_spare, pushed=pushed)
    return info, new_times, new_modes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix', required=True, help='Mode model prefix (.pkl)')
    ap.add_argument('--calendar', required=True, help='Pickle of (times, modes)')
    ap.add_argument('--output', required=True, help='Output pickle prefix')
    ap.add_argument('--horizon', type=int, default=863)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--stop', type=int, default=840)
    ap.add_argument('--step', type=int, default=12)
    ap.add_argument('--lookahead', type=int, default=12)
    ap.add_argument('--early', type=int, default=5)
    ap.add_argument('--late', type=int, default=3)
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--load-spare', type=int, default=0)
    ap.add_argument('--objective', choices=('none', 'tail', 'deadline'), default='none')
    ap.add_argument('--log', action='store_true')
    ap.add_argument('--dense', default='alu,load,flow',
                    help='Comma list of densities to enforce (alu, load, flow)')
    args = ap.parse_args()
    M = Model(args.prefix, args.horizon)
    times, modes = pickle.loads(Path(args.calendar).read_bytes())
    lo = args.start
    spare = args.load_spare
    history = []
    while lo < args.stop:
        target = min(args.stop, lo + args.step)
        hi = target + args.lookahead
        started = time.time()
        result = solve_window(M, times, modes, lo, target, hi, args.early, args.late,
                              args.seconds, args.workers, args.seed, spare, args.objective,
                              args.log, tuple(x for x in args.dense.split(',') if x))
        if isinstance(result, dict):
            print(json.dumps(dict(lo=lo, target=target, hi=hi, **result)), flush=True)
            break
        info, times, modes = result
        spare -= info.get('spare_used', 0)
        audit = M.audit(times, modes, upto=target)
        lanes = M.lanes(times, modes)
        waste = sum(60 - lanes[c] for c in range(max(1, lo), target))
        row = dict(lo=lo, target=target, hi=hi, wall=round(time.time() - started, 1),
                   waste=waste, audit=audit, spare_left=spare, **info)
        print(json.dumps(row), flush=True)
        history.append(row)
        Path(args.output + f'-{target}.pkl').write_bytes(pickle.dumps((times, modes)))
        lo = target
    Path(args.output + '.json').write_text(json.dumps(history, indent=1))


if __name__ == '__main__':
    main()
