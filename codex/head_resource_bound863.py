"""Maximum arithmetic lanes in the first K cycles with every engine modeled.

Unlike the relaxed ASAP prefix bound, this CP model keeps ALU, VALU, LOAD,
STORE and FLOW capacities, native/scalar group modes and all dependency lags.
An operation either runs inside [0, K) after all its predecessors or is
deferred. The optimum is an upper bound on any calendar's first K cycles.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from ortools.sat.python import cp_model

from codex.zero_slack_roll863 import CAPS, Model


def bound(M, K, seconds, workers):
    m = M.m
    cand = [j for j in range(M.n) if M.est[j] < K]
    cset = set(cand)
    model = cp_model.CpModel()
    x = {}
    for j in cand:
        for t in range(M.est[j], K):
            x[j, t] = model.new_bool_var('')
    groups = [g for g, (_, _, ids, _) in enumerate(m.groups) if all(j in cset for j in ids)]
    y = {}
    for g in groups:
        ids = m.groups[g][2]
        for t in range(max(M.est[j] for j in ids), K):
            y[g, t] = model.new_bool_var('')
    gt = defaultdict(list)
    for g, t in y:
        gt[g].append(t)
    inside = {}
    T = {}
    for j in cand:
        terms = [(t, x[j, t]) for t in range(M.est[j], K)]
        g = M.member.get(j)
        if g in gt:
            terms += [(t, y[g, t]) for t in gt[g]]
        inside[j] = model.new_bool_var('')
        model.add(sum(v for _, v in terms) == inside[j])
        T[j] = model.new_int_var(M.est[j], K, '')
        model.add(T[j] == sum(t * v for t, v in terms) + K * (1 - inside[j]))
    for j in cand:
        for d, lag in m.deps[j].items():
            if d in cset:
                model.add(T[j] >= T[d] + lag).only_enforce_if(inside[j])
            else:
                model.add(inside[j] == 0)
    by = defaultdict(list)
    for (j, t), v in x.items():
        by[M.engine[j], t].append(v)
    for (g, t), v in y.items():
        by['valu', t].append(v)
    for t in range(K):
        for e, cap in CAPS.items():
            terms = by.get((e, t))
            if terms:
                model.add(sum(terms) <= cap)
    lanes = []
    for (j, t), v in x.items():
        if M.engine[j] == 'alu':
            lanes.append(v)
        elif M.engine[j] == 'valu':
            lanes.append(8 * v)
    lanes += [8 * v for v in y.values()]
    model.maximize(sum(lanes))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = workers
    status = solver.solve(model)
    return dict(K=K, status=solver.status_name(status), lanes=solver.objective_value,
                upper=solver.best_objective_bound, capacity=8 + 60 * (K - 1),
                candidates=len(cand), seconds=round(solver.wall_time, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--ks', default='4,6,8,12,16,24')
    ap.add_argument('--seconds', type=float, default=120)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    M = Model(args.prefix, 863)
    rows = []
    for K in map(int, args.ks.split(',')):
        row = bound(M, K, args.seconds, args.workers)
        row['waste_lower_bound'] = row['capacity'] - int(row['upper'])
        print(json.dumps(row), flush=True)
        rows.append(row)
    if args.output:
        args.output.write_text(json.dumps(rows, indent=1) + '\n')


if __name__ == '__main__':
    main()
