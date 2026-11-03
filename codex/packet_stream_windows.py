"""Audit arithmetic energy windows for a fixed LF calendar.

The caller supplies a proved suffix waste bound for this graph. Positive
excess rules out this calendar with the specified frozen compute prefix;
absence of excess does not certify an executable packet schedule.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from pathlib import Path
import pickle
import json


def bounds(ir, times, horizon, freeze_before):
    fixed = {j for j, op in enumerate(ir.ops)
             if op.engine not in ('alu', 'valu') or times[j] < freeze_before}
    first = [times[j] if j in fixed else freeze_before for j in range(len(ir.ops))]
    last = [times[j] if j in fixed else horizon - 1 for j in range(len(ir.ops))]
    degree = list(map(len, ir.deps))
    ready = deque(j for j, d in enumerate(degree) if not d)
    order = []
    while ready:
        j = ready.popleft()
        order.append(j)
        for q, lag in ir.succ[j]:
            first[q] = max(first[q], first[j] + lag)
            degree[q] -= 1
            if not degree[q]:
                ready.append(q)
    assert len(order) == len(ir.ops)
    for j in reversed(order):
        for q, lag in ir.succ[j]:
            last[j] = min(last[j], last[q] - lag)
    return first, last, fixed, order


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix', type=Path, required=True)
    ap.add_argument('--calendar', type=Path)
    ap.add_argument('--horizon', type=int, required=True)
    ap.add_argument('--freeze-before', type=int, default=600)
    ap.add_argument('--tail-waste', type=int, default=116)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--max-cuts', type=int, default=5)
    args = ap.parse_args()
    ir = pickle.loads(Path(str(args.prefix) + '.pkl').read_bytes())
    data = list(map(int, (args.calendar or Path(str(args.prefix) + '.sol')).read_text().split()))
    n, ng = data[:2]
    assert (n, ng) == (len(ir.ops), len(ir.groups))
    times = data[2:2 + n]
    first, last, fixed, _ = bounds(ir, times, args.horizon, args.freeze_before)
    bad = [j for j in range(n) if first[j] > last[j]]
    jobs = [j for j in range(n) if j not in fixed]
    weight = {j: 8 if ir.ops[j].engine == 'valu' else 1 for j in jobs}
    at_release = {}
    for j in jobs:
        at_release.setdefault(first[j], []).append(j)
    hist = Counter()
    violations = []
    for a in range(args.horizon - 1, args.freeze_before - 1, -1):
        for j in at_release.get(a, ()):
            hist[last[j] + 1] += weight[j]
        demand = 0
        for b in range(a + 1, args.horizon + 1):
            demand += hist[b]
            capacity = 60 * (b - a)
            if a == 0:
                capacity -= 52
            if b == args.horizon:
                capacity -= (args.tail_waste if b - a >= 3 else 104 if b - a == 2 else 60)
            excess = demand - capacity
            if excess > 0:
                violations.append(dict(a=a, b=b, demand=demand, capacity=capacity, excess=excess))
    cuts = []
    ranked = sorted(violations, key=lambda x: x['excess'], reverse=True)
    release = [item for item in ranked if item['b'] == args.horizon]
    forced_release = []
    for item in release:
        if all(abs(item['a'] - other['a']) >= 12 for other in forced_release):
            forced_release.append(item)
        if len(forced_release) == 2:
            break
    if release and ranked:
        ranked = [ranked[0], *forced_release] + ranked[1:]
    for item in ranked:
        if any(item['a'] == c['a'] and item['b'] == c['b'] for c in cuts):
            continue
        if not any(item is release_item for release_item in forced_release) and any(
                abs(item['a'] - c['a']) + abs(item['b'] - c['b']) < 12 for c in cuts):
            continue
        item['jobs'] = [j for j in jobs if first[j] >= item['a'] and last[j] < item['b']]
        cuts.append(item)
        if len(cuts) == args.max_cuts:
            break
    result = dict(horizon=args.horizon, freeze_before=args.freeze_before,
                  inconsistent_nodes=bad, positive_windows=len(violations), cuts=cuts,
                  supplied_tail_waste=args.tail_waste,
                  scope='fixed LF calendar and frozen compute prefix; necessary tests only')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print({k: v for k, v in result.items() if k != 'cuts'} | {
        'cuts': [{k: v for k, v in c.items() if k != 'jobs'} for c in cuts]}, flush=True)


if __name__ == '__main__':
    main()
