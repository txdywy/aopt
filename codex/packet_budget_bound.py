"""Bound prefix/suffix arithmetic waste with freely split or merged lanes.

Uses optional OR-Tools for research only. The two boundary maxima are relaxed
independently, so their combination is a lower bound for this particular IR,
not a lower bound for every implementation of the kernel.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys


def lane_weight(ir, j, productive=False, needed=None):
    op = ir.ops[j]
    if op.engine not in ('alu', 'valu'):
        return 0
    if productive and op.engine == 'alu' and not any(v in needed for v in op.args[0]):
        return 0
    return 8 if op.engine == 'valu' else 1


def stream_bounds(ir):
    """Necessary Hall-window bounds for the three fixed instruction streams."""
    degree = list(map(len, ir.deps))
    ready = deque(j for j, n in enumerate(degree) if n == 0)
    order = []
    head = [0] * len(ir.ops)
    while ready:
        j = ready.popleft()
        order.append(j)
        for q, lag in ir.succ[j]:
            head[q] = max(head[q], head[j] + lag)
            degree[q] -= 1
            if degree[q] == 0:
                ready.append(q)
    assert len(order) == len(ir.ops)
    tail = [0] * len(ir.ops)
    for j in reversed(order):
        for q, lag in ir.succ[j]:
            tail[j] = max(tail[j], lag + tail[q])
    result = {}
    for engine, capacity in (('load', 2), ('store', 2), ('flow', 1)):
        ids = [j for j, op in enumerate(ir.ops) if op.engine == engine]
        aa = sorted({head[j] for j in ids}, reverse=True)
        bb = sorted({tail[j] for j in ids}, reverse=True)
        best = {'cycle_lower_bound': 0}
        for a in aa:
            histogram = Counter(tail[j] for j in ids if head[j] >= a)
            count = 0
            for b in bb:
                count += histogram[b]
                if not count:
                    continue
                bound = a + b + (count + capacity - 1) // capacity
                if bound > best['cycle_lower_bound']:
                    best = {'cycle_lower_bound': bound, 'prefix_at_least': a,
                        'suffix_at_least': b, 'instructions': count,
                        'capacity': capacity}
        result[engine] = best
    return result


def boundary(ir, width, reverse, seconds, workers, productive=False, witness=False,
             target_horizon=None, canonical_start=False):
    from ortools.sat.python import cp_model
    from codex.word_ssa import observed

    needed = observed(ir) if productive else None
    weights = [lane_weight(ir, j, productive, needed) for j in range(len(ir.ops))]

    deps = [dict(edges) for edges in ir.succ] if reverse else ir.deps
    succ = [list(edges.items()) for edges in ir.deps] if reverse else ir.succ
    degrees = list(map(len, deps))
    earliest = [0] * len(ir.ops)
    ready = deque(j for j, degree in enumerate(degrees) if degree == 0)
    seen = 0
    order = []
    while ready:
        j = ready.popleft()
        seen += 1
        order.append(j)
        for q, lag in succ[j]:
            earliest[q] = max(earliest[q], earliest[j] + lag)
            degrees[q] -= 1
            if degrees[q] == 0:
                ready.append(q)
    assert seen == len(ir.ops), 'boundary graph is cyclic'
    eligible = {j for j, first in enumerate(earliest) if first < width}
    model = cp_model.CpModel()
    single = {(j, t): model.NewBoolVar(f'x{j}_{t}') for j in eligible
              for t in range(earliest[j], width)}
    native = {}
    owner = {}
    for g, (_, _, ids, _) in enumerate(ir.groups):
        for j in ids:
            assert j not in owner, 'overlapping groups need a different model'
            owner[j] = g
        if all(j in eligible for j in ids):
            first = max(earliest[j] for j in ids)
            for t in range(first, width):
                native[g, t] = model.NewBoolVar(f'v{g}_{t}')
            model.Add(sum(native.get((g, t), 0) for t in range(width)) <= 1)

    def at(j, t):
        return single.get((j, t), 0) + native.get((owner.get(j, -1), t), 0)

    if canonical_start:
        assert not reverse and width >= 4
        for j, op in enumerate(ir.ops):
            pin = None
            if op.tag == ('input', (0,)) or op.tag == ('input', (1,)):
                pin = 1
            elif op.tag == ('tree', (0,)):
                pin = 3
            elif op.engine == 'load' and op.code == 'const' and op.args[1] == 2318:
                pin = 0
            elif op.engine == 'flow' and op.code == 'add_imm':
                value = ir.known.get(op.args[0][0])
                pin = {0x7ed55d16: 0, 0xc761c23c: 1,
                       2899272192: 2, 0xfd7046c5: 3, 3925396509: 4,
                       0xb55a4f09: op.time}.get(value)
            elif op.engine == 'valu' and op.code == 'vbroadcast':
                value = ir.known.get(op.args[0][0])
                if value in (16, 256, 2310, 0x7ed55d16) or (
                        value is None and op.tag == ('setup', ())):
                    pin = 1
            elif op.engine == 'valu' and op.code == 'multiply_add' and (
                    ir.known.get(op.args[0][0]) == 4097):
                pin = 2
            if pin is not None:
                model.Add(at(j, pin) == 1)
        for g, (_, args, ids, _) in enumerate(ir.groups):
            if all(ir.known.get(v) == 2 for v in args[0]):
                model.Add(native.get((g, 1), 0) == 1)

    for j in eligible:
        model.Add(sum(at(j, t) for t in range(width)) <= 1)
        for d, lag in deps[j].items():
            for t in range(earliest[j], width):
                model.Add(at(j, t) <= sum(at(d, s) for s in range(t - lag + 1)))
    limits = {'alu': 12, 'valu': 6, 'load': 2, 'store': 2, 'flow': 1}
    for t in range(width):
        for engine, limit in limits.items():
            demand = sum(single.get((j, t), 0) for j in eligible
                         if ir.ops[j].engine == engine)
            if engine == 'valu':
                demand += sum(v for (g, tt), v in native.items() if tt == t)
            model.Add(demand <= limit)
    if target_horizon is not None:
        assert not reverse
        tails = [0] * len(ir.ops)
        for j in reversed(order):
            for q, lag in ir.succ[j]:
                tails[j] = max(tails[j], lag + tails[q])
        for engine, capacity in (('load', 2), ('flow', 1), ('store', 2)):
            ids = [j for j, op in enumerate(ir.ops) if op.engine == engine]
            thresholds = set(tails[j] for j in ids)
            for k in range(1, width + 1):
                lower, b = max((sum(tails[j] >= b for j in ids)
                    - capacity * max(0, target_horizon - k - b), b)
                    for b in thresholds)
                if lower > 0:
                    model.Add(sum(at(j, t) for j in ids if j in eligible and tails[j] >= b
                                  for t in range(k)) >= lower)
    objective = sum(v * weights[j]
                    for (j, t), v in single.items() if ir.ops[j].engine in ('alu', 'valu'))
    objective += sum(v * sum(weights[j] for j in ir.groups[g][2])
                     for (g, t), v in native.items())
    if not reverse:
        # Before k cycles only nodes with ASAP < k can have executed. Grant
        # all such arithmetic for free and all later cycles full capacity.
        # In particular this exposes the c0 8-lane limit to the CP presolve.
        availability_upper = min(60 * (width - k) + sum(
            weights[j] for j in eligible if earliest[j] < k)
            for k in range(1, width + 1))
        model.Add(objective <= availability_upper)
    structural_upper = None
    if reverse:
        # Charge arithmetic with a single STORE descendant to that store;
        # grant every other eligible arithmetic node free placement once.
        # A store at t exposes only ancestors at distance < width-t.
        owners = {}
        roots = []
        for j in order:
            if j not in eligible:
                continue
            if not deps[j]:
                owners[j] = {j}
                roots.append(j)
            else:
                owners[j] = set().union(*(owners[d] for d in deps[j]))
        store_roots = {j for j in roots if ir.ops[j].engine == 'store'}
        if store_roots:
            rewards = {root: [0] * width for root in store_roots}
            free_work = 0
            for j in eligible:
                if ir.ops[j].engine not in ('alu', 'valu'):
                    continue
                weight = weights[j]
                if len(owners[j]) != 1 or not owners[j] <= store_roots:
                    # Unused scalar lanes and shared ancestors may all execute
                    # in the suffix in this relaxation; charge them only once.
                    free_work += weight
                    continue
                root = next(iter(owners[j]))
                for t in range(width - earliest[j]):
                    rewards[root][t] += weight
            structural_upper = free_work + sum(sum(sorted(
                (rewards[root][t] for root in store_roots), reverse=True)[:2])
                for t in range(width))
            model.Add(objective <= structural_upper)
    model.Maximize(objective)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = 1
    status = solver.Solve(model)
    assert status not in (cp_model.INFEASIBLE, cp_model.MODEL_INVALID), solver.StatusName(status)
    upper = min(60 * width, max(0, math.ceil(solver.BestObjectiveBound())))
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        upper = 60 * width
    if structural_upper is not None:
        upper = min(upper, structural_upper)
    result = {'side': 'suffix' if reverse else 'prefix', 'width': width,
            'status': solver.StatusName(status), 'eligible_ops': len(eligible),
            'objective_incumbent': solver.ObjectiveValue(),
            'objective_upper_bound': upper, 'waste_lower_bound': 60 * width - upper,
            'structural_upper_bound': structural_upper,
            'seconds': solver.WallTime()}
    if target_horizon is not None:
        result['conditional_horizon'] = target_horizon
    if canonical_start:
        result['canonical_start_constrained'] = True
    if witness and status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        times = {j: t for (j, t), var in single.items() if solver.Value(var)}
        modes = {}
        for (g, t), var in native.items():
            if solver.Value(var):
                modes[g] = 1
                times.update({j: t for j in ir.groups[g][2]})
        for j in times:
            if j in owner:
                modes.setdefault(owner[j], 0)
        result['witness'] = {'times': times, 'modes': modes}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model', type=Path)
    ap.add_argument('--prefix', type=int, default=6)
    ap.add_argument('--suffix', type=int, default=3)
    ap.add_argument('--seconds', type=float, default=30)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--ortools-path', type=Path)
    ap.add_argument('--productive', action='store_true',
        help='exclude independently schedulable unused scalar lane instructions')
    ap.add_argument('--prefix-witness', type=Path)
    ap.add_argument('--target-horizon', type=int,
                    help='require prefix stream throughput necessary at this horizon')
    ap.add_argument('--canonical-start', action='store_true',
                    help='constrain a specific startup; resulting bound has this extra scope')
    args = ap.parse_args()
    if args.ortools_path:
        sys.path.insert(0, str(args.ortools_path))
    with args.model.open('rb') as stream:
        ir = pickle.load(stream)
    counts = Counter(op.engine for op in ir.ops)
    if args.productive:
        from codex.word_ssa import observed
        needed = observed(ir)
        work = sum(lane_weight(ir, j, True, needed) for j in range(len(ir.ops)))
    else:
        work = counts['alu'] + 8 * counts['valu']
    head = boundary(ir, args.prefix, False, args.seconds, args.workers, args.productive,
                    args.prefix_witness is not None, args.target_horizon,
                    args.canonical_start)
    if args.prefix_witness and 'witness' in head:
        witness = head.pop('witness')
        with args.prefix_witness.open('w') as stream:
            stream.write(str(len(witness['times'])) + '\n')
            for j, t in sorted(witness['times'].items()):
                stream.write(f'{j} {t}\n')
            stream.write(str(len(witness['modes'])) + '\n')
            for g, md in sorted(witness['modes'].items()):
                stream.write(f'{g} {md}\n')
    print(json.dumps(head), flush=True)
    tail = boundary(ir, args.suffix, True, args.seconds, args.workers, args.productive)
    print(json.dumps(tail), flush=True)
    disjoint = math.ceil(work / 60) >= args.prefix + args.suffix
    bound = math.ceil((work + head['waste_lower_bound']
                       + tail['waste_lower_bound']) / 60) if disjoint else math.ceil(work / 60)
    if args.target_horizon:
        bound = min(bound, args.target_horizon + 1)
    model_text = args.model.with_suffix('.txt')
    streams = stream_bounds(ir)
    result = {'arithmetic_lanes': work, 'prefix': head, 'suffix': tail,
              'arithmetic_cycle_lower_bound': bound,
              'fixed_stream_bounds': streams,
              'cycle_lower_bound': max(bound, *(v['cycle_lower_bound'] for v in streams.values())),
              'boundary_windows_disjoint_by_compute_bound': disjoint,
              'model_file_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
              'model_text_sha256': (hashlib.sha256(model_text.read_bytes()).hexdigest()
                                    if model_text.is_file() else None),
              'scope': 'this IR and its exported scalar/native groups; no instruction elimination'}
    if args.canonical_start:
        result['scope'] += '; the specified canonical startup is fixed'
    result['independent_unused_scalar_lanes_relaxed'] = args.productive
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
