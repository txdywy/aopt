"""Recheck restore-search UNKNOWNs using independent CDCL SAT engines.

The named-Boolean CNF bridge follows the approach in Claude's
cnf_cegis863.py (commit 589ab06). Each candidate is evaluated on all 2**11
inputs; counterexamples rebuild the CNF. A per-case subprocess deadline
keeps external solvers bounded. Reduced SAT is never an executable result.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pysat
from pysat.solvers import Solver
import z3

from codex.cegis_c1_absorb863 import middle, run_numpy
from codex.search_cross_encoding863 import PRE_C5, necessary_inputs
from codex.search_restore_wirings863 import expression


def bitvec(name, width):
    bits = [z3.Bool(f'{name}_{i}') for i in range(width)]
    value = z3.Concat(*[z3.If(b, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)) for b in bits[::-1]])
    return value, bits


def cnf(assertions):
    goal = z3.Goal()
    goal.add(*assertions)
    result = z3.Then('simplify', 'bit-blast', 'tseitin-cnf')(goal)
    assert len(result) == 1
    atoms, names, clauses = {}, {}, []

    def literal(e):
        neg = z3.is_not(e)
        atom = e.arg(0) if neg else e
        assert z3.is_const(atom) and atom.sort().kind() == z3.Z3_BOOL_SORT
        if atom.get_id() not in atoms:
            atoms[atom.get_id()] = len(atoms)+1
            names[str(atom)] = atoms[atom.get_id()]
        value = atoms[atom.get_id()]
        return -value if neg else value

    for clause in result[0]:
        if z3.is_true(clause):
            continue
        if z3.is_false(clause):
            return None, names
        terms = list(clause.children()) if z3.is_or(clause) else [clause]
        if any(z3.is_true(v) for v in terms):
            continue
        terms = [v for v in terms if not z3.is_false(v)]
        clauses.append([literal(v) for v in terms])
    return clauses, names


def solve_case(case, engine, control=False):
    order, wiring = case['order'], case['wiring']
    width, mask = 11, 2047
    params = [[bitvec(f'k{i}_{j}', width) for j in range(2)] for i in range(4)]
    key, keybits = bitvec('input_key', width)
    constants = [[p[0] for p in kk] for kk in params]
    output_key = 0 if control else PRE_C5
    assertions = []
    if control:
        assertions += [constants[0][0] == 33, constants[1][0] == (16896 & mask), constants[3][0] == 9]
    points = list(dict.fromkeys(list(range(64)) + necessary_inputs(width)))

    def add(p):
        x = z3.BitVecVal(p, width)
        assertions.append(expression(order, wiring, constants, x) ==
                          (middle(x ^ key, mask) ^ (output_key & mask)))

    for p in points:
        add(p)
    started = time.monotonic()
    xs = np.arange(2048, dtype=np.uint64)
    rounds = 0
    while True:
        rounds += 1
        clauses, names = cnf(assertions)
        if clauses is None:
            return dict(status='unsat', engine=engine, width=width, rounds=rounds, samples=len(points),
                        how='simplification', seconds=time.monotonic()-started)
        with Solver(name=engine, bootstrap_with=clauses) as solver:
            sat = solver.solve()
            model = set(solver.get_model()) if sat else None
            try:
                stats = solver.accum_stats()
            except NotImplementedError:
                stats = None
        if not sat:
            return dict(status='unsat', engine=engine, width=width, rounds=rounds, samples=len(points),
                        clauses=len(clauses), stats=stats, seconds=time.monotonic()-started)

        def decode(bits):
            return sum(1 << i for i, b in enumerate(bits) if names.get(str(b)) in model)

        cv = [[decode(p[1]) for p in kk] for kk in params]
        iv = decode(keybits)
        got = run_numpy(order, wiring, cv, xs, mask)
        want = middle(xs ^ np.uint64(iv), np.uint64(mask)) ^ np.uint64(output_key & mask)
        bad = np.flatnonzero(got != want)
        if not len(bad):
            return dict(status='reduced_sat', engine=engine, width=width, rounds=rounds,
                        constants=cv, input_key=iv, samples=len(points), verified_inputs=2048,
                        clauses=len(clauses), seconds=time.monotonic()-started)
        for i in bad[:8]:
            p = int(xs[i])
            assert p not in points, 'CNF model decoding or encoding error'
            points.append(p)
            add(p)


def run_bounded(args):
    case, engine, seconds, control = args
    cmd = [sys.executable, '-m', 'codex.sat_restore_wirings863', '--single', json.dumps(case), '--engine', engine]
    if control:
        cmd += ['--control']
    started = time.monotonic()
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds, check=True)
        row = json.loads(out.stdout)
    except subprocess.TimeoutExpired:
        row = dict(status='unknown', reason='subprocess deadline', engine=engine)
    row.update(order=case['order'], wiring=case['wiring'], wall_seconds=round(time.monotonic()-started, 3))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--single')
    ap.add_argument('--engine', default='cadical195')
    ap.add_argument('--control', action='store_true')
    ap.add_argument('--input', type=Path)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--seconds', type=float, default=15)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    if args.single:
        print(json.dumps(solve_case(json.loads(args.single), args.engine, args.control)), flush=True)
        return
    assert args.input and args.output
    data = json.loads(args.input.read_text())
    cases = data['unresolved'][:args.limit or None]
    report = dict(engine=args.engine, pysat=pysat.__version__, z3=z3.get_version_string(),
                  source=str(args.input), jobs=len(cases), seconds_per_case=args.seconds,
                  rows=[], counts={}, new_executable=False)
    counts = Counter()
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_bounded, (c, args.engine, args.seconds, False)) for c in cases]
        for done, future in enumerate(as_completed(futures), 1):
            row = future.result()
            report['rows'].append(row)
            counts[row['status']] += 1
            report.update(counts=dict(counts), completed=done)
            if done % 10 == 0 or done == len(cases):
                args.output.write_text(json.dumps(report, indent=2)+'\n')
                print(json.dumps(dict(done=done, counts=dict(counts))), flush=True)


if __name__ == '__main__':
    main()
