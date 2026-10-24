"""Reduce the complete two-op tail search with eight valid live inputs.

Congruence classes exclude 100 opcode pairs. A high-output identity test
handles any final SHR, division, ceil division, or remainder. These are
necessary conditions for a local rewrite, not a kernel optimality proof.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import time

import z3

from codex.search_full_live_shift863 import MASK, inverse_target, live_words, output_probes, target
from codex.search_hash_transport863 import OPS, operation
from codex.synthesize_hash_cegis import choose

COMPATIBLE = ('+', '-', '*', 'mad', '^', '&', '|')
SPLITTING = ('<<', '<', '==')
IDENTITY_HIGH = ('>>', '//', 'cdiv', '%')


def congruence_certificate():
    inverse = pow(4097, -1, 1 << 32)
    points = [((j*(1 << 16)-0x7ed55d16)*inverse)&MASK for j in range(8)]
    words = [live_words(p, 'tail') for p in points]
    lows = [[v & 0xffff for v in row] for row in words]
    assert all(row == lows[0] for row in lows)
    outputs = [target(p, 'tail') for p in points]
    assert len({y & 7 for y in outputs}) == 8
    return dict(prior_inputs=points, live_words=words, common_low16=lows[0],
        target_outputs=outputs, target_low3=[y & 7 for y in outputs],
        maximum_low16_outputs_after_two_instructions=4,
        compatible_opcodes=COMPATIBLE, splitting_opcodes=SPLITTING,
        excluded_sequences=[list(seq) for seq in itertools.product(COMPATIBLE+SPLITTING, repeat=2)])


def high_identity_lemmas():
    a, b = z3.BitVecs('high_numerator high_divisor', 32)
    rows = []
    for op in IDENTITY_HIGH:
        y, valid = operation(op, a, b, a)
        solver = z3.SolverFor('QF_BV')
        solver.set(timeout=10000)
        solver.add(valid, z3.UGT(y, 1 << 31), y != a)
        status = solver.check()
        row = dict(op=op, result=str(status), condition='output > 2**31 implies output == numerator')
        if status == z3.unknown:
            row['reason'] = solver.reason_unknown()
        rows.append(row)
    return rows


def high_points(scope='tail'):
    ys = sorted({y for y in output_probes() if y > 1 << 31}
        | {(1 << 31)+(1 << bit)+delta for bit in range(30) for delta in (0, 1)})
    return [inverse_target(y, scope) for y in ys]


def prefix_query(op, timeout_ms=5000, control=None):
    if op not in OPS or timeout_ms <= 0 or control not in (None, 'mad', 'identity'):
        raise ValueError('valid opcode, positive timeout and known control required')
    points = high_points('middle' if control == 'mad' else 'tail')
    if control == 'identity':
        # x % MASK is identity only below MASK. A remainder can never
        # produce MASK with a uint32 divisor, so do not use that as a
        # purported positive control.
        points = [(1 << 31)+1, MASK-1, 0xffffff00, 0x87654321]
    fn = (lambda p: target(p, 'middle')) if control == 'mad' else (
        (lambda p: p) if control == 'identity' else (lambda p: target(p, 'tail')))
    outputs = [fn(p) for p in points]
    assert all(y > 1 << 31 for y in outputs)
    a, b, c = z3.BitVecs('high_prefix_a high_prefix_b high_prefix_c', 4)
    k0, k1 = z3.BitVecs('high_prefix_k0 high_prefix_k1', 32)
    solver = z3.SolverFor('QF_BV')
    solver.set(timeout=timeout_ms)
    for selector in (a, b, c):
        solver.add(z3.ULT(selector, 10))
    if op in ('+', '*', 'mad', '^', '&', '|', '=='):
        solver.add(z3.ULE(a, b))
    if op != 'mad':
        solver.add(c == 0)
    for p, output in zip(points, outputs):
        pool = [z3.BitVecVal(v, 32) for v in live_words(p, 'tail')]+[k0, k1]
        av, bv, cv = [choose(s, pool) for s in (a, b, c)]
        if op in IDENTITY_HIGH:
            # Exact on this high-output domain, not a heuristic cut.
            solver.add(av == output)
            if op == '%':
                solver.add(z3.UGT(bv, av))
            else:
                solver.add(bv == (0 if op == '>>' else 1))
        else:
            y, valid = operation(op, av, bv, cv)
            solver.add(valid, y == output)
    started = time.monotonic()
    status = solver.check()
    row = dict(op=op, control=control, inputs=points, outputs=outputs,
        timeout_ms=timeout_ms, result=str(status), seconds=round(time.monotonic()-started, 3))
    if status == z3.unknown:
        row['reason'] = solver.reason_unknown()
    elif status == z3.sat:
        model = solver.model()
        row['program'] = dict(op=op, operands=[model.eval(v, model_completion=True).as_long()
            for v in (a, b, c)], constants=[model.eval(v, model_completion=True).as_long()
            for v in (k0, k1)])
        row['verification'] = 'necessary samples only'
    return row


def bypass_witnesses():
    points = high_points()
    outputs = [target(p, 'tail') for p in points]
    rows = []
    for source in range(8):
        p = next(p for p, y in zip(points, outputs) if live_words(p, 'tail')[source] != y)
        rows.append(dict(source=source, prior=p, live=live_words(p, 'tail')[source], target=target(p, 'tail')))
    assert len(set(outputs)) > 1
    return dict(live_sources=rows, distinct_constant_witnesses=[dict(prior=p, target=y)
        for p, y in list(zip(points, outputs))[:2]])


def coverage(report, searches=(), division_certificate=None):
    evidence = {seq: [] for seq in itertools.product(OPS, repeat=2)}
    for seq in report['congruence']['excluded_sequences']:
        evidence[tuple(seq)].append('at most four low16 outputs; target has eight')
    for seq in evidence:
        if seq[-1] in ('<', '=='):
            evidence[seq].append('boolean final output cannot equal the target permutation')
    lemmas = {row['op'] for row in report['identity_lemmas'] if row['result'] == 'unsat'}
    for prefix in report['prefixes']:
        if prefix['result'] == 'unsat':
            for last in lemmas:
                evidence[prefix['op'], last].append('high-output identity and excluded one-op prefix')
    for search in searches:
        assert search['scope'] == 'tail' and search['width'] == 32
        for row in search['rows']:
            if row['result'] == 'grammar_excluded' and row['fixed_shift'] is None and row['first_source'] is None:
                evidence[tuple(row['sequence'])].append('full32 necessary-sample UNSAT')
    if division_certificate is not None and division_certificate['excluded']:
        assert division_certificate['scope'] == 'tail' and division_certificate['width'] == 32
        assert division_certificate['low_bits'] == 16 and len(division_certificate['rows']) == 16
        assert all(row['excluded'] and row['positive_control']['retained']
                   for row in division_certificate['rows'])
        for first in ('//', '%', 'cdiv'):
            for last in ('+', '-', '*', 'mad'):
                evidence[first, last].append('complete nonzero uint32 divisor interval enumeration; '
                                             'add/subtract/multiply are subsets of a final MAD')
    rows = [dict(sequence=list(seq), excluded=bool(reasons), evidence=reasons)
            for seq, reasons in sorted(evidence.items())]
    return dict(total=len(rows), closed=sum(r['excluded'] for r in rows),
        unresolved=sum(not r['excluded'] for r in rows), orders=rows,
        scope='Two scalar-expression instructions from the 14-op ISA, with eight existing '
              'full32 live words and arbitrary constants. One original state determines all words.',
        new_executable=False, best_verified_cycles=864)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--searches', type=Path, nargs='*', default=[])
    ap.add_argument('--timeout-ms', type=int, default=5000)
    ap.add_argument('--division-certificate', type=Path)
    args = ap.parse_args()
    report = dict(scope='tail', width=32, congruence=congruence_certificate(), identity_lemmas=high_identity_lemmas(),
        bypass=bypass_witnesses(), prefixes=[], controls=[])
    for op, control in (('mad', 'mad'), ('>>', 'identity'), ('%', 'identity'), ('cdiv', 'identity')):
        report['controls'].append(prefix_query(op, max(args.timeout_ms, 10000), control))
    if any(row['result'] != 'sat' for row in report['controls']):
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        raise RuntimeError('A necessary-condition positive control failed; inspect the saved report')
    for op in OPS:
        row = prefix_query(op, args.timeout_ms)
        report['prefixes'].append(row)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k: row[k] for k in ('op', 'result', 'seconds')}), flush=True)
    division = json.loads(args.division_certificate.read_text()) if args.division_certificate else None
    report['coverage'] = coverage(report, [json.loads(p.read_text()) for p in args.searches], division)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report['coverage'][k] for k in ('total', 'closed', 'unresolved')}))


if __name__ == '__main__':
    main()
