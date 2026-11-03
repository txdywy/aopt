"""Constructive joint search: six middle/tail instructions into five.

This bounded grammar uses constant-factor MAD, XOR/AND/OR and a literal
right shift. It supports earlier real live words or direct uint32 b, finite
or arbitrary factors, and up to three arbitrary extra constants. UNKNOWN
is retained. Every proposed formula must pass a full uint32 counterexample
query before integration.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import z3

from codex.search_full_live_shift863 import (MASK, LIVE_NAMES, inverse_target,
    live_words, samples, target)
from codex.synthesize_hash_cegis import choose


FACTORS = (1, 9, 33, 512, 4097, 16896, MASK)
CONSTANTS = (0, 1, 2, 3, 4, 9, 16, 19, 33, 256, 512, 4097, 16896,
             0xe9f8cc1d, 0xaccf6200, 0xfd7046c5, 0xc761c23c,
             0x7ed55d16, 0xb55a4f09, 0xb55afa53, MASK)
OPS = ('mad', '^', '&', '|', '>>')
ORIGINAL = ('mad', 'mad', '^', 'mad', '>>', '^')
DIRECT_STAGES = {'b': 'first', 'bu': 'u', 'bv': 'v', 'bbranches': 'branches'}
DIRECT_LIVE = {'b': ['b'], 'bu': ['b', 'u'], 'bv': ['b', 'v'],
               'bbranches': ['b', 'u', 'v']}
PREFIX_COST = {'first': 0, 'u': 1, 'v': 1, 'branches': 2,
               'b': 0, 'bu': 1, 'bv': 1, 'bbranches': 2}


def factor_values(profile='base'):
    if profile == 'sparse':
        return (1, 9, 33, 16896)
    if profile == 'base':
        return FACTORS
    if profile != 'extended':
        raise ValueError('unknown constant profile')
    return tuple(dict.fromkeys(FACTORS+(297, 152064, 16929, 152361, 36873, 135201,
        65536, (-9)&MASK, (-18)&MASK, (-33)&MASK, (-512)&MASK,
        pow(9, -1, 1 << 32), pow(33, -1, 1 << 32), pow(4097, -1, 1 << 32))))


def constant_values(profile='base'):
    if profile == 'sparse':
        return (0, 1, MASK, 0xe9f8cc1d, 0xaccf6200, 0xfd7046c5)
    if profile == 'base':
        return CONSTANTS
    if profile != 'extended':
        raise ValueError('unknown constant profile')
    key = (0xaccf6200-512*0xe9f8cc1d)&MASK
    return tuple(dict.fromkeys(CONSTANTS+((9*0xe9f8cc1d+0xfd7046c5)&MASK,
        (9*0xaccf6200)&MASK, (9*(0xe9f8cc1d+0xaccf6200)+0xfd7046c5)&MASK,
        key, (9*key+0xfd7046c5)&MASK)))


def initial_words(prior, stage='first'):
    if stage not in PREFIX_COST:
        raise ValueError('unknown preserved prefix')
    if stage in DIRECT_STAGES:
        if isinstance(prior, int) and not 0 <= prior <= MASK:
            raise ValueError('b must be uint32')
        if not isinstance(prior, int) and (not z3.is_bv(prior) or prior.size() != 32):
            raise ValueError('symbolic b must have width 32')
        words = [prior, (33*prior+0xe9f8cc1d)&MASK,
                 (16896*prior+0xaccf6200)&MASK]
        return (words[:1] if stage == 'b' else words[:2] if stage == 'bu'
                else [words[0], words[2]] if stage == 'bv' else words)
    if stage == 'first':
        return live_words(prior, 'middle')
    words = live_words(prior, 'tail')
    if stage in ('u', 'v'):
        return words[:5]+[words[5 if stage == 'u' else 6]]
    return words[:7]


def sequences(stage='first'):
    if stage not in PREFIX_COST:
        raise ValueError('unknown preserved prefix')
    if stage not in ('first', 'b'):
        length = 5-PREFIX_COST[stage]
        return sorted(set(order for mad_count in range(length) for order in
            itertools.permutations(('mad',)*mad_count+('^',)*(length-1-mad_count)+('>>',))))
    return sorted(set(itertools.permutations(('mad', 'mad', 'mad', '^', '>>')))
                  | set(itertools.permutations(('mad', 'mad', '^', '^', '>>'))))


def execute(prior, program, stage='first', profile='base'):
    """Concrete interpreter for candidate filtering, independent of Z3 selectors."""
    values = initial_words(prior, stage)
    for row in program:
        pool = values+list(constant_values(profile))+row.get('extra_constants', [])
        a = pool[row['a']]
        if row['op'] == 'mad':
            out = a*row['factor']+pool[row['c']]
        elif row['op'] == '>>':
            out = a >> row.get('amount', 16)
        else:
            b = pool[row['b']]
            out = {'^': lambda: a ^ b, '&': lambda: a & b, '|': lambda: a | b}[row['op']]()
        values.append(out & MASK)
    return values[-1]


def expression(prior, program, stage='first', profile='base'):
    values = initial_words(prior, stage)
    for row in program:
        pool = values+[z3.BitVecVal(k, 32) for k in
                       (*constant_values(profile), *row.get('extra_constants', []))]
        a = pool[row['a']]
        if row['op'] == 'mad':
            out = a*row['factor']+pool[row['c']]
        elif row['op'] == '>>':
            out = z3.LShR(a, row.get('amount', 16))
        else:
            b = pool[row['b']]
            out = {'^': lambda: a ^ b, '&': lambda: a & b, '|': lambda: a | b}[row['op']]()
        values.append(out)
    return values[-1]


def original_program(stage='first', profile='base'):
    count = len(initial_words(0, stage))
    base = DIRECT_STAGES.get(stage, stage)
    b = 0 if stage in DIRECT_STAGES else 4
    program = []

    def append(op, **kwargs):
        dest = count+len(program)
        program.append(dict(op=op, **kwargs))
        return dest

    def mad(source, factor, addend):
        const = count+len(program)+constant_values(profile).index(addend)
        return append('mad', a=source, factor=factor, c=const)

    if base == 'first':
        u, v = mad(b, 33, 0xe9f8cc1d), mad(b, 16896, 0xaccf6200)
    elif base in ('u', 'v'):
        known = count-1
        other = mad(b, 16896 if base == 'u' else 33,
                    0xaccf6200 if base == 'u' else 0xe9f8cc1d)
        u, v = sorted((known, other))
    else:
        u, v = count-2, count-1
    z = append('^', a=u, b=v)
    e = mad(z, 9, 0xfd7046c5)
    shifted = append('>>', a=e)
    append('^', a=e, b=shifted)
    return program


def reference_value(value, stage='first'):
    """Direct-b modes range over all uint32 b; no low-bit truncation."""
    if stage not in DIRECT_STAGES:
        return target(value, 'tail')
    u, v = (33*value+0xe9f8cc1d)&MASK, (16896*value+0xaccf6200)&MASK
    e = (9*(u ^ v)+0xfd7046c5)&MASK
    return e ^ (e >> 16 if isinstance(e, int) else z3.LShR(e, 16))


def search_samples(stage='first'):
    original = samples('tail', True)
    if stage not in DIRECT_STAGES:
        return original
    return list(dict.fromkeys([0, 1, MASK, 1 << 31]
        + [1 << bit for bit in range(32)]
        + [(1 << bit)-1 for bit in range(1, 33)]
        + [live_words(p)[4] for p in original]))


def search(sequence, seconds=8, check_ms=5000, *, pin_original=False, control=False,
           stage='first', profile='base', shift_amount=16, free_constants=0,
           free_factors=False):
    if not 1 <= len(sequence) <= 6 or any(op not in OPS for op in sequence):
        raise ValueError('one to six supported instructions required')
    if seconds <= 0 or check_ms <= 0:
        raise ValueError('positive time limits required')
    if not 0 <= shift_amount <= 32:
        raise ValueError('literal right shift must be 0..32')
    if not isinstance(free_constants, int) or not 0 <= free_constants <= 3:
        raise ValueError('zero to three arbitrary shared constants required')
    count = len(initial_words(0, stage))
    constants, factors = constant_values(profile), factor_values(profile)
    witness = original_program(stage, profile)
    if pin_original and (tuple(sequence) != tuple(row['op'] for row in witness) or control or shift_amount != 16):
        raise ValueError('pinned original requires the original six-op target')
    if control and tuple(sequence) != ('mad', '^'):
        raise ValueError('automatic control uses MAD,XOR')
    solver = z3.SolverFor('QF_BV')
    extra = [z3.BitVec(f'joint_extra{i}', 32) for i in range(free_constants)]
    spec = []
    for i, op in enumerate(sequence):
        a, b, c = z3.BitVecs(f'joint_a{i} joint_b{i} joint_c{i}', 6)
        factor = z3.BitVec(f'joint_factor{i}', 32 if free_factors else 5)
        limit = count+i+len(constants)+len(extra)
        for selector in (a, b, c):
            solver.add(z3.ULT(selector, limit))
        if op == 'mad':
            solver.add(b == 0)
            if not free_factors:
                solver.add(z3.ULT(factor, len(factors)))
        elif op == '>>':
            solver.add(b == 0, c == 0, factor == 0)
        else:
            solver.add(z3.ULE(a, b), c == 0, factor == 0)
        spec.append((op, a, b, c, factor))
    # Constructive cut: each nonfinal new value has a later consumer.
    # The finite grammar and this cut are recorded; no universal lower
    # bound is inferred if a query is UNSAT.
    for i in range(len(sequence)-1):
        readers = []
        for op, a, b, c, factor in spec[i+1:]:
            readers += [a == count+i]
            if op == 'mad':
                readers += [c == count+i]
            elif op != '>>':
                readers += [b == count+i]
        solver.add(z3.Or(*readers))
    if pin_original:
        for row, fixed in zip(spec, witness):
            op, a, b, c, factor = row
            solver.add(a == fixed['a'])
            if op == 'mad':
                solver.add(c == fixed['c'], factor == (fixed['factor'] if free_factors
                                                       else factors.index(fixed['factor'])))
            elif op != '>>':
                solver.add(b == fixed['b'])

    def program(prior):
        values = initial_words(prior, stage)
        for op, a, b, c, factor in spec:
            pool = values+[z3.BitVecVal(k, 32) for k in constants]+extra
            av = choose(a, pool)
            if op == 'mad':
                out = (av*factor if free_factors else choose(factor, [av*k for k in factors]))+choose(c, pool)
            elif op == '>>':
                out = z3.LShR(av, shift_amount)
            else:
                bv = choose(b, pool)
                out = {'^': lambda: av ^ bv, '&': lambda: av & bv, '|': lambda: av | bv}[op]()
            values.append(out)
        return z3.simplify(values[-1])

    def reference(prior):
        if control:
            b = initial_words(prior, stage)[0 if stage in DIRECT_STAGES else 4]
            return ((33*b+0xe9f8cc1d)&MASK) ^ b
        return reference_value(prior, stage)

    points = list(dict.fromkeys([0, 1, MASK, 1 << 31,
        inverse_target(0, 'tail'), inverse_target(MASK, 'tail')]))
    if stage in DIRECT_STAGES:
        points = list(dict.fromkeys([0, 1, MASK, 1 << 31]
            + [live_words(p)[4] for p in points[-2:]]
            # These eight inputs have the same low 16 bits but distinct
            # low target bits. They expose the required high-to-low path
            # before a weak, six-sample selector query consumes its budget.
            + [(j << 16) ^ 0xc761c23c for j in range(8)]))
    rng = random.Random(879)
    checks = list(dict.fromkeys(search_samples(stage)+[rng.getrandbits(32) for _ in range(128)]))
    expected = {p: reference(p) for p in checks}
    for p in points:
        x = z3.BitVecVal(p, 32)
        solver.add(program(x) == reference(x))
    started = time.monotonic()
    deadline = started+seconds
    report = dict(sequence=list(sequence), seconds_limit=seconds, check_ms=check_ms,
        pin_original=pin_original, control=control, width=32,
        stage=stage, preserved_prefix_instructions=PREFIX_COST[stage],
        total_instructions=PREFIX_COST[stage]+len(sequence),
        independent_input='b' if stage in DIRECT_STAGES else 'prior',
        live_pool=(DIRECT_LIVE[stage]
                   if stage in DIRECT_STAGES else list(LIVE_NAMES[:5])+(['u' if stage == 'u' else 'v'] if stage in ('u','v')
                                      else list(LIVE_NAMES[5:7]) if stage == 'branches' else [])),
        profile=profile, factors=factors, constants=constants,
        free_constant_count=free_constants, arbitrary_mad_factors=free_factors,
        factor_domain='arbitrary_uint32_constant' if free_factors else 'listed_factors',
        literal_shift=shift_amount, every_nonfinal_value_consumed=True,
        initial_samples=points[:], concrete_validation_inputs=checks,
        result='deadline', events=[], new_executable=False)
    while time.monotonic() < deadline:
        solver.set(timeout=min(check_ms, max(1, int((deadline-time.monotonic())*1000))))
        status = solver.check()
        event = dict(samples=len(points), synthesis=str(status))
        report['events'].append(event)
        if status != z3.sat:
            report['result'] = 'finite_grammar_unsat' if status == z3.unsat else 'unknown'
            if status == z3.unknown:
                report['reason'] = solver.reason_unknown()
            break
        model = solver.model()
        extra_values = [model.eval(k, model_completion=True).as_long() for k in extra]
        candidate = []
        for op, a, b, c, factor in spec:
            row = dict(op=op, a=model.eval(a, model_completion=True).as_long())
            if op == 'mad':
                row.update(c=model.eval(c, model_completion=True).as_long(),
                           factor=(model.eval(factor, model_completion=True).as_long() if free_factors
                                   else factors[model.eval(factor, model_completion=True).as_long()]))
            elif op != '>>':
                row['b'] = model.eval(b, model_completion=True).as_long()
            else:
                row['amount'] = shift_amount
            if extra:
                row['extra_constants'] = extra_values
            candidate.append(row)
        report['candidate'] = candidate
        bad = [p for p in checks if execute(p, candidate, stage, profile) != expected[p]][:4]
        if bad:
            event['concrete_counterexamples'] = bad
            for p in bad:
                assert p not in points
                points.append(p)
                x = z3.BitVecVal(p, 32)
                solver.add(program(x) == reference(x))
            continue
        event['concrete_validation_passed'] = len(checks)
        x = z3.BitVec('joint_verify_prior', 32)
        verify = z3.SolverFor('QF_BV')
        verify.set(timeout=min(check_ms, max(1, int((deadline-time.monotonic())*1000))))
        verify.add(expression(x, candidate, stage, profile) != reference(x))
        checked = verify.check()
        event['full32_verification'] = str(checked)
        if checked == z3.unsat:
            report['result'] = 'proved_equivalent_full32'
            break
        if checked == z3.unknown:
            report.update(result='verification_unknown', reason=verify.reason_unknown())
            break
        p = verify.model().eval(x, model_completion=True).as_long()
        event['symbolic_counterexample'] = p
        points.append(p)
        x = z3.BitVecVal(p, 32)
        solver.add(program(x) == reference(x))
    report['seconds'] = round(time.monotonic()-started, 3)
    return report


def run(sequence, seconds, check_ms, stage='first', profile='base', shift_amount=16,
        free_constants=0, free_factors=False):
    cmd = [sys.executable, '-m', 'codex.search_joint_tail863', '--query',
           '--sequence='+','.join(sequence), '--seconds', str(seconds), '--check-ms', str(check_ms),
           '--stage', stage, '--profile', profile, '--shift', str(shift_amount),
           '--free-constants', str(free_constants)]
    if free_factors:
        cmd.append('--free-factors')
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=max(40, seconds*3+15))
    if proc.returncode:
        return dict(sequence=list(sequence), result='process_error', stderr=proc.stderr, command=cmd)
    row = json.loads(proc.stdout)
    row['command'] = cmd
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--query', action='store_true')
    ap.add_argument('--sequence')
    ap.add_argument('--seconds', type=float, default=8)
    ap.add_argument('--check-ms', type=int, default=5000)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--workers', type=int, choices=(1, 2), default=2)
    ap.add_argument('--stage', choices=tuple(PREFIX_COST), default='first')
    ap.add_argument('--profile', choices=('sparse', 'base', 'extended'), default='base')
    ap.add_argument('--shift', type=int, default=16)
    ap.add_argument('--free-constants', type=int, default=0)
    ap.add_argument('--free-factors', action='store_true')
    args = ap.parse_args()
    if args.query:
        if not args.sequence:
            ap.error('query requires sequence')
        print(json.dumps(search(args.sequence.split(','), args.seconds, args.check_ms,
                                stage=args.stage, profile=args.profile, shift_amount=args.shift,
                                free_constants=args.free_constants, free_factors=args.free_factors)))
        return
    if not args.output:
        ap.error('scan requires output')
    witness_order = tuple(row['op'] for row in original_program(args.stage, args.profile))
    options = dict(stage=args.stage, profile=args.profile,
                   free_constants=args.free_constants, free_factors=args.free_factors)
    controls = [search(witness_order, 10, 9000, pin_original=True, **options),
                search(('mad', '^'), 10, 9000, control=True, **options)]
    if any(row['result'] != 'proved_equivalent_full32' for row in controls):
        args.output.write_text(json.dumps(dict(controls=controls), indent=2)+'\n')
        raise RuntimeError('A full32 control failed; inspect the saved result')
    report = dict(scope='Constructive bounded constant-factor MAD/XOR/literal-SHR grammar; '
        'each row records its complete live-input, factor and constant scope; '
        'five total instructions with no unused nonfinal result',
        stage=args.stage, profile=args.profile, literal_shift=args.shift,
        free_constant_count=args.free_constants, arbitrary_mad_factors=args.free_factors,
        controls=controls, rows=[], new_executable=False)
    orders = sequences(args.stage) if not args.sequence else [tuple(args.sequence.split(','))]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run, order, args.seconds, args.check_ms, args.stage, args.profile,
                            args.shift, args.free_constants, args.free_factors)
                for order in orders]
        for job in as_completed(jobs):
            row = job.result()
            report['rows'].append(row)
            report['rows'].sort(key=lambda r: r['sequence'])
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({k: row.get(k) for k in ('sequence', 'result', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
