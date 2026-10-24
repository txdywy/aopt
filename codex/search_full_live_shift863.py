"""Full32 CEGIS with live values from earlier stages of the real hash.

Unlike the earlier restricted-domain screen, no word is truncated before
SHR. The original state is the sole independent variable; every free live
word is derived from that same state. Only a full32 counterexample UNSAT
can certify a replacement. Results do not imply a schedulable kernel.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import permutations
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import z3

from codex.search_hash_transport863 import OPS, operation
from codex.synthesize_hash_cegis import choose


MASK = (1 << 32)-1
LIVE_NAMES = ('prior', 'a', 'a_shr19', 'a_xor_c1', 'b', 'u', 'v', 'z')


def live_words(prior, scope='middle'):
    if scope not in ('middle', 'tail'):
        raise ValueError('scope must be middle or tail')
    if isinstance(prior, int):
        if not 0 <= prior <= MASK:
            raise ValueError('prior must be uint32')
        a = (4097*prior+0x7ed55d16)&MASK
        s = a >> 19
    else:
        if not z3.is_bv(prior) or prior.size() != 32:
            raise ValueError('symbolic prior must have width 32')
        a = 4097*prior+0x7ed55d16
        s = z3.LShR(a, 19)
    mix = a ^ 0xc761c23c
    b = mix ^ s
    words = [prior, a, s, mix, b]
    if scope == 'tail':
        u = (33*b+0xe9f8cc1d)&MASK
        v = (16896*b+0xaccf6200)&MASK
        words += [u, v, u ^ v]
    return words


def target(prior, scope='middle'):
    words = live_words(prior, 'tail')
    e = (9*words[-1]+0xfd7046c5)&MASK
    if scope == 'middle':
        return e
    if scope != 'tail':
        raise ValueError('scope must be middle or tail')
    shifted = e >> 16 if isinstance(e, int) else z3.LShR(e, 16)
    return e ^ shifted


def inverse_target(value, scope='middle'):
    """Invert the target permutation to create exact, output-directed samples."""
    if scope not in ('middle', 'tail') or not isinstance(value, int) or not 0 <= value <= MASK:
        raise ValueError('uint32 output and a valid scope required')
    e = value ^ (value >> 16) if scope == 'tail' else value
    z = ((e-0xfd7046c5)*pow(9, -1, 1 << 32))&MASK
    key = (0xaccf6200-512*0xe9f8cc1d)&MASK
    u = 0
    for _ in range(4):
        u = z ^ ((512*u+key)&MASK)
    b = ((u-0xe9f8cc1d)*pow(33, -1, 1 << 32))&MASK
    return prior_for_b(b)


def prior_for_b(b):
    if not isinstance(b, int) or not 0 <= b <= MASK:
        raise ValueError('b must be uint32')
    folded = b ^ 0xc761c23c
    a = folded ^ (folded >> 19)
    return ((a-0x7ed55d16)*pow(4097, -1, 1 << 32))&MASK


def zero_witnesses(scope='tail'):
    """Every existing live word can be zero on a valid original input."""
    inv4097 = pow(4097, -1, 1 << 32)
    from_a = lambda a: ((a-0x7ed55d16)*inv4097)&MASK
    key = (0xaccf6200-512*0xe9f8cc1d)&MASK
    z_zero_u = (key*pow(1-512, -1, 1 << 32))&MASK
    points = [0, from_a(0), from_a(0), from_a(0xc761c23c), prior_for_b(0),
        prior_for_b((-0xe9f8cc1d*pow(33, -1, 1 << 32))&MASK),
        prior_for_b((-(0xaccf6200//512)*pow(33, -1, 1 << 23))&((1 << 23)-1)),
        prior_for_b(((z_zero_u-0xe9f8cc1d)*pow(33, -1, 1 << 32))&MASK)]
    count = len(live_words(0, scope))
    rows = []
    for i, p in enumerate(points[:count]):
        assert live_words(p, scope)[i] == 0
        rows.append(dict(source=i, name=LIVE_NAMES[i], prior=p))
    return rows


def output_probes():
    return list(dict.fromkeys([*range(16), MASK, MASK-1, 1 << 31, (1 << 31)-1]
        + [1 << bit for bit in range(32)]
        + [(1 << bit)-1 for bit in range(1, 33)]
        + [((1 << bit)+1)&MASK for bit in range(1, 32)]))


def samples(scope='middle', output_preimages=False):
    rng = random.Random(872)
    inv = pow(4097, -1, 1 << 32)
    aa = [*range(16), 32, 40, 256, 264, 288, 296, 2047, 2048,
          (1 << 19)-1, 1 << 19, (1 << 19)+1, MASK]
    points = list(dict.fromkeys([0, 1, MASK, 1 << 31]
        + [1 << bit for bit in range(32)]
        + [rng.getrandbits(32) for _ in range(16)]
        + [((a-0x7ed55d16)*inv)&MASK for a in aa]))
    if output_preimages:
        points = list(dict.fromkeys(points+[inverse_target(y, scope) for y in output_probes()]
                                   +[row['prior'] for row in zero_witnesses(scope)]))
    return points


def sequences(scope):
    if scope == 'middle':
        return sorted(set(p for op in ('+', '-', '*', 'mad', '^', '&', '|', '<<', '>>')
                          for p in permutations(('mad', '>>', op))))
    if scope == 'tail':
        return sorted({('>>', op) for op in OPS} | {(op, '>>') for op in OPS})
    raise ValueError('unknown scope')


def search(sequence, scope='middle', timeout_ms=3000, seconds=None,
           fixed_shift=None, first_source=None, control=False, output_preimages=False):
    if scope not in ('middle', 'tail') or not 1 <= len(sequence) <= 4:
        raise ValueError('invalid scope or instruction count')
    if any(op not in OPS for op in sequence) or timeout_ms <= 0:
        raise ValueError('invalid opcode or timeout')
    if seconds is None:
        seconds = timeout_ms/1000+2
    if seconds <= 0:
        raise ValueError('positive total search limit required')
    count = 5 if scope == 'middle' else 8
    if fixed_shift is not None and not 0 <= fixed_shift <= 32:
        raise ValueError('representative literal shifts are 0..32')
    if first_source is not None and (sequence[0] != '>>' or not 0 <= first_source < count):
        raise ValueError('first source requires initial SHR and an available live input')
    if control and (tuple(sequence) != ('>>', 'mad') or scope != 'middle'):
        raise ValueError('control uses middle pool with SHR,MAD')
    solver = z3.SolverFor('QF_BV')
    spec = []
    for i, op in enumerate(sequence):
        a, b, c = z3.BitVecs(f'live_a{i} live_b{i} live_c{i}', 4)
        k0, k1 = z3.BitVecs(f'live_k{i}_0 live_k{i}_1', 32)
        for selector in (a, b, c):
            solver.add(z3.ULT(selector, count+i+2))
        if op in ('+', '*', 'mad', '^', '&', '|', '=='):
            solver.add(z3.ULE(a, b))
        if op != 'mad':
            solver.add(c == 0)
        if op == '>>' and fixed_shift is not None:
            solver.add(b == count+i+1, k1 == fixed_shift)
        if i == 0 and first_source is not None:
            solver.add(a == first_source)
        if i == 0 and op in ('//', '%', 'cdiv'):
            # Every live operand has a real zero preimage, so no live word
            # can be a universally valid first divisor. Rename the two
            # unrestricted constants to place the nonzero divisor last.
            solver.add(b == count+1, k1 != 0)
        spec.append((op, a, b, c, k0, k1))

    def program(prior, model=None):
        values = live_words(prior, scope)
        valid = []
        for op, a, b, c, k0, k1 in spec:
            fields = (a, b, c, k0, k1)
            a, b, c, k0, k1 = [model.eval(v, model_completion=True) for v in fields] if model is not None else fields
            pool = values + [k0, k1]
            out, ok = operation(op, choose(a, pool), choose(b, pool), choose(c, pool))
            values.append(out)
            valid.append(ok)
        return z3.simplify(values[-1]), z3.simplify(z3.And(*valid))

    def reference(prior):
        if control:
            return (z3.LShR(prior, 7)*33+live_words(prior)[4])&MASK
        return target(prior, scope)

    points = samples(scope, output_preimages)
    for value in points:
        x = z3.BitVecVal(value, 32)
        y, valid = program(x)
        solver.add(valid, y == reference(x))
    started = time.monotonic()
    deadline = started + seconds
    report = dict(scope=scope, sequence=list(sequence), width=32, timeout_ms=timeout_ms,
        seconds_limit=seconds, fixed_shift=fixed_shift, first_source=first_source,
        output_preimages=output_preimages,
        first_division_constant_cut=sequence[0] in ('//', '%', 'cdiv'),
        control=control, live_pool=list(LIVE_NAMES[:count]), initial_samples=points,
        result='deadline', events=[], new_executable=False)
    while time.monotonic() < deadline:
        solver.set(timeout=min(timeout_ms, max(1, int((deadline-time.monotonic())*1000))))
        status = solver.check()
        event = dict(samples=len(points), synthesis=str(status))
        report['events'].append(event)
        if status != z3.sat:
            report['result'] = 'grammar_excluded' if status == z3.unsat else 'unknown'
            if status == z3.unknown:
                report['reason'] = solver.reason_unknown()
            break
        model = solver.model()
        candidate = [dict(op=row[0], operands=[model.eval(v, model_completion=True).as_long()
            for v in row[1:4]], constants=[model.eval(v, model_completion=True).as_long()
            for v in row[4:]]) for row in spec]
        report['candidate'] = candidate
        x = z3.BitVec('full_live_counterexample', 32)
        y, valid = program(x, model)
        verify = z3.SolverFor('QF_BV')
        verify.set(timeout=min(timeout_ms, max(1, int((deadline-time.monotonic())*1000))))
        verify.add(z3.Or(z3.Not(valid), y != reference(x)))
        checked = verify.check()
        event['verification'] = str(checked)
        if checked == z3.unsat:
            report['result'] = 'proved_equivalent_full32'
            break
        if checked == z3.unknown:
            report.update(result='verification_unknown', reason=verify.reason_unknown())
            break
        value = verify.model().eval(x, model_completion=True).as_long()
        event['counterexample'] = value
        points.append(value)
        x = z3.BitVecVal(value, 32)
        y, valid = program(x)
        solver.add(valid, y == reference(x))
    report['seconds'] = round(time.monotonic()-started, 3)
    return report


def run_query(scope, order, timeout_ms, fixed_shift=None, first_source=None, output_preimages=False):
    cmd = [sys.executable, '-m', 'codex.search_full_live_shift863', '--query',
        '--scope', scope, '--sequence='+','.join(order), '--timeout-ms', str(timeout_ms)]
    if fixed_shift is not None:
        cmd += ['--fixed-shift', str(fixed_shift)]
    if first_source is not None:
        cmd += ['--first-source', str(first_source)]
    if output_preimages:
        cmd += ['--output-preimages']
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=max(30, 10+3*timeout_ms/1000))
    except subprocess.TimeoutExpired as exc:
        return dict(scope=scope, sequence=list(order), result='process_timeout',
                    stderr=str(exc), command=cmd)
    if proc.returncode:
        return dict(scope=scope, sequence=list(order), result='process_error', stderr=proc.stderr,
                    returncode=proc.returncode, command=cmd)
    row = json.loads(proc.stdout)
    row['command'] = cmd
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--query', action='store_true')
    ap.add_argument('--scope', choices=('middle', 'tail'), default='middle')
    ap.add_argument('--sequence')
    ap.add_argument('--timeout-ms', type=int, default=3000)
    ap.add_argument('--fixed-shift', type=int)
    ap.add_argument('--first-source', type=int)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--workers', type=int, choices=(1, 2), default=2)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--output-preimages', action='store_true')
    ap.add_argument('--unresolved-from', type=Path)
    args = ap.parse_args()
    if args.query:
        if not args.sequence:
            ap.error('query requires sequence')
        print(json.dumps(search(args.sequence.split(','), args.scope, args.timeout_ms,
              fixed_shift=args.fixed_shift, first_source=args.first_source,
              output_preimages=args.output_preimages)))
        return
    if not args.output:
        ap.error('scan requires output')
    report = None
    if args.resume and args.output.exists():
        report = json.loads(args.output.read_text())
        if report['scope'] != args.scope or report['width'] != 32:
            ap.error('resume report has a different scope or word width')
        failed = [row for row in report['rows'] if row['result'].startswith('process_')]
        report.setdefault('failed_attempts', []).extend(failed)
        report['rows'] = [row for row in report['rows'] if not row['result'].startswith('process_')]
        if any(row['fixed_shift'] != args.fixed_shift or row['first_source'] != args.first_source
               or row.get('output_preimages', False) != args.output_preimages
               for row in report['rows']):
            ap.error('resume report has different operand restrictions')
    if report is None:
        control = search(('>>', 'mad'), timeout_ms=10000, fixed_shift=7, first_source=0, control=True)
        if control['result'] != 'proved_equivalent_full32':
            raise RuntimeError('Full32 positive control failed: '+json.dumps(control))
        report = dict(z3=z3.get_version_string(), scope=args.scope, width=32, control=control,
                      new_executable=False, rows=[])
    orders = sequences(args.scope)
    if args.unresolved_from:
        prior = json.loads(args.unresolved_from.read_text())
        if prior['scope'] != args.scope or prior['width'] != 32:
            ap.error('refinement scope mismatch')
        if 'coverage' in prior:
            orders = [tuple(row['sequence']) for row in prior['coverage']['orders'] if not row['excluded']]
        else:
            orders = [tuple(row['sequence']) for row in prior['rows'] if row['result'] in
                      ('unknown', 'verification_unknown', 'deadline')]
        report['prior_report'] = str(args.unresolved_from)
    if args.sequence:
        orders = [tuple(args.sequence.split(','))]
    if args.first_source is not None:
        orders = [order for order in orders if order[0] == '>>']
    completed = {tuple(row['sequence']) for row in report['rows']}
    orders = [order for order in orders if tuple(order) not in completed]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_query, args.scope, order, args.timeout_ms,
                args.fixed_shift, args.first_source, args.output_preimages) for order in orders]
        for job in as_completed(jobs):
            row = job.result()
            report['rows'].append(row)
            report['rows'].sort(key=lambda r: tuple(r['sequence']))
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({k: row.get(k) for k in ('scope', 'sequence', 'result', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
