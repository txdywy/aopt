"""Search four-instruction SSA graphs across the middle hash's XOR boundaries.

The earlier cross-encoding scan fixed a parallel affine/XOR/affine graph.
Here every operand may use the input, any earlier result, or a free constant.
Reduced-width equivalence is a candidate only; UNKNOWN is never an exclusion.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import permutations
import json
from pathlib import Path
import subprocess
import sys
import time

from codex.search_cross_encoding863 import C1, PRE_C5
from codex.close_remaining_shifts863 import rectangle
from codex.synthesize_hash_cegis import middle_reference


KEYS = {'input': (C1, 0), 'output': (0, PRE_C5), 'both': (C1, PRE_C5)}


def sequences():
    return sorted(set(permutations(('mad', 'mad', 'mad', '^')))
                  | set(permutations(('mad', 'mad', '^', '^'))))


def prefix_certificate(mode, order):
    """Bitwise-only prefix followed by a polynomial has zero mixed difference.

    This holds for arbitrary constants and arbitrary polynomial degree, so
    it includes all operand choices of both indicated four-op sequences.
    """
    if mode not in KEYS:
        raise ValueError('unknown encoding')
    if tuple(order) not in (('^', 'mad', 'mad', 'mad'), ('^', '^', 'mad', 'mad')):
        return None
    ik, ok = KEYS[mode]
    row = rectangle(3, 8, 1 if mode == 'both' else 0, 11,
                    lambda x: middle_reference(x, ik, ok))
    assert row['mixed_difference'] != 0
    return dict(mode=mode, sequence=list(order), rectangle=row,
        reason='Orthogonal input-bit variations survive any bitwise-only prefix; '
               'all later integer polynomials have mixed difference zero modulo 2048.')


def command(order, mode, width=11, check_ms=5000):
    if mode not in KEYS or not 1 <= width <= 32 or check_ms <= 0:
        raise ValueError('invalid encoding, width or time limit')
    if order not in sequences():
        raise ValueError('unsupported four-instruction sequence')
    ik, ok = KEYS[mode]
    return [sys.executable, '-m', 'codex.synthesize_hash_cegis',
        '--slice', 'middle', '--width', str(width), '--instructions', '4',
        '--op-sequence='+','.join(order), '--canonical-operands',
        '--middle-input-xor', hex(ik), '--middle-output-xor', hex(ok),
        '--extra-samples', '16', '--consecutive-samples', '32',
        '--high-bit-pairs', '--seconds', str(max(4, (check_ms+999)//1000+2)),
        '--check-ms', str(check_ms)]


def run(order, mode, width, check_ms):
    cmd = command(order, mode, width, check_ms)
    start = time.monotonic()
    process = subprocess.run(cmd, capture_output=True, text=True)
    events = [json.loads(line) for line in process.stdout.splitlines() if line.startswith('{')]
    return dict(sequence=order, mode=mode, width=width, command=cmd,
        returncode=process.returncode, events=events, stderr=process.stderr,
        seconds=round(time.monotonic()-start, 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--width', type=int, default=11)
    ap.add_argument('--check-ms', type=int, default=5000)
    ap.add_argument('--workers', type=int, choices=(1, 2), default=2)
    ap.add_argument('--modes', nargs='+', choices=tuple(KEYS), default=list(KEYS))
    ap.add_argument('--unresolved-from', type=Path,
                    help='Refine only earlier UNKNOWN/deadline cases; skip proved prefix cases')
    args = ap.parse_args()
    if not 1 <= args.width <= 32 or args.check_ms <= 0:
        ap.error('width must be 1..32 and time limit must be positive')
    if len(set(args.modes)) != len(args.modes):
        ap.error('modes must be distinct')
    cases = [(mode, order) for mode in args.modes for order in sequences()]
    if args.unresolved_from:
        previous = json.loads(args.unresolved_from.read_text())
        if previous['width'] != args.width or args.width < 11:
            ap.error('refinement requires the same width, at least 11')
        prior = {(r['mode'], tuple(r['sequence'])): r for r in previous['rows']}
        if any(case not in prior for case in cases):
            ap.error('prior report does not contain every requested case')
        cases = [(mode, order) for mode, order in cases
            if prior[mode, order]['events'][-1]['result'] in
               ('inconclusive', 'inconclusive_verification', 'deadline')
            and prefix_certificate(mode, order) is None]
    report = dict(scope='four instructions, arbitrary SSA operands, one input; '
        '3 MAD + 1 XOR or 2 MAD + 2 XOR; arbitrary constants',
        width=args.width, new_executable=False, rows=[],
        prior_report=str(args.unresolved_from) if args.unresolved_from else None,
        polynomial_prefixes=[cert for mode in args.modes for order in sequences()
                             if (cert := prefix_certificate(mode, order)) is not None])
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run, order, mode, args.width, args.check_ms)
                for mode, order in cases]
        for job in as_completed(jobs):
            row = job.result()
            report['rows'].append(row)
            report['rows'].sort(key=lambda r: (r['mode'], r['sequence']))
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(dict(mode=row['mode'], sequence=row['sequence'],
                last=row['events'][-1] if row['events'] else row['stderr'])), flush=True)


if __name__ == '__main__':
    main()
