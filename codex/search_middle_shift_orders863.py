"""Reproducible full32 scan of three-op middle-hash orders containing SHR.

Uses the existing operand-selecting CEGIS model. Timeouts remain UNKNOWN.
This runner keeps raw solver events and does not infer performance scores.
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


def sequences():
    ops = ('+','-','^','&','|','<<','>>','*','mad')
    return sorted(set(p for op in ops for p in permutations(('mad','>>',op))))


def command(order, check_ms=1500, consecutive_samples=24, extra_samples=24,
            high_bit_pairs=False):
    if check_ms <= 0 or not 0 <= consecutive_samples <= 4096 or extra_samples < 0:
        raise ValueError('invalid time limit or sample count')
    args = [sys.executable, '-m', 'codex.synthesize_hash_cegis',
        '--slice', 'middle', '--width', '32', '--instructions', '3',
        '--op-sequence='+','.join(order), '--canonical-operands',
        '--extra-samples', str(extra_samples), '--consecutive-samples', str(consecutive_samples),
        '--seconds', str(max(4, (check_ms+999)//1000+2)), '--check-ms', str(check_ms)]
    if high_bit_pairs:
        args.append('--high-bit-pairs')
    return args


def run(order, check_ms, **kwargs):
    args = command(order, check_ms, **kwargs)
    start = time.monotonic()
    process = subprocess.run(args, capture_output=True, text=True)
    rows = [json.loads(line) for line in process.stdout.splitlines()
            if line.startswith('{')]
    return dict(sequence=order, returncode=process.returncode, results=rows,
        stderr=process.stderr, seconds=round(time.monotonic()-start,3), command=args)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--check-ms', type=int, default=1500)
    ap.add_argument('--workers', type=int, choices=(1,2), default=2)
    ap.add_argument('--unresolved-from', type=Path,
                    help='Read coverage.orders from a certificate; search only unresolved orders')
    ap.add_argument('--consecutive-samples', type=int, default=24)
    ap.add_argument('--extra-samples', type=int, default=24)
    ap.add_argument('--high-bit-pairs', action='store_true')
    args = ap.parse_args()
    if args.check_ms <= 0:
        ap.error('positive timeout required')
    if not 0 <= args.consecutive_samples <= 4096 or args.extra_samples < 0:
        ap.error('consecutive samples must be in0..4096; extra samples must be nonnegative')
    results = []
    orders = sequences()
    if args.unresolved_from:
        coverage = json.loads(args.unresolved_from.read_text())['coverage']['orders']
        orders = [tuple(row['sequence']) for row in coverage if not row['excluded']]
        if len(set(orders)) != len(orders) or any(order not in sequences() for order in orders):
            ap.error('coverage contains duplicated or unsupported unresolved orders')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run, order, args.check_ms,
            consecutive_samples=args.consecutive_samples, extra_samples=args.extra_samples,
            high_bit_pairs=args.high_bit_pairs) for order in orders]
        for job in as_completed(jobs):
            result = job.result()
            results.append(result)
            args.output.write_text(json.dumps(sorted(results,key=lambda r:r['sequence']),indent=2)+'\n')
            print(json.dumps(dict(sequence=result['sequence'],
                last=result['results'][-1] if result['results'] else result['stderr'])), flush=True)


if __name__ == '__main__':
    main()
