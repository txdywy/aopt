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


def command(order, check_ms=1500):
    return [sys.executable, '-m', 'codex.synthesize_hash_cegis',
        '--slice', 'middle', '--width', '32', '--instructions', '3',
        '--op-sequence='+','.join(order), '--canonical-operands',
        '--extra-samples', '24', '--consecutive-samples', '24',
        '--seconds', str(max(4, (check_ms+999)//1000+2)), '--check-ms', str(check_ms)]


def run(order, check_ms):
    args = command(order, check_ms)
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
    args = ap.parse_args()
    if args.check_ms <= 0:
        ap.error('positive timeout required')
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run, order, args.check_ms) for order in sequences()]
        for job in as_completed(jobs):
            result = job.result()
            results.append(result)
            args.output.write_text(json.dumps(sorted(results,key=lambda r:r['sequence']),indent=2)+'\n')
            print(json.dumps(dict(sequence=result['sequence'],
                last=result['results'][-1] if result['results'] else result['stderr'])), flush=True)


if __name__ == '__main__':
    main()
