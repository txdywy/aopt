"""Partition a bounded first-hash search; unknowns remain explicitly unresolved.

Search every distinct ordering of MAD, logical right shift, and one operation
from the existing grammar. This is not every three-instruction ISA program.
Print JSON lines so completed subproblems survive an interrupted run.
"""
import argparse
from itertools import permutations
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-ms', type=int, default=2000)
    args = parser.parse_args()
    if args.check_ms < 1:
        parser.error('check-ms must be positive')
    operations = ('+', '-', '^', '&', '|', '<<', '>>', '*', 'mad')
    sequences = sorted({seq for op in operations for seq in permutations(('mad', '>>', op))})
    counts = {}
    for sequence in sequences:
        command = [sys.executable, str(Path(__file__).with_name('synthesize_hash_cegis.py')),
                   '--slice', 'first', '--instructions', '3', '--op-sequence=' + ','.join(sequence),
                   '--check-ms', str(args.check_ms), '--seconds', str(max(2, (args.check_ms+999)//1000+1))]
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=max(10, args.check_ms/1000+5))
            report = json.loads(result.stdout.splitlines()[-1]) if result.returncode == 0 else {
                'result': 'process_error', 'stderr': result.stderr}
        except subprocess.TimeoutExpired:
            report = {'result': 'process_timeout'}
        counts[report['result']] = counts.get(report['result'], 0) + 1
        print(json.dumps({'sequence': sequence, **report}), flush=True)
    print(json.dumps({'summary': counts, 'sequences': len(sequences),
                      'scope': 'one MAD, one logical right shift, one grammar op; uint32'}))


if __name__ == '__main__':
    main()
