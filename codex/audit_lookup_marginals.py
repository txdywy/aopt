"""Measure isolated table substitutions. Deltas are NOT additive proofs.

Every sample builds the actual DAG in an isolated subprocess. Assembly-only
consumers/instructions, cross-group setup sharing and scheduling are excluded.
Use these measurements to choose experiments, never as executable scores.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', default='3,4,14,15')
    parser.add_argument('--groups', default=','.join(map(str, range(32))))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = [sys.executable, str(root / 'codex/experiment_paired_base.py'),
            '--graph-only', '--arity', '4', '--keep-final', '--compact-return',
            '--recycle-stride', '--parallel', '--restore-hash-count', '88',
            '--omit-fused-node-transform']

    def run(round_number, group=None):
        command = base + ['--round', str(round_number), '--groups', '0']
        if group is not None:
            command += ['--group-ids', str(group)]
        result = subprocess.run(command, cwd=root, capture_output=True, text=True)
        if result.returncode:
            return {'error': result.stderr.splitlines()[-1]}
        return json.loads(result.stdout)['operations']

    reports = []
    for round_number in map(int, args.rounds.split(',')):
        baseline = run(round_number)
        for group in map(int, args.groups.split(',')):
            counts = run(round_number, group)
            record = {'round': round_number, 'group': group}
            if 'error' in counts or 'error' in baseline:
                record['error'] = counts.get('error', baseline.get('error'))
            else:
                delta = {engine: counts.get(engine, 0) - baseline.get(engine, 0)
                         for engine in ('alu', 'valu', 'load', 'flow', 'store')}
                record.update(delta=delta, arithmetic_lane_work_delta=
                              delta['alu'] + 8 * delta['valu'])
            reports.append(record)
    print(json.dumps({'scope': 'isolated DAG marginals; no execution; not additive',
                      'reports': reports}, indent=2))


if __name__ == '__main__':
    main()
