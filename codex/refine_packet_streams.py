"""Iteratively repair necessary energy cuts in a projected LF calendar.

The output is a relaxation, not an executable score. Exact scalar/vector
packing, scratch allocation and frozen-machine replay are still required.
The input must have a valid frozen compute prefix and a feasible LF projection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys


def delay_outputs(ir, data, horizon):
    """Move terminal output stores later without changing any other operation."""
    n = data[0]
    times = data[2:2 + n]
    outputs = [j for j, op in enumerate(ir.ops) if op.tag[0] == 'output']
    assert all(ir.ops[j].engine == 'store' and not ir.succ[j] for j in outputs)
    output_set = set(outputs)
    used = Counter(times[j] for j, op in enumerate(ir.ops)
                   if op.engine == 'store' and j not in output_set)
    assert all(count <= 2 for count in used.values())
    for j in sorted(outputs, key=lambda j: (times[j], j), reverse=True):
        cycle = next((t for t in range(horizon - 1, times[j] - 1, -1)
                      if used[t] < 2), None)
        assert cycle is not None, 'No later output slot within the horizon'
        times[j] = cycle
        used[cycle] += 1
    return data[:2] + times + data[2 + n:]


def write_calendar(path, data):
    n, ng = data[:2]
    assert len(data) == 2 + n + ng
    path.write_text(f'{n} {ng}\n' + ' '.join(map(str, data[2:2 + n]))
                    + '\n' + ' '.join(map(str, data[2 + n:])) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix', type=Path, required=True)
    ap.add_argument('--calendar', type=Path)
    ap.add_argument('--output-prefix', type=Path, required=True)
    ap.add_argument('--horizon', type=int, required=True)
    ap.add_argument('--compute-before', type=int, required=True)
    ap.add_argument('--stream-before', type=int, default=0)
    ap.add_argument('--radius', type=int, default=12)
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--max-cuts', type=int, default=8)
    ap.add_argument('--tail-waste', type=int, default=116)
    ap.add_argument('--initial-cuts', type=Path)
    ap.add_argument('--delay-outputs', action='store_true')
    args = ap.parse_args()
    source = Path(str(args.prefix) + '.pkl')
    ir = pickle.loads(source.read_bytes())
    data = list(map(int, (args.calendar or Path(str(args.prefix) + '.sol'))
                    .read_text().split()))
    assert data[:2] == [len(ir.ops), len(ir.groups)]
    if args.delay_outputs:
        data = delay_outputs(ir, data, args.horizon)
    cuts = {'horizon': args.horizon, 'freeze_before': args.compute_before, 'cuts': []}
    if args.initial_cuts:
        cuts = json.loads(args.initial_cuts.read_text())
        assert cuts['horizon'] == args.horizon
        assert cuts['freeze_before'] == args.compute_before
    known = {(cut['a'], cut['b']) for cut in cuts['cuts']}
    current = Path(str(args.output_prefix) + '-0')
    write_calendar(Path(str(current) + '.sol'), data)
    for iteration in range(args.rounds + 1):
        shutil.copyfile(source, str(current) + '.pkl')
        audit_path = Path(str(current) + '-audit.json')
        subprocess.run([sys.executable, '-m', 'codex.packet_stream_windows',
                        '--prefix', str(current), '--horizon', str(args.horizon),
                        '--freeze-before', str(args.compute_before),
                        '--tail-waste', str(args.tail_waste),
                        '--output', str(audit_path), '--max-cuts', str(args.max_cuts)],
                       check=True)
        audit = json.loads(audit_path.read_text())
        assert not audit['inconsistent_nodes'], 'Invalid projected calendar'
        if not audit['positive_windows'] or iteration == args.rounds:
            print({'calendar': str(current) + '.sol',
                   'remaining_positive_windows': audit['positive_windows'],
                   'relaxed_calendar_only': True}, flush=True)
            return
        added = 0
        for cut in audit['cuts']:
            key = cut['a'], cut['b']
            if key not in known:
                # Global-energy repair considers every movable compute job.
                cuts['cuts'].append({k: v for k, v in cut.items() if k != 'jobs'})
                known.add(key)
                added += 1
        cut_path = Path(str(current) + '-cuts.json')
        cut_path.write_text(json.dumps(cuts, indent=2) + '\n')
        following = Path(str(args.output_prefix) + f'-{iteration + 1}')
        output = Path(str(following) + '.sol')
        assert not output.exists(), 'Choose a fresh output prefix'
        print({'round': iteration + 1, 'added_cuts': added,
               'accumulated_cuts': len(known)}, flush=True)
        subprocess.run([sys.executable, '-m', 'codex.packet_stream_repair',
                        '--prefix', str(current), '--output', str(output),
                        '--horizon', str(args.horizon),
                        '--compute-before', str(args.compute_before),
                        '--stream-before', str(args.stream_before),
                        '--radius', str(args.radius), '--free-outputs',
                        '--energy-cuts', str(cut_path), '--global-energy', '--soft-energy',
                        '--seconds', str(args.seconds), '--workers', str(args.workers)],
                       check=True)
        if not output.exists():
            print({'stopped': 'No repaired calendar returned',
                   'last_calendar': str(current) + '.sol'}, flush=True)
            return
        current = following


if __name__ == '__main__':
    main()
