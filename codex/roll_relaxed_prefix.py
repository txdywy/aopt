"""Extend a packed compute prefix, learning future arithmetic-window cuts.

Every intermediate calendar still relaxes computation after the boundary.
No result is a score until complete packing and frozen-machine replay pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix', type=Path, required=True)
    ap.add_argument('--output-prefix', type=Path, required=True)
    ap.add_argument('--lo', type=int, required=True)
    ap.add_argument('--stop', type=int, required=True)
    ap.add_argument('--horizon', type=int, required=True)
    ap.add_argument('--step', type=int, default=10)
    ap.add_argument('--lookahead', type=int, default=0,
                    help='Plan additional cycles but commit only one step before replanning')
    ap.add_argument('--prefix-waste-budget', type=int, default=53)
    ap.add_argument('--seconds', type=float, default=30)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--cut-rounds', type=int, default=12)
    args = ap.parse_args()
    assert args.step > 0 and args.lo < args.stop < args.horizon
    source = Path(str(args.prefix) + '.pkl')
    ir = pickle.loads(source.read_bytes())
    current = args.prefix
    boundary = args.lo
    cut_cache = {}
    records = []
    while boundary < args.stop:
        high = min(args.stop, boundary + args.step)
        plan_high = min(args.horizon - 3, high + args.lookahead)
        assert high <= plan_high
        data = list(map(int, Path(str(current) + '.sol').read_text().split()))
        n = data[0]
        times = data[2:2 + n]
        used = sum(8 if op.engine == 'valu' else 1 for j, op in enumerate(ir.ops)
                   if op.engine in ('alu', 'valu') and times[j] < boundary)
        wasted = 60 * boundary - used
        remaining = args.prefix_waste_budget - wasted
        assert remaining >= 0
        accepted = None
        for allowed in range(remaining + 1):
            for iteration in range(args.cut_rounds):
                trial = Path(f'{args.output_prefix}-{high}-w{allowed}-r{iteration}')
                output = Path(str(trial) + '.sol')
                assert not output.exists(), 'Choose a fresh output prefix'
                cuts = {'horizon': args.horizon, 'freeze_before': high,
                        'cuts': [cut for (a, _), cut in cut_cache.items() if a >= high]}
                cutfile = Path(str(trial) + '-cuts.json')
                cutfile.write_text(json.dumps(cuts, indent=2) + '\n')
                subprocess.run([sys.executable, '-m', 'codex.pack_packet_prefix',
                    '--prefix', str(current), '--output', str(output),
                    '--lo', str(boundary), '--hi', str(plan_high),
                    '--horizon', str(args.horizon), '--max-waste', str(allowed),
                    '--seconds', str(args.seconds), '--workers', str(args.workers),
                    '--energy-cuts', str(cutfile)], check=True)
                if not output.exists():
                    break
                auditfile = Path(str(trial) + '-audit.json')
                subprocess.run([sys.executable, '-m', 'codex.packet_stream_windows',
                    '--prefix', str(current), '--calendar', str(output),
                    '--horizon', str(args.horizon), '--freeze-before', str(high),
                    '--output', str(auditfile), '--max-cuts', '8'], check=True)
                audit = json.loads(auditfile.read_text())
                assert not audit['inconsistent_nodes']
                records.append({'boundary': high, 'allowed_waste': allowed,
                                'cut_round': iteration,
                                'positive_windows': audit['positive_windows'],
                                'calendar': str(output)})
                if not audit['positive_windows']:
                    accepted = trial
                    break
                new = 0
                for cut in audit['cuts']:
                    key = cut['a'], cut['b']
                    if key not in cut_cache:
                        cut_cache[key] = {k: v for k, v in cut.items() if k != 'jobs'}
                        new += 1
                assert new, 'Selected energy cuts failed independent audit'
            if accepted:
                break
        if accepted is None:
            break
        shutil.copyfile(source, str(accepted) + '.pkl')
        current, boundary = accepted, high
        print({'accepted_prefix': boundary, 'calendar': str(current) + '.sol',
               'relaxed_calendar_only': True}, flush=True)
    result = {'packed_before': boundary, 'calendar': str(current) + '.sol',
              'relaxed_calendar_only': True, 'rounds': records}
    Path(str(args.output_prefix) + '-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print({k: v for k, v in result.items() if k != 'rounds'}, flush=True)


if __name__ == '__main__':
    main()
