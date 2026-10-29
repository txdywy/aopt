"""Four-op joint window with a synthesized prepared right-shift count.

Exactly one SHR, no free e, and the previous eight real intermediates.
The full relevant count domain16..31 is symbolic in each query. This
samples wirings; it is not an exhaustive four-op grammar or a score.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import time

from codex.scout_joint_wirings863 import write_report
from codex.search_focused_joint_boundary863 import edge_points, manifest
from codex.search_joint_live_boundary863 import C0, C4, NAMES, NODE, solve


def with_prepared_shift(row):
    assert row['order'].count('shr16') == 1
    return dict(order=['shr' if op == 'shr16' else op for op in row['order']],
                wiring=[list(w) for w in row['wiring']])


def control():
    row = dict(order=['mad', 'shr', '^', 'mad'],
               wiring=[[0, 8, 9], [8], [8, 9], [10, 11, 12]])
    # The count is not pinned: recover16 from the entire16..31 domain.
    result = solve(row, 0, 15, [(0, 0, 9), (0, 1, C4), (3, 0, 4097), (3, 1, C0)],
                   True, edge_points(0))
    assert result['status'] == 'proved_equivalent', result
    assert result['constants'][1][0] == 16, result
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--per-order', type=int, default=32)
    ap.add_argument('--seed', type=int, default=8642932)
    ap.add_argument('--seconds', type=float, default=2)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--manifest-from', type=Path)
    ap.add_argument('--refine-from', type=Path)
    ap.add_argument('--edge-targets', action='store_true')
    ap.add_argument('--split-counts', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); source = {}
    started = time.monotonic()
    if args.manifest_from or args.refine_from:
        path = args.manifest_from or args.refine_from
        raw = path.read_bytes(); decoded = gzip.decompress(raw) if path.suffix == '.gz' else raw
        previous = json.loads(decoded)
        assert previous['completed'] == len(previous['manifest']) == len(previous['results'])
        args.seed, args.per_order = previous['seed'], previous['per_order']
        source = dict(source=str(path), source_sha256=hashlib.sha256(raw).hexdigest())
        if args.refine_from:
            closed = {'unsat', 'shift_wordcut_excluded', 'lsb_dependency_excluded', 'one_high_tail_excluded', 'polynomial_quotient_excluded', 'quotient_pair_excluded', 'proved_equivalent'}
            rows = [{k: r[k] for k in ('order', 'wiring', 'count_pin') if k in r}
                    for r in previous['results'] if r['status'] not in closed]
        else:
            rows = [with_prepared_shift(r) for r in previous['manifest']]
        orders = previous['orders']
    else:
        old, orders = manifest(args.per_order, args.seed)
        rows = [with_prepared_shift(r) for r in old]
    if args.split_counts:
        rows = [dict(row,count_pin=count) for row in rows
                for count in ([row['count_pin']] if 'count_pin' in row else range(16,32))]
    extra = edge_points() if args.edge_targets else []
    report = dict(node=NODE, input_names=NAMES, free_e=False, sampled_not_exhaustive=True,
                  no_complete_four_op_grammar_claim=True, prepared_count_domain=[16,31],
                  count_outside_domain_exclusion='All initial low16 remain equal under b^65536, while target bit0 flips. A single shift of count<16 or>=32 cannot distinguish this pair through T operations.',
                  split_prepared_counts=any('count_pin' in row for row in rows),
                  seconds_per_case=args.seconds, edge_primary_points=extra,
                  seed=args.seed, per_order=args.per_order, manifest=rows, orders=orders,
                  positive_control=control(), results=[], completed=0, counts={},
                  best_executable_cycles=864, new_executable=False, **source)
    write_report(args.output, report)
    print('prepared-count cases', len(rows), 'domain16..31', flush=True)
    counts = Counter(); checkpoint = min(100, max(1, len(rows)//10))
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(solve, row, NODE, args.seconds,
                               [(row['order'].index('shr'),0,row['count_pin'])] if 'count_pin' in row else (),
                               True, extra) for row in rows]
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result(); report['results'].append(result); counts[result['status']] += 1
            if result['status'] == 'proved_equivalent':
                print(result, flush=True)
            if done % checkpoint == 0 or done == len(rows):
                report.update(completed=done, counts=dict(counts), seconds=round(time.monotonic()-started,3))
                write_report(args.output, report); print(done, dict(counts), report['seconds'], flush=True)


if __name__ == '__main__':
    main()
