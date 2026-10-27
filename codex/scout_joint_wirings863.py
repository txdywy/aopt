"""Bounded scout over explicit five-instruction C1+C5 joint wirings.

This intentionally samples, rather than exhausts, each instruction order.
Two runtime MAD multipliers, squares, AND/OR masks and original-input
bypasses are allowed. Results retain the complete input manifest. Low11
UNSAT excludes only that wiring; every timeout remains unresolved.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations, product
import hashlib
import json
from pathlib import Path
import random
import time

from codex.cegis_c1_absorb863 import C1, _instr_options
from codex.search_restore_wirings863 import solve


def live(order, wiring):
    return all(any(j+1 in wiring[i] for i in range(j+1, len(order)))
               for j in range(len(order)-1))


def depth(wiring):
    d = [0]
    for w in wiring:
        d.append(1+max((d[i] for i in w if i < len(d)), default=0))
    return d[-1]


def orders(first='mad'):
    for mads in (2,3,4):
        for positions in combinations(range(1,5), mads-int(first == 'mad')):
            positions = set(positions) | ({0} if first == 'mad' else set())
            for bits in product(('^','&','|'), repeat=5-mads-int(first != 'mad')):
                it = iter(bits)
                yield (first, *('mad' if i in positions else next(it) for i in range(1,5)))


def cases(per_order, max_depth, seed, first='mad'):
    rng = random.Random(seed)
    out, stats = [], []
    for order in orders(first):
        opts = [_instr_options(i, op) for i,op in enumerate(order)]
        opts[0] = [(0,1,2), (0,0,1), (0,0,0)] if first == 'mad' else [(0,1)]
        selected = set()
        attempts = 0
        while len(selected) < per_order and attempts < 50000:
            attempts += 1
            wiring = tuple(rng.choice(oo) for oo in opts)
            if order[-1] != 'mad' and max(wiring[-1]) >= 5:
                # Constant final bitop: fixed/free output-key <=4-op case.
                continue
            if not live(order, wiring) or depth(wiring) > max_depth:
                continue
            if first != 'mad' and not any(0 in w for w in wiring[1:]):
                # A real input bypass is the reason to search a first bitop.
                continue
            selected.add(wiring)
        for wiring in sorted(selected):
            out.append(dict(order=list(order), wiring=[list(w) for w in wiring],
                            dependency_depth=depth(wiring)))
        stats.append(dict(order=list(order), selected=len(selected), attempts=attempts))
    return out, stats


def job(args):
    row, seconds = args
    trace = []
    for width in (11,12,16,32):
        result = solve(row['order'], row['wiring'], width, seconds,
                       dense=True, fixed_input_key=C1)
        trace.append({k:result[k] for k in ('width','status','seconds')})
        if result['status'] != 'reduced_sat':
            break
    result.update(row, ladder=trace)
    if result['status'] == 'proved_full32':
        result['macro'] = {k: result[k] for k in ('order','wiring','constants','input_key')}
    return result


def write_report(path, report):
    # One complete circuit per line keeps manifests and evidence reviewable.
    fields = []
    for name, value in report.items():
        if name in ('orders', 'manifest', 'results'):
            encoded = '[\n'+',\n'.join('    '+json.dumps(row) for row in value)+'\n  ]'
        else:
            encoded = json.dumps(value)
        fields.append('  '+json.dumps(name)+': '+encoded)
    path.write_text('{\n'+',\n'.join(fields)+'\n}\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--per-order', type=int, default=12)
    ap.add_argument('--max-depth', type=int, choices=(3,4,5), default=4)
    ap.add_argument('--seed', type=int, default=8642808)
    ap.add_argument('--first', choices=('mad','^','&','|'), default='mad')
    ap.add_argument('--seconds', type=float, default=0.75)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--refine-from', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    source = {}
    if args.refine_from:
        data = args.refine_from.read_bytes()
        old = json.loads(data)
        assert old['completed'] == len(old['manifest']) == len(old['results'])
        manifest = [{k:r[k] for k in ('order','wiring','dependency_depth')}
                    for r in old['results'] if r['status'] not in ('unsat','proved_full32')]
        stats = []
        args.first = old.get('first', 'mad')
        args.per_order, args.max_depth, args.seed = (old[k] for k in ('per_order','max_depth','seed'))
        source = dict(source=str(args.refine_from), source_sha256=hashlib.sha256(data).hexdigest())
    else:
        manifest, stats = cases(args.per_order, args.max_depth, args.seed, args.first)
    report = dict(sampled_not_exhaustive=True, fixed_input_key=C1,
                  per_order=args.per_order, max_depth=args.max_depth, seed=args.seed,
                  first=args.first,
                  seconds_per_width=args.seconds, workers=args.workers,
                  orders=stats, manifest=manifest, results=[], completed=0,
                  best_executable_cycles=864, new_executable=False)
    report.update(source)
    counts = Counter()
    start = time.monotonic()
    print(json.dumps(dict(jobs=len(manifest), orders=len(stats))), flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(job, (row,args.seconds)) for row in manifest]
        for done,f in enumerate(as_completed(futures),1):
            result = f.result()
            report['results'].append(result)
            counts[result['status']] += 1
            if result['status'] == 'proved_full32':
                print(json.dumps(result), flush=True)
            if done % 100 == 0 or done == len(manifest):
                report.update(completed=done, counts=dict(counts),
                              seconds=round(time.monotonic()-start,3))
                write_report(args.output, report)
                print(json.dumps({k:report[k] for k in ('completed','counts','seconds')}), flush=True)


if __name__ == '__main__':
    main()
