"""Prioritize four-op programs whose sole shift reads a MAD descendant.

This is a sampled search, not an exhaustive grammar or a lower bound.
It uses the existing full32 synthesis/verification implementation and its
positive controls. Sources never include e=9*z+C4 as a free input.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
from itertools import permutations, product
import json
from pathlib import Path
import random
import time

from codex.cegis_c1_absorb863 import _instr_options
from codex.audit_division_first_t863 import source_inverse
from codex.search_joint_live_boundary863 import C0, MASK, NAMES, NODE, controls, inputs, parity_models, solve, wanted
from codex.scout_joint_wirings863 import write_report


def key(row):
    return tuple(row['order']), tuple(tuple(w) for w in row['wiring'])


def edge_points(node=NODE):
    """Legal full32 b with small/edge initial words or prescribed targets."""
    words = [0, 1, 2, 3, 4, 7, 15, 31, 255, 65535, 65536, 65537,
             (1 << 31)-1, 1 << 31, (1 << 31)+1, MASK-1, MASK]
    points = [source_inverse(src, word) for src in (0, 1, 2, 4, 5, 6, 8) for word in words]
    inverse = pow(4097, -1, 1 << 32)
    targets = list(dict.fromkeys([*words, *range(32), *(1 << i for i in range(32)),
                                 *(MASK ^ (1 << i) for i in range(32))]))
    for y in targets:
        folded = (((y-C0)*inverse) & MASK) ^ node
        e = folded ^ (folded >> 16)
        b = source_inverse(0, e)
        assert wanted(inputs(b), node) == y
        points.append(b)
    return list(dict.fromkeys([*points, *(b ^ 65536 for b in points)]))


def shift_has_mad_ancestor(row):
    n = len(NAMES)
    shift = row['order'].index('shr16')
    def visit(index):
        if index < n:
            return False
        producer = index-n
        if row['order'][producer] == 'mad':
            return True
        return any(visit(j) for j in row['wiring'][producer] if j < n+producer)
    return visit(row['wiring'][shift][0])


def manifest(per_order, seed, excluded=()):
    rng = random.Random(seed); n = len(NAMES); rows = []; stats = []; seen_orders = set()
    excluded = set(excluded)
    for mads in (1, 2, 3):
        for bits in product(('^', '&', '|'), repeat=3-mads):
            for order in sorted(set(permutations(('shr16', *('mad',)*mads, *bits)))):
                shift = order.index('shr16')
                if order in seen_orders or shift == 3 or 'mad' not in order[:shift]:
                    continue
                seen_orders.add(order)
                options = [[(j,) for j in range(n+i)] if op == 'shr16'
                           else _instr_options(n+i-1, op) for i, op in enumerate(order)]
                selected = set(); counts = Counter()
                for _ in range(200000):
                    if len(selected) >= per_order:
                        break
                    wiring = tuple(rng.choice(oo) for oo in options)
                    if wiring in selected or (order, wiring) in excluded:
                        counts['duplicate'] += 1; continue
                    if not all(any(n+j in wiring[i] for i in range(j+1, 4)) for j in range(3)):
                        counts['dead_computed_word'] += 1; continue
                    row = dict(order=list(order), wiring=[list(w) for w in wiring])
                    if not shift_has_mad_ancestor(row):
                        counts['different_search_priority'] += 1; continue
                    if not parity_models(row):
                        counts['lsb_dependency'] += 1; continue
                    selected.add(wiring)
                rows.extend(dict(order=list(order), wiring=[list(w) for w in wiring])
                            for wiring in sorted(selected))
                stats.append(dict(order=list(order), selected=len(selected), selection_counts=dict(counts)))
    assert len(seen_orders) == 44
    assert len({key(row) for row in rows}) == len(rows)
    assert all(key(row) not in excluded and shift_has_mad_ancestor(row) for row in rows)
    return rows, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--per-order', type=int, default=256)
    ap.add_argument('--seed', type=int, default=8642930)
    ap.add_argument('--seconds', type=float, default=1)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--exclude', type=Path, action='append', default=[])
    ap.add_argument('--refine-from', type=Path)
    ap.add_argument('--edge-targets', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    excluded = set(); sources = []
    for path in args.exclude:
        raw = path.read_bytes(); decoded = gzip.decompress(raw) if path.suffix == '.gz' else raw
        previous = json.loads(decoded)
        excluded.update(key(row) for row in previous['manifest'])
        sources.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                            decoded_sha256=hashlib.sha256(decoded).hexdigest()))
    started = time.monotonic(); provenance = {}
    if args.refine_from:
        raw = args.refine_from.read_bytes(); decoded = gzip.decompress(raw) if args.refine_from.suffix == '.gz' else raw
        previous = json.loads(decoded)
        assert previous['completed'] == len(previous['manifest']) == len(previous['results'])
        rows = [{k: row[k] for k in ('order', 'wiring')} for row in previous['results']
                if row['status'] not in ('unsat', 'shift_wordcut_excluded', 'lsb_dependency_excluded', 'one_high_tail_excluded', 'polynomial_quotient_excluded', 'quotient_pair_excluded', 'proved_equivalent')]
        orders = previous['orders']; args.per_order = previous['per_order']; args.seed = previous['seed']
        provenance = dict(source=str(args.refine_from), source_sha256=hashlib.sha256(raw).hexdigest())
    else:
        rows, orders = manifest(args.per_order, args.seed, excluded)
    extra = edge_points() if args.edge_targets else []
    report = dict(node=NODE, input_names=NAMES, free_e=False, sampled_not_exhaustive=True,
                  family='sample', focus='The sole SHR16 reads a computed word with a MAD in its ancestry.',
                  no_claim_of_complete_four_op_grammar=True, per_order=args.per_order, seed=args.seed,
                  seconds_per_case=args.seconds, dense_fibres=True, full32_semantics=True,
                  edge_primary_points=extra,
                  excluded_sources=sources, excluded_unique_wirings=len(excluded),
                  controls=controls(), manifest=rows, orders=orders, results=[], completed=0, counts={},
                  best_executable_cycles=864, new_executable=False, **provenance)
    write_report(args.output, report)
    print('focused cases', len(rows), 'orders', len(orders), flush=True)
    counts = Counter()
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(solve, row, NODE, args.seconds, (), True, extra) for row in rows]
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result(); report['results'].append(result); counts[result['status']] += 1
            if result['status'] == 'proved_equivalent':
                print(result, flush=True)
            if done % 1000 == 0 or done == len(rows):
                report.update(completed=done, counts=dict(counts), seconds=round(time.monotonic()-started, 3))
                write_report(args.output, report); print(done, dict(counts), report['seconds'], flush=True)


if __name__ == '__main__':
    main()
