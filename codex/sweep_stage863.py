"""Systematic sweep of the 864 graph builder's configuration flags.

Every configuration rebuilds the full instruction graph and records the
necessary 863 conditions from scan_stage863.measure: arithmetic lanes,
prefix/suffix waste bounds and LOAD/FLOW stream bounds. The prefix bound here
ignores LOAD/FLOW congestion, so any configuration reaching 863 must be
re-checked with head_resource_bound863. Results are bounds, never scores.

Part A changes one option at a time from the verified baseline; part B draws
random combinations of two to four changes. Part C (--deep) does the same for
options that research_stage864.build hard-codes when it calls the frontier
builder (pointer packing, head C5 load swap, scalar shifts, terminal tables,
cache vectors, header and native packs, hash/coefficient packing, stride16).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import random
import traceback

BOOLS = ('cheap_coeff', 'head_fill', 'head_flow', 'alu_layout', 'tail_low', 'low_alu',
         'tail_single', 'early_literals', 'scalar_leaves', 'fold_raw_hash',
         'scalar_root_head', 'root_flow_exchange', 'low_bank_records', 'early_small_low_bank')
CHOICES = {
    'small_stage': ('4', '4,-2', '4,9', '4,19', '3,4', '3,4,-2', ''),
    'stage_hash': ('final', 'coeff,final', 'coeff,add,final', 'add,final', ''),
    'stage_uniform': ('-60,-40', '', '-60', '-40', '-60,-40,-20'),
    'tail_transposes': ('29,31', '', '31', '30,31', '28,29,30,31', '27,29,31'),
    'bootstrap_flow': (2, 0, 1, 3, 4),
    'bootstrap_early': (2, 1),
    'sink': (8, 0, 4, 16),
    'literals': (2, 0, 4),
    'literal_flow': (7, 5, 9),
}


DEEP = {  # options hard-coded inside research_stage864.build; part C varies them
    'head_c5': (False, True), 'pointer_pack': (False, True),
    'scalar_shifts': (0, 1, 2, 4), 'terminal_count': (0, 1, 2),
    'cache_vector': (False, True), 'header_pack': ('', '>>:1', '-:2', '+:16', '+:1'),
    'replicate_pack': (False, True), 'native_pack': ('', 'coeff', 'coeff,addend', 'coeff,addend,final'),
    'hash_pack': (True, False), 'coefficients': (True, False),
    'head4097': (True, False), 'head_full': (True, False),
    'stride16': (True, False), 'early_input': (True, False),
}


def one(changes):
    from functools import lru_cache
    from codex import stride4_family as family
    from codex import scan_stage863 as scan
    from codex import research_frontier863 as frontier
    if not getattr(family.capture, 'cache_info', None):
        family.capture = lru_cache(maxsize=1)(family.capture)
    deep = {k: v for k, v in changes.items() if k in DEEP}
    if not hasattr(frontier, '_orig_build'):
        frontier._orig_build = frontier.build
    def patched(args, _deep=deep):
        for k, v in _deep.items():
            setattr(args, k, v)
        return frontier._orig_build(args)
    frontier.build = patched
    config = scan.baseline(**{k: v for k, v in changes.items() if k not in DEEP})
    try:
        r = scan.measure(config, 2)
        return dict(changes=changes, ok=True, work=r['arithmetic_lanes'],
                    head=r['prefix']['waste_lower_bound'], tail=r['suffix']['waste_lower_bound'],
                    effective=r['effective_budget'], bound=r['cycle_lower_bound'],
                    streams={k: v['cycle_lower_bound'] for k, v in r['streams'].items()})
    except Exception as exc:  # invalid combination for this builder
        return dict(changes=changes, ok=False, error=f'{type(exc).__name__}: {exc}'[:200],
                    where=traceback.format_exc().strip().splitlines()[-2][:160])


def variants(samples, seed):
    from codex import scan_stage863 as scan
    base = scan.baseline()
    out = []
    for b in BOOLS:
        out.append({b: not base[b]})
    for k, vals in CHOICES.items():
        for v in vals:
            if v != base[k]:
                out.append({k: v})
    rng = random.Random(seed)
    keys = list(BOOLS) + list(CHOICES)
    seen = {json.dumps(v, sort_keys=True) for v in out}
    while len(out) < len(seen) + samples and len(seen) < 100000:
        ch = {}
        for k in rng.sample(keys, rng.randint(2, 4)):
            ch[k] = (not base[k]) if k in BOOLS else rng.choice([v for v in CHOICES[k] if v != base[k]])
        key = json.dumps(ch, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(ch)
        if len(out) >= 2 * len(BOOLS) + samples + sum(len(v) for v in CHOICES.values()):
            break
    return out


def deep_variants(samples, seed):
    base = {k: v[0] for k, v in DEEP.items()}
    single = [{k: v} for k, vals in DEEP.items() for v in vals if v != base[k]]
    # several packs require constant 3 in the small stage, which the baseline lacks
    out = single + [dict(c, small_stage=st) for c in single for st in ('3,4', '3,4,-2')]
    rng = random.Random(seed)
    seen = {json.dumps(v, sort_keys=True) for v in out}
    target = len(out) + samples
    while len(out) < target:
        ch = {k: rng.choice([v for v in DEEP[k] if v != base[k]])
              for k in rng.sample(list(DEEP), rng.randint(2, 4))}
        if rng.random() < 0.6:
            ch['small_stage'] = rng.choice(('3,4', '3,4,-2'))
        key = json.dumps(ch, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(ch)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--samples', type=int, default=160)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=863)
    ap.add_argument('--deep', action='store_true', help='Part C: vary options hard-coded in the builder')
    args = ap.parse_args()
    jobs = deep_variants(args.samples, args.seed) if args.deep else variants(args.samples, args.seed)
    print(json.dumps(dict(jobs=len(jobs))), flush=True)
    with args.output.open('w') as out, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(one, jobs):
            out.write(json.dumps(row) + '\n')
            out.flush()
            if row.get('ok') and (row['bound'] <= 863 or row['effective'] < 51833):
                print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
