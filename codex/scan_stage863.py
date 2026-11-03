"""Reproducible structural scan starting from the verified 864 compiler.

Results are necessary bounds, never executable scores. Each full graph keeps
the original ISA, and prefix/suffix bounds allow every scalar/native mode.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

from codex import research_stage864 as stage
from codex import stride4_family as family
from codex.packet_budget_bound import boundary, stream_bounds
from codex.packet_modes import expand
from codex.word_ssa import rebuild


def baseline(**changes):
    config = dict(small_stage='4', stage_hash='final', stage_uniform='-60,-40',
        cheap_coeff=True, head_fill=True, head_flow=True, alu_layout=True,
        tail_transposes='29,31', tail_low=True, low_alu=True, tail_single=False,
        early_small_low_bank=False, sink=8, literals=2, literal_flow=7,
        early_literals=False, bootstrap_flow=2, bootstrap_early=2,
        scalar_leaves=True, scalar_root_head=False, root_flow_exchange=False,
        fold_raw_hash=True, low_bank_records=False)
    config.update(changes)
    return config


def measure(config, seconds=2):
    ir, builder = stage.build(SimpleNamespace(**config))
    counts = Counter(op.engine for op in ir.ops)
    work = counts['alu'] + 8 * counts['valu']
    expand(ir)
    ir.views = [a for _, aa, _, _ in ir.groups for a in aa]
    rebuild(ir)
    head = boundary(ir, 6, False, seconds, 2)
    tail = boundary(ir, 3, True, seconds, 2)
    streams = stream_bounds(ir)
    effective = work + head['waste_lower_bound'] + tail['waste_lower_bound']
    return dict(config=config, arithmetic_lanes=work, counts=dict(counts),
        prefix=head, suffix=tail, streams=streams, effective_budget=effective,
        cycle_lower_bound=max((effective + 59) // 60,
            *(v['cycle_lower_bound'] for v in streams.values())),
        scope='Necessary bound for this graph; no canonical startup fixed; '
              'not a feasible schedule or a universal algorithm bound')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seconds', type=float, default=2)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--group', choices=('staging', 'trade', 'root'), default='staging')
    args = ap.parse_args()
    family.capture = lru_cache(maxsize=1)(family.capture)
    if args.group == 'staging':
        variants = [dict(small_stage=s, stage_hash=h) for s, h in itertools.product(
            ('4', '4,-2', '4,9', '4,19', '4,9,19', '3,4', '3,4,-2', '3,4,-2,9,19'),
            ('final', 'coeff,final', 'coeff,add,final'))]
    elif args.group == 'root':
        variants = [dict(scalar_root_head=True, root_flow_exchange=e,
            tail_single=t, small_stage=s) for e,t,s in itertools.product(
                (False,True), (False,True), ('4','4,-2'))]
    else:
        variants = [dict(sink=s, literals=l, tail_single=t, small_stage=b,
            stage_hash=h) for s,l,t,b,h in itertools.product(
                (8,16,24), (0,2), (False,True),
                ('4','3,4,-2,9,19'), ('final',))]
    results = json.loads(args.output.read_text()) if args.resume and args.output.exists() else []
    for i, changes in enumerate(variants):
        config = baseline(**changes)
        if any(item['config'] == config for item in results):
            continue
        result = measure(config, args.seconds)
        result['id'] = i
        results.append(result)
        args.output.write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(dict(id=i, changes=changes,
            C=result['arithmetic_lanes'], counts=result['counts'],
            head=result['prefix']['waste_lower_bound'],
            bound=result['cycle_lower_bound'],
            streams={k:v['cycle_lower_bound'] for k,v in result['streams'].items()})),
            flush=True)


if __name__ == '__main__':
    main()
