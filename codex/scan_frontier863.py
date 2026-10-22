"""Compare startup graph changes by a relaxed exact boundary budget."""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import itertools
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

from codex import research_sub863 as previous
from codex import research_sub864 as research
from codex import stride4_family as family
from codex.packet_budget_bound import boundary, lane_weight
from codex.packet_modes import expand
from codex.research_frontier863 import pack_header, build as frontier_build
from codex.word_ssa import rebuild, observed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--group', choices=('base', 'cache', 'headers', 'pipeline', 'c5'), default='base')
    ap.add_argument('--ortools-path', type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.ortools_path))
    family.capture = lru_cache(maxsize=1)(family.capture)
    common = dict(layout='shared', small_stage='3,4,-2', extra_stage='',
        coefficients=True, hash_pack=True, replicate_pack=False, native_pack='',
        head4097=False, head_full=False, cache_vector=False, native_cache='',
        sink=0, literals=0, literal_flow=6, header_pack='')
    variants = []
    if args.group == 'base':
        for stage, head, extra in itertools.product(
                ('', '3,4,-2', '3,4,-2,9', '3,4,-2,9,19'), (False, True),
                ('', '2310,-60,-40')):
            variants.append(dict(small_stage=stage, head4097=head,
                head_full=head, extra_stage=extra))
    elif args.group == 'cache':
        for stage, head, native in itertools.product(
                ('3,4,-2', '3,4,-2,9', '3,4,-2,9,19'), (False, True),
                ('', '1', '1,2')):
            variants.append(dict(small_stage=stage, head4097=head,
                head_full=head, cache_vector=True, native_cache=native))
    elif args.group == 'headers':
        for stage, head, pack in itertools.product(
                ('3,4,-2', '3,4,-2,9'), (False, True),
                ('>>:1', '-:2', '+:16', '>>:1,-:2', '>>:1,+:16', '-:2,+:16',
                 '>>:1,-:2,+:16')):
            variants.append(dict(small_stage=stage, head4097=head,
                head_full=head, header_pack=pack))
    elif args.group == 'pipeline':
        common.update(joint_pack=True, early_input=True, stride16=True,
            head4097=True, head_full=True, sink=4, literal_flow=7)
        for stage, pack in itertools.product(
                ('', '3', '4', '-2', '3,4', '3,-2', '4,-2', '3,4,-2'),
                ('', '+:1', '>>:1', '-:2', '+:1,>>:1', '+:1,-:2')):
            variants.append(dict(small_stage=stage, header_pack=pack))
    else:
        common.update(joint_pack=True, early_input=True, stride16=True,
            head4097=True, head_full=True, head_c5=True, cache_vector=True,
            sink=4, literal_flow=7)
        for stage, native in itertools.product(
                ('', '3', '3,4', '3,4,-2'),
                ('coeff', 'coeff,addend', 'coeff,addend,final')):
            variants.append(dict(small_stage=stage, native_pack=native))
    results = []
    for k, changes in enumerate(variants):
        config = dict(common, **changes)
        if args.group in ('pipeline', 'c5'):
            ir, builder = frontier_build(SimpleNamespace(**config))
        else:
            ir, builder = previous.build(SimpleNamespace(**config))
        if config['header_pack'] and args.group not in ('pipeline', 'c5'):
            pack_header(builder, config['header_pack'].split(','))
            builder.sink = builder.literals = builder.literal_flow = 0
            ir = research.clean(builder)
        counts = Counter(op.engine for op in ir.ops)
        work = counts['alu'] + 8 * counts['valu']
        expanded = expand(ir)
        expanded.views = [a for _, aa, _, _ in expanded.groups for a in aa]
        rebuild(expanded)
        needed = observed(expanded)
        productive_work = sum(lane_weight(expanded, j, True, needed)
            for j in range(len(expanded.ops)))
        head = boundary(expanded, 6, False, 4, 2, True)
        tail = boundary(expanded, 3, True, 3, 2, True)
        effective = productive_work + head['waste_lower_bound'] + (
            tail['waste_lower_bound'])
        item = dict(id=k, config=config, arithmetic_lanes=work, counts=dict(counts),
            head=head, tail=tail, productive_lanes=productive_work,
            bound_kind='productive arithmetic; independent boundary relaxations',
            effective_budget=effective,
            cycle_lower_bound=(effective + 59) // 60)
        results.append(item)
        print(json.dumps({key: item[key] for key in ('id', 'arithmetic_lanes',
            'effective_budget', 'cycle_lower_bound')} | {
            'head': head['waste_lower_bound'], 'L': counts['load'],
            'F': counts['flow'], 'changes': changes}), flush=True)
        args.output.write_text(json.dumps(results, indent=2) + '\n')
        if effective <= 51840:
            with args.output.with_name(args.output.stem + f'-{k}.pkl').open('wb') as f:
                pickle.dump(expanded, f)


if __name__ == '__main__':
    main()
