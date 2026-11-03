"""Charge actual record-layout copies when moving d8 XORs into the loop.

Raw vload words cannot alias the reversed child pairs of d7 vstore records.
Real scalar copies replace the removed XORs, and state restoration adds 256
arithmetic lanes. The naive cost-neutral estimate therefore does not apply.
"""
from copy import deepcopy
from collections import Counter
import argparse
import json
from pathlib import Path

from codex import research_flow_stage863 as flow
from codex import research_frontier863 as frontier
from codex import research_scalar_uniform863 as scalar
from codex import research_stage864 as stage
from codex import research_sub864 as research
from codex.packet_modes import expand
from codex.word_ssa import rebuild


def raw_children(builder):
    writers = {v: op for op in builder.ops if op.engine != 'store' for v in op.args[0]}
    copied = set()
    contradiction = None
    for op in builder.ops:
        if op.tag[0] != 'd7_image':
            continue
        assert op.code == 'vstore' and len(op.args[1]) == 8
        payload = list(op.args[1])
        if contradiction is None:
            raw = [writers[payload[lane]].args[1][0] for lane in (0, 1)]
            source = writers[raw[0]]
            assert source is writers[raw[1]] and source.code == 'vload'
            offsets = [source.args[0].index(v) for v in raw]
            assert offsets[1]-offsets[0] == -1
            contradiction = dict(source_tag=source.tag, source_offsets=offsets,
                source_difference=-1, required_record_difference=1,
                result='direct_raw_alias_has_inconsistent_contiguity')
        for lane in (0, 1, 4, 5):
            definition = writers[payload[lane]]
            assert definition.engine == 'alu' and definition.code == '^'
            assert definition.args[2] == builder.scalar[0xb55a4f09]
            copied.add(id(definition))
            definition.code = '|'
            definition.args[2] = (builder.zero[0],)
            definition.tag = ('raw_d8_layout_copy', (op.tag[1][0], lane))
    assert len(copied) == 256
    for group in range(32):
        xors = [op for op in builder.ops if op.tag == ('node_xor', (8, group))]
        assert len(xors) == 8 and all(op.engine == 'alu' and op.code == '^' for op in xors)
        state = tuple(op.args[1][0] for op in xors)
        restored = builder.new(8)
        builder.put('valu', '^', [restored, state, builder.vectors[0xb55a4f09]],
                    min(op.time for op in xors)-1, ('raw_d8_restore', (group,)))
        for lane, op in enumerate(xors):
            op.args[1] = (restored[lane],)
    builder.raw_d8_alias_obstruction = contradiction
    builder.stats['raw_d8_payload_copies'] = len(copied)
    builder.stats['raw_d8_restores'] = 32


def build(config):
    builder = deepcopy(flow.original())
    raw_children(builder)
    initializer = config.get('init4097')
    if initializer == 'flow':
        scalar.flow_4097(builder)
    elif initializer == 'load':
        flow.early_4097(builder)
    elif initializer is not None:
        raise ValueError('unknown initializer')
    for name in config.get('full', ()):
        scalar.scalar_uniform(builder, name)
    flow.fuse_digits(builder, list(range(config.get('fused', 0))))
    frontier.logical_order(builder)
    methods = {key if key == 'root' else int(key): value
               for key, value in config.get('methods', {}).items()}
    packs = flow.materialize_uniforms(builder, methods) if methods else []
    for key, _, source, out in packs:
        if key != 'root':
            builder.vectors[int(key) & 0xffffffff] = out
            builder.scalar[int(key) & 0xffffffff] = source
    if config.get('prolog19'):
        pack = next((p for p in packs if p[0] == 19), None)
        scalar.scalar_uniform(builder, 19, True, vector=pack[3] if pack else None,
            source=pack[2] if pack else None, head_groups=config['prolog19'])
    flow.trade_loads(builder, config.get('trades', 0))
    frontier.logical_order(builder)
    builder.sink = builder.literals = builder.literal_flow = 0
    for _ in range(3):
        stage.prune_select_inputs(builder)
        ir = research.clean(builder)
    return ir, builder


def measure(ir):
    counts = Counter(op.engine for op in ir.ops)
    work = counts['alu'] + 8*counts['valu']
    expand(ir)
    ir.views = [arg for _, args, _, _ in ir.groups for arg in args]
    rebuild(ir)
    grouped = {j for _, _, ids, _ in ir.groups for j in ids}
    mandatory = sum(op.engine == 'alu' and j not in grouped for j, op in enumerate(ir.ops))
    return dict(counts=dict(counts), arithmetic_lanes=work,
        arithmetic_only_cycle_lower_bound=(work+59)//60, mandatory_alu=mandatory,
        mandatory_alu_cycle_bound=(mandatory+11)//12,
        cycle_lower_bound=max((work+59)//60, (mandatory+11)//12), new_executable=False,
        scope='Arithmetic work and mandatory-ALU necessary bounds only; no startup/drain '
              'or other engine bound needed to exclude 863 in these two graphs.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--controls', action='store_true')
    args = parser.parse_args()
    core = ['cache0', 0xb55a4f09]
    configs = [{},
        dict(full=core+[-2, 0xc761c23c], methods={9: 'stage', 19: 'stage', 3: 'third'},
             prolog19=8, init4097='flow')]
    report = []
    for config in configs:
        ir, builder = build(config)
        row = dict(config=config, **measure(deepcopy(ir)), stats=dict(builder.stats),
            alias_obstruction=builder.raw_d8_alias_obstruction)
        if args.controls:
            row['serial_control_seeds'] = 2
            row['serial_control'] = flow.serial_control(ir, 2)
        report.append(row)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k: row[k] for k in ('config', 'arithmetic_lanes', 'mandatory_alu',
                                             'cycle_lower_bound')}), flush=True)


if __name__ == '__main__':
    main()
