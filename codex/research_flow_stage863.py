"""Joint address/FLOW and uniform-materialization experiments for the 864 IR.

Every operation uses the frozen ISA. Bounds are necessary conditions, not
executable scores. The standalone submissions are never modified here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
from types import SimpleNamespace

from codex import research_frontier863 as frontier
from codex import research_stage864 as stage
from codex import research_sub864 as research
from codex import stride4_family as family
from codex.packet_budget_bound import boundary, stream_bounds
from codex.packet_modes import expand
from codex.scan_stage863 import baseline
from codex.word_ssa import rebuild, kinds


@lru_cache(maxsize=1)
def original():
    return stage.build(SimpleNamespace(**baseline()))[1]


def fuse_digits(builder, groups):
    """For ordinary bits, select(b5, select(b6,3,2),b6) = 2*b5+b6."""
    writer = {v: op for op in builder.ops if op.engine != 'store'
              for v in op.args[0]}
    for g in groups:
        gather = next(op for op in builder.ops if op.tag == ('gather', (7, g, 0)))
        pointer = writer[gather.args[1][0]]
        digit = writer[pointer.args[1][0]]
        arm = writer[digit.args[2][0]]
        assert pointer.code == 'multiply_add'
        assert digit.code == arm.code == 'vselect'
        assert arm.args[1] == digit.args[3]
        assert arm.args[2:] == [builder.vectors[3], builder.vectors[2]]
        digit.engine, digit.code, digit.args, digit.kinds = (
            'valu', 'multiply_add',
            [digit.args[0], digit.args[1], builder.vectors[2], digit.args[3]],
            kinds('valu', 'multiply_add'))
        digit.tag = ('fused_d7_digit', (g,))
    builder.stats['fused_d7_digits'] = len(groups)


def materialize_uniforms(builder, methods, root_mode='alu'):
    """Replace native constant vectors by real memory or three-word FLOW packs.

    A method is 'stage' or 'third'. All selected vector definitions disappear;
    independent scalar roots are created before their consumers as necessary.
    For header lanes the FLOW pack makes three copies rather than forcing the
    live header to overlap a different contiguous view.
    """
    known, _ = family.evaluate(builder)
    original_ops = list(builder.ops)
    selected = {}
    for key, mode in methods.items():
        vec = builder.rootraw if key == 'root' else builder.vectors[int(key) & family.MASK]
        op = next(op for op in original_ops if op.engine == 'valu' and op.args[0] == vec)
        assert mode in ('stage', 'third')
        selected[key] = (vec, op, mode)
    blocked = {v for vec, _, _ in selected.values() for v in vec}
    writer = {v: op for op in original_ops if op.engine != 'store' for v in op.args[0]}
    # The first word of a vector being replaced is not an independent root.
    scalar_sources = {value: arg for value, arg in builder.scalar.items()
                      if len(arg) == 1 and arg[0] not in blocked and arg[0] in known}
    scalar_sources[0] = (builder.zero[0],)
    added = []

    def emit(engine, code, args, time=1, tag=('uniform_root', ())):
        builder.put(engine, code, args, time, tag)
        op = builder.ops.pop()
        added.append(op)
        return args[0]

    def root(value):
        value &= family.MASK
        if value in scalar_sources:
            return scalar_sources[value]
        dest = builder.new(1)
        # Prefer exact one-ALU forms from header/one/zero, independent of all
        # selected vectors. No search-generated or unvalidated constants.
        forms = {2: ('//', 16, 7), 3: ('cdiv', 16, 7),
                 9: ('-', 10, 1), 19: ('+', 16, 3),
                 family.MASK - 1: ('-', 0, 2)}
        if root_mode == 'alu' and value in forms:
            code, a, b = forms[value]
            left, right = root(a), root(b)
            emit('alu', code, [dest, left, right], 2)
        else:
            emit('flow', 'add_imm', [dest, (builder.zero[0],), value], 2)
        scalar_sources[value] = dest
        return dest

    replacements = {}
    packs = []
    for rank, (key, (vec, op, mode)) in enumerate(selected.items()):
        source = (builder.header[7],) if key == 'root' else root(int(key))
        if mode == 'stage':
            tag_id = -10000 - rank
            for lane in range(8):
                emit('store', 'store', [builder.bankptr[lane], source],
                     5 + lane // 2, ('constant_stage', (tag_id, lane)))
            out = builder.new(8)
            emit('load', 'vload', [out, builder.bankptr[0]], 9,
                 ('constant_reload', (tag_id,)))
        else:
            # Three originals plus five result words make eight equal words.
            standalone = (source[0] not in writer or
                          writer[source[0]].args[0] == source)
            block = list(source) if standalone else []
            while len(block) < 3:
                copy = builder.new(1)
                emit('alu', '|', [copy, source, (builder.zero[0],)], 2)
                block.extend(copy)
            block = tuple(block)
            full = builder.new(8)
            cond = builder.vectors[1][5:] + builder.zero[:5]
            yes = block + builder.new(5, undefined=True)
            no = builder.new(3, undefined=True) + block + builder.new(2, undefined=True)
            emit('flow', 'vselect', [full, cond, yes, no], 3,
                 ('third_uniform', (str(key),)))
            out = block + full[:5]
        replacements.update(zip(vec, out))
        builder.stats['uniform_' + mode] += 1
        packs.append((key, mode, source, out))
    removed = {id(op) for _, op, _ in selected.values()}
    kept = [op for op in original_ops if id(op) not in removed]
    # Replace scalar consumers as well. Existing scalar uses can share the
    # independent root rather than wait for a staged vector's reload.
    scalar_replacements = {}
    for key, (vec, _, _) in selected.items():
        if key != 'root':
            scalar_replacements[vec[0]] = root(int(key))[0]
    for op in kept:
        for i, (kind, arg) in enumerate(zip(op.kinds, op.args)):
            if kind != 'imm' and not kind.endswith('w'):
                mapping = scalar_replacements if len(arg) == 1 else replacements
                op.args[i] = tuple(mapping.get(v, v) for v in arg)
    # A remaining derived startup vector need not wait for a staged input.
    for value in (2, 3, 9, 19, family.MASK - 1):
        if value in methods or (value == family.MASK - 1 and -2 in methods):
            continue
        vec = builder.vectors[value]
        op = next((op for op in kept if op.engine == 'valu' and op.args[0] == vec), None)
        if op is not None and any(v in replacements.values() for arg in op.args[1:] for v in arg):
            source = root(value)
            # root(value) might still be the old vector's own scalar alias.
            if source[0] in vec:
                scalar_sources.pop(value)
                source = root(value)
            op.code, op.args, op.kinds = 'vbroadcast', [vec, source], kinds('valu', 'vbroadcast')
    # Memory epochs are deliberately before the first hash, as in stage_hash.
    insertion = next(i for i, op in enumerate(kept) if op.tag[0] == 'hash_mad')
    builder.ops = kept[:insertion] + added + kept[insertion:]
    return packs


def trade_loads(builder, count, include_input=False):
    """Trade existing literal LOADs for FLOW, optionally the second input ptr."""
    candidates = [op for op in builder.ops if op.engine == 'load' and op.code == 'const'
                  and (include_input or op.args[1] != 2318)]
    candidates.sort(key=lambda op: (op.args[1] == 2318, -op.time))
    if count > len(candidates):
        raise ValueError((count, len(candidates)))
    for op in candidates[:count]:
        dest, value = op.args
        if value == 2318:
            op.engine, op.code, op.args, op.kinds, op.time = (
                'alu', '|', [dest, builder.scalar[2310], builder.scalar[10]],
                kinds('alu', '|'), 1)
        else:
            op.engine, op.code, op.args, op.kinds = (
                'flow', 'add_imm', [dest, (builder.zero[0],), value],
                kinds('flow', 'add_imm'))
    builder.stats['additional_literal_trades'] = count


def early_4097(builder):
    """Use the input-pointer c0 LOAD for 4097, with a real c1 pointer OR."""
    pointer = next(op for op in builder.ops if op.code == 'const' and op.args[1] == 2318)
    pointer.engine, pointer.code, pointer.args, pointer.kinds, pointer.time = (
        'alu', '|', [pointer.args[0], builder.scalar[2310], builder.scalar[10]],
        kinds('alu', '|'), 1)
    vector = next(op for op in builder.ops if op.engine == 'valu'
                  and op.args[0] == builder.vectors[4097])
    scalar = builder.new(1)
    builder.put('load', 'const', [scalar, 4097], 0, ('early_4097', ()))
    builder.ops.insert(0, builder.ops.pop())
    vector.code, vector.args, vector.kinds, vector.time = (
        'vbroadcast', [vector.args[0], scalar], kinds('valu', 'vbroadcast'), 1)
    builder.scalar[4097] = scalar
    builder.stats['early_4097_literal'] = 1


def build(methods=None, fused=0, trades=0, root_mode='alu', include_input=False,
          load4097=False):
    builder = deepcopy(original())
    if load4097:
        early_4097(builder)
    fuse_digits(builder, list(range(fused)))
    if methods:
        materialize_uniforms(builder, methods, root_mode)
    trade_loads(builder, trades, include_input)
    frontier.logical_order(builder)
    builder.sink = builder.literals = builder.literal_flow = 0
    for _ in range(3):
        stage.prune_select_inputs(builder)
        ir = research.clean(builder)
    return ir, builder


def measure(ir, seconds=1):
    counts = Counter(op.engine for op in ir.ops)
    work = counts['alu'] + 8 * counts['valu']
    expand(ir)
    ir.views = [a for _, aa, _, _ in ir.groups for a in aa]
    rebuild(ir)
    head = boundary(ir, 6, False, seconds, 2)
    tail = boundary(ir, 3, True, seconds, 2)
    streams = stream_bounds(ir)
    return dict(counts=dict(counts), arithmetic_lanes=work, prefix=head, suffix=tail,
                streams=streams, cycle_lower_bound=max(
                    (work + head['waste_lower_bound'] + tail['waste_lower_bound'] + 59) // 60,
                    *(v['cycle_lower_bound'] for v in streams.values())),
                new_executable=False)


def serial_control(ir, seeds=3):
    return frontier.serial_control(ir, seeds)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seconds', type=float, default=1)
    ap.add_argument('--controls', action='store_true')
    args = ap.parse_args()
    if args.controls:
        configs = [dict(fused=3), dict(methods={9:'stage', 2310:'third'}, fused=2, trades=1),
                   dict(methods={3:'stage', -2:'third', 4097:'stage'}, fused=3, trades=2),
                   dict(methods={'root':'third', 2:'stage'}, fused=3, trades=1)]
        result = []
        for cfg in configs:
            ir, b = build(**cfg)
            row = dict(config=cfg, control=serial_control(ir), stats=dict(b.stats))
            result.append(row)
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(row), flush=True)
        return
    presets = [(), (9,), (-2,), (2310,), (4097,), (3,9,19,-2),
               (9,-2,2310), (4097,9,-2,2310),
               (16,2,3,9,19,-2,4097,0x7ed55d16,0xc761c23c,2310),
               (2,3,9,19,-2,4097,0xc761c23c,2310)]
    results = []
    for values in presets:
        for mode in ('stage', 'third'):
            for extra in (0, 2):
                # Examine the minimal plausible FLOW repair and two extra
                # fusions; actual Hall-window bounds determine feasibility.
                stages = len(values) if mode == 'stage' else 0
                trades = min(stages, 3)
                extra_flow = trades + (len(values) if mode == 'third' else 0) + (4097 in values)
                fused = max(0, (extra_flow - 1) // 2) + extra
                cfg = dict(methods={v:mode for v in values}, fused=fused, trades=trades)
                ir, b = build(**cfg)
                row = dict(config=cfg, **measure(ir, args.seconds))
                results.append(row)
                args.output.write_text(json.dumps(results, indent=2) + '\n')
                print(json.dumps(dict(config=cfg, C=row['arithmetic_lanes'],
                    bound=row['cycle_lower_bound'], head=row['prefix']['waste_lower_bound'],
                    streams={k:v['cycle_lower_bound'] for k,v in row['streams'].items()})), flush=True)


if __name__ == '__main__':
    main()
