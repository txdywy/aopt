"""Research alternate constant packs and late child transposes for 864.

All transformations emit the original ISA. This module never changes the
submission, simulator, or tests; each candidate needs scheduling and replay.
"""
from __future__ import annotations

import argparse
from collections import Counter
import heapq
from pathlib import Path
import pickle
from types import SimpleNamespace

from codex import research_frontier863 as frontier
from codex import research_sub864 as research
from codex import stride4_family as family
from codex.word_ssa import allocate, kinds, observed
from codex.research_packet_870 import verify


def prune_select_inputs(builder):
    """Decouple only inputs whose results cannot reach an observed value."""
    known, _ = family.evaluate(builder)
    needed = observed(SimpleNamespace(ops=builder.ops, known=known,
                                     undef=builder.undef))
    changes = 0
    for op in builder.ops:
        if op.code not in ('vselect', 'select'):
            continue
        for lane, dest in enumerate(op.args[0]):
            cond = op.args[1][lane]
            cv = known.get(cond)
            used = ({1, 3} if cv == 0 else {1, 2} if cv is not None
                    else {1, 2, 3}) if dest in needed else set()
            for k in range(1, 4):
                if k not in used and op.args[k][lane] not in builder.undef:
                    a = list(op.args[k])
                    a[lane] = builder.new(1, undefined=True)[0]
                    op.args[k] = tuple(a)
                    changes += 1
    return changes


def stage_hash(builder, which, uniforms=()):
    """Build shifted 9-word constant views with eight real scalar stores."""
    specs = []
    if 'coeff' in which:
        specs.append(('coeff', builder.hash_coeff_x, builder.hash_coeff_y))
    if 'add' in which:
        specs.append(('add', builder.hash_add_x, builder.hash_add_y))
    if 'final' in which:
        specs.append(('final', builder.hash_final_add, None))
    for value in uniforms:
        specs.append((str(value), builder.vectors[value & family.MASK], None))
    newops = []
    replacements = {}
    constants = {'coeff': (33, 16896), 'add': (3925396509, 2899272192),
                 'final': (0xfd7046c5,)}
    constants.update({str(v): (v & family.MASK,) for v in uniforms})
    for rank, (name, x, y) in enumerate(specs):
        # The source definitions remain standalone scalar words; the only
        # contiguous relationship is imposed by the reload and its consumers.
        sources = tuple(builder.scalar[v][0] for v in constants[name])
        prefix = sources[:1] if y is not None else ()
        key = -100 - rank
        for lane in range(8):
            source = sources[(lane + 1) % 2] if y is not None else sources[0]
            builder.put('store', 'store', [builder.bankptr[lane], (source,)],
                        5 + lane // 2, ('constant_stage', (key, lane)))
            newops.append(builder.ops.pop())
        out = builder.new(8)
        builder.put('load', 'vload', [out, builder.bankptr[0]], 9,
                    ('constant_reload', (key,)))
        newops.append(builder.ops.pop())
        if y is not None:
            view = prefix + out
            replacements[x], replacements[y] = view[:8], view[1:]
        else:
            replacements[x] = out
        builder.stats['hash_stage_' + name] += 1
    for op in builder.ops:
        for k, arg in enumerate(op.args):
            if op.kinds[k] != 'imm' and not op.kinds[k].endswith('w'):
                op.args[k] = replacements.get(arg, arg)
    insertion = next(j for j, op in enumerate(builder.ops)
                     if op.tag[0] == 'hash_mad')
    builder.ops[insertion:insertion] = newops


def cheap_coeff(builder):
    """Reuse a real preload pointer to avoid two scalar constant roots."""
    definitions = {v: op for op in builder.ops if op.engine != 'store'
                   for v in op.args[0]}
    p270 = definitions[builder.treeptr[33][0]]
    p270.code, p270.args, p270.time = ('&',
        [p270.args[0], builder.scalar[2047], builder.inptr[1]], 1)
    s33 = definitions[builder.hash_coeff_x[0]]
    s33.code, s33.args, s33.time = ('>>',
        [s33.args[0], p270.args[0], builder.scalar[3]], 3)
    scalar9 = builder.scalar[9]
    scale = definitions[builder.hash_coeff_x[1]]
    scale.code, scale.args, scale.time = ('<<',
        [scale.args[0], s33.args[0], scalar9], 4)
    builder.stats['cheap_coefficient_roots'] = 2


def rederive_head(builder):
    """Pull late scalar roots to c1 using exact known header operands."""
    known, _ = family.evaluate(builder)
    writer = {v: j for j, op in enumerate(builder.ops) if op.engine != 'store'
              for v in op.args[0]}
    seed = {known[v]: v for op in builder.ops if op.time == 0
            for v in op.args[0] if v in known and op.engine != 'store'}
    seed.update({0: builder.zero[0]})
    for v in builder.header[:7]:
        seed[known[v]] = v
    forms = {}
    for a, va in seed.items():
        for b, vb in seed.items():
            for code in ('+', '-', '*', '&', '|', '^', '>>', '<<', '//', '%', 'cdiv'):
                if code in ('//', '%', 'cdiv') and b == 0:
                    continue
                if code in ('<<', '>>') and b >= 32:
                    continue
                value = next(iter(family.calc(code, [frozenset([a]), frozenset([b])])))
                forms.setdefault(value, (code, (va,), (vb,)))
    changed = 0
    for j, op in enumerate(builder.ops):
        if op.engine != 'alu' or len(op.args[0]) != 1 or op.time <= 1:
            continue
        dest = op.args[0][0]
        if dest not in known or known[dest] not in forms:
            continue
        code, a, b = forms[known[dest]]
        if dest in a + b or any(writer.get(v, -1) == j for v in a + b):
            continue
        op.code, op.args, op.time = code, [op.args[0], a, b], 1
        changed += 1
    builder.stats['rederived_head_roots'] = changed


def head_flow(builder):
    """Trade one late FLOW literal for an early C1 FLOW, keeping counts."""
    definitions = {v: op for op in builder.ops if op.engine != 'store'
                   for v in op.args[0]}
    c0 = definitions[builder.scalar[0x7ed55d16][0]]
    c1 = definitions[builder.scalar[0xc761c23c][0]]
    b = definitions[builder.scalar[2899272192][0]]
    c5 = definitions[builder.scalar[0xb55a4f09][0]]
    assert c1.engine == 'load' and c1.code == 'const'
    late = max((op for op in builder.ops if op.engine == 'flow'
                and op.code == 'add_imm' and op.time >= 20), key=lambda op: op.time)
    builder.head_flow_exchange = late.args[0]
    late.engine, late.code, late.args, late.kinds = ('load', 'const',
        [late.args[0], late.args[2]], kinds('load', 'const'))
    for op, previous, value, previous_value, t in (
            (c1, c0, 0xc761c23c, 0x7ed55d16, 1),
            (b, c1, 2899272192, 0xc761c23c, 2),
            (c5, b, 0xb55a4f09, 2899272192, 3)):
        op.engine, op.code, op.args, op.kinds, op.time = (
            'flow', 'add_imm', [op.args[0], previous.args[0],
                (value - previous_value) & family.MASK], kinds('flow', 'add_imm'), t)
    builder.stats['head_flow_literal_exchange'] = 1


def tail_transposes(builder, groups, low_bank=False, low_alu=False, single_arm=False):
    """Replace two half selects by two real memory packs and one full select."""
    all_packs = []
    for group_rank, g in enumerate(groups):
        old = [op for op in builder.ops if op.tag[0] == 'child_select'
               and op.tag[1][:2] == (15, g)]
        old.sort(key=lambda op: op.tag[1][2])
        assert len(old) == 2
        packets = {op.tag[1][2]: op.args[0] for op in builder.ops
                   if op.tag[0] == 'gather' and op.tag[1][:2] == (14, g)}
        assert len(packets) == 8
        newops = []
        arms = []
        t = builder.hashhint[15, g] - 10
        bankptr = builder.bankptr
        if low_bank and group_rank == len(groups) - 1:
            five = family.Builder.value(builder, 'alu', '>>', builder.scalar[10],
                                        builder.scalar[1], time=1)
            if low_alu:
                builder.protected_alu = {five[0]}
            bankptr = [builder.scalar[0], builder.scalar[1], builder.scalar[2],
                       builder.scalar[3], builder.scalar[4], five,
                       builder.treeptr[0], builder.scalar[7]]
            builder.stats['low_tail_bank'] += 1
        for part, field in enumerate((1,) if single_arm else (1, 0)):
            key = -200 - 2 * g - part
            for lane in range(8):
                builder.put('store', 'store', [bankptr[lane],
                    (packets[lane][field],)], t + 4 * part + lane // 2,
                    ('constant_stage', (key, lane)))
                newops.append(builder.ops.pop())
            out = builder.new(8)
            builder.put('load', 'vload', [out, bankptr[0]],
                t + 4 * part + 4, ('constant_reload', (key,)))
            newops.append(builder.ops.pop())
            arms.append(out)
        if single_arm:
            # Ascending, overlapping gather destinations retain field zero.
            # The YES stores and node XORs read before later gathers clobber
            # fields one/two. Rebuild derives those ordinary scratch WARs.
            arms.append(tuple(packets[lane][0] for lane in range(8)))
            builder.stats['single_arm_tail_transposes'] += 1
        out = builder.new(8)
        builder.put('flow', 'vselect', [out, old[0].args[1], *arms],
                    builder.hashhint[15, g] - 3, ('child_transpose', (15, g)))
        select = builder.ops.pop()
        replace = {old[part].args[0][2 * i]: out[2 * i + part]
                   for part in range(2) for i in range(4)}
        kept = []
        for op in builder.ops:
            if any(op is a for a in old):
                if op is old[0]:
                    kept.append(select)
                continue
            for k, arg in enumerate(op.args):
                if op.kinds[k] != 'imm' and not op.kinds[k].endswith('w'):
                    op.args[k] = tuple(replace.get(v, v) for v in arg)
            kept.append(op)
        all_packs.extend(newops)
        builder.ops = kept
        builder.stats['tail_child_transposes'] += 1
    # Low memory contains d7 record words 6/7 until the final r7 gather. All
    # tail packs are source-ordered after the complete gather stream, so its
    # ordinary WAR edges protect those words without assumptions about time.
    insertion = next(j for j, op in enumerate(builder.ops) if op.tag[0] == 'output')
    builder.ops[insertion:insertion] = all_packs


def alu_layout_roots(builder):
    definitions = {v: op for op in builder.ops if op.engine != 'store'
                   for v in op.args[0]}
    for value, code, a, b, time in (
            (-60, '*', builder.scalar[0xfffffffe], builder.treeptr[3], 3),
            (-40, '-', builder.treeptr[0], builder.treeptr[5], 4)):
        op = definitions[builder.scalar[value & family.MASK][0]]
        assert op.engine == 'flow'
        op.engine, op.code, op.args, op.kinds, op.time = (
            'alu', code, [op.args[0], a, b], kinds('alu', code), time)
    builder.stats['alu_layout_roots'] = 2


def bootstrap_flow(builder, count, early_count=2):
    """Move large hash literals to FLOW so c2 LOAD slots can read inputs."""
    definitions = {v: op for op in builder.ops if op.engine != 'store'
                   for v in op.args[0]}
    previous = builder.scalar[2899272192]
    previous_value = 2899272192
    sequence=(((0xb55a4f09,3),(3925396509,4)) if early_count==1 else
              ((0xfd7046c5,3),(3925396509,4),(0xb55a4f09,5)))
    for value, t in sequence:
        out = builder.scalar[value]
        op = definitions[out[0]]
        assert op.engine in ('load', 'flow')
        op.engine, op.code, op.args, op.kinds, op.time = (
            'flow', 'add_imm', [out, previous, (value - previous_value) & family.MASK],
            kinds('flow', 'add_imm'), t)
        previous, previous_value = out, value
    frontier.logical_order(builder)
    known, _ = family.evaluate(builder)
    uses = {}
    for op in builder.ops:
        for kind, arg in zip(op.kinds, op.args):
            if kind != 'imm' and not kind.endswith('w'):
                for v in arg:
                    uses[v] = min(uses.get(v, 10**9), op.time)
    candidates = [op for op in builder.ops if op.engine == 'load'
                  and op.code == 'const' and op.args[1] != 2318]
    candidates.sort(key=lambda op: uses.get(op.args[0][0], 0), reverse=True)
    assert count>=early_count
    for op in candidates[:count - early_count]:
        out, value = op.args
        op.engine, op.code, op.args, op.kinds, op.time = (
            'flow', 'add_imm', [out, builder.scalar[0], value],
            kinds('flow', 'add_imm'), max(8, uses[out[0]] - 1))
    # Far-depth arm literals may wait for a FLOW slot after the first r1
    # selects. This is a scheduling hint only, never a release constraint.
    for op in builder.ops:
        if op.engine == 'flow' and op.code == 'add_imm' and known.get(op.args[0][0]) in (
                1026, 1027, 1028, 1029):
            op.time = 40
    builder.stats['bootstrap_loads_to_flow'] = count
    builder.stats['bootstrap_early_literals'] = early_count


def scalar_leaf_reads(builder):
    """Bias d9 pointers by three so d10 can use scalar payload loads."""
    definitions = {v: op for op in builder.ops if op.engine != 'store'
                   for v in op.args[0]}
    old_scalars = {v: builder.scalar[v] for v in range(1026, 1030)}
    old_vectors = {v: builder.vectors[v] for v in range(1026, 1030)}
    for value, source in old_scalars.items():
        op = definitions[source[0]]
        assert op.code in ('const', 'add_imm')
        op.args[-1] += 3
    for mapping, old in ((builder.scalar, old_scalars), (builder.vectors, old_vectors)):
        for value in old:
            mapping.pop(value)
        mapping.update({value + 3: source for value, source in old.items()})
    replacement = {}
    for op in builder.ops:
        if op.tag[0] != 'raw_gather':
            continue
        r, _, _ = op.tag[1]
        if r == 9:
            replacement[op.args[0][6]] = op.args[0][3]
        elif r == 10:
            op.code, op.args, op.kinds = ('load', [(op.args[0][6],), op.args[1]],
                                        kinds('load', 'load'))
    for op in builder.ops:
        for k, arg in enumerate(op.args):
            if op.kinds[k] != 'imm' and not op.kinds[k].endswith('w'):
                op.args[k] = tuple(replacement.get(v, v) for v in arg)
    targets = {}
    for g in range(32):
        rows = [op for op in builder.ops if op.tag == ('node_xor', (10, g))]
        assert len(rows) == 8
        out = tuple(op.args[0][0] for op in rows)
        left = tuple(op.args[1][0] for op in rows)
        right = tuple(op.args[2][0] for op in rows)
        vector = SimpleNamespace(engine='valu', code='^', args=[out, left, right],
            kinds=kinds('valu', '^'), time=min(op.time for op in rows), tag=('node_xor', (10, g)))
        targets[id(rows[0])] = vector
        targets.update({id(op): None for op in rows[1:]})
    builder.ops = [targets.get(id(op), op) for op in builder.ops
                   if id(op) not in targets or targets[id(op)] is not None]
    builder.stats['scalar_leaf_payloads'] = 256
    builder.stats['additional_leaf_xor_groups'] = 32


def early_small_low_bank(builder):
    """Stage four in consumed header memory before the first d7 image write.

    Reuse the late transpose's existing 0..7 pointers. Distinct early tags
    keep the ordinary constant-pack sorter from moving this epoch past d7.
    Rebuilding memory edges protects the original root read and later image.
    """
    source = builder.scalar[4][0]
    stages = [op for op in builder.ops
              if op.tag[0] == 'constant_stage' and op.tag[1][0] == source]
    reloads = [op for op in builder.ops if op.tag == ('constant_reload', (source,))]
    assert len(stages) == 8 and len(reloads) == 1
    tail = [op for op in builder.ops if op.tag[0] == 'constant_stage'
            and op.tag[1][0] == -262]
    assert len(tail) == 8, 'Requires the existing group-31 low transpose bank'
    pointers = {op.tag[1][1]: op.args[0] for op in tail}
    for op in stages:
        lane = op.tag[1][1]
        op.args[0] = pointers[lane]
        op.tag = ('early_constant_stage', op.tag[1])
        op.time = 5 + lane // 2
    reload = reloads[0]
    reload.args[1] = pointers[0]
    reload.tag = ('early_constant_reload', reload.tag[1])
    reload.time = 10
    early = sorted(stages, key=lambda op: op.tag[1][1]) + reloads
    moved = {id(op) for op in early}
    builder.ops = [op for op in builder.ops if id(op) not in moved]
    insertion = next(j for j, op in enumerate(builder.ops) if op.tag[0] == 'd7_image')
    builder.ops[insertion:insertion] = early
    builder.stats['early_small_low_bank'] = 1


def fold_raw_hash(builder):
    """Restore the raw hash alongside its shift instead of after the hash.

    The two XOR destinations exchange semantic roles. Keeping both SSA names
    lets a verified old calendar transfer exactly by swapping their packets.
    """
    definitions = {v: op for op in builder.ops if op.engine != 'store'
                   for v in op.args[0]}
    constant = builder.vectors[0xb55a4f09]
    shift_constant = builder.vectors[16]
    pairs = []
    for restore in builder.ops:
        if (restore.engine, restore.code) != ('valu', '^') or restore.args[2] != constant:
            continue
        final = definitions.get(restore.args[1][0])
        if final is None or (final.engine, final.code) != ('valu', '^'):
            continue
        shift = definitions.get(final.args[2][0])
        if (shift is None or shift.code != '>>' or shift.args[2] != shift_constant
                or shift.args[1] != final.args[1]):
            continue
        users = {id(op) for op in builder.ops for kind, arg in zip(op.kinds, op.args)
                 if kind != 'imm' and not kind.endswith('w')
                 and set(arg).intersection(final.args[0])}
        assert users == {id(restore)}, 'The encoded hash has another consumer'
        pairs.append((restore, final, shift, final.args[1]))
    assert len(pairs) == 64, len(pairs)
    replacement = {v: h for restore, final, _, _ in pairs
                   for v, h in zip(restore.args[0], final.args[0])}
    for op in builder.ops:
        for k, (kind, arg) in enumerate(zip(op.kinds, op.args)):
            if kind != 'imm' and not kind.endswith('w'):
                op.args[k] = tuple(replacement.get(v, v) for v in arg)
    swaps = {}
    for restore, final, shift, e in pairs:
        restore.args = [restore.args[0], e, constant]
        final.args = [final.args[0], restore.args[0], shift.args[0]]
        restore.time, final.time = shift.time, shift.time + 1
        swaps.update(zip(restore.args[0], final.args[0]))
        swaps.update(zip(final.args[0], restore.args[0]))
    builder.raw_hash_swaps = swaps
    builder.stats['raw_hash_parallel_restorations'] = len(pairs)


def scalar_root_head(builder, flow_exchange=False):
    """Replace the root broadcast by scalar XORs and load three at cycle zero.

    Header lane six OR lane three is 2318, so the second input pointer needs
    one real scalar ALU instead of a literal LOAD. Broadcasting three early
    then exposes the independent four/nine/nineteen vectors at cycle two.
    """
    definitions = {v: op for op in builder.ops if op.engine != 'store'
                   for v in op.args[0]}
    root = definitions[builder.rootraw[0]]
    assert root.engine == 'valu' and root.code == 'vbroadcast'
    root_words = set(root.args[0])
    rows = []
    changed = 0
    for op in builder.ops:
        if op.engine == 'valu' and op.code == '^' and set(op.args[2]) == root_words:
            for lane in range(8):
                rows.append(SimpleNamespace(engine='alu', code='^',
                    args=[(op.args[0][lane],), (op.args[1][lane],), (builder.header[7],)],
                    kinds=kinds('alu', '^'), time=op.time, tag=op.tag))
            changed += 1
        else:
            rows.append(op)
    assert changed == 32, changed
    builder.ops = rows
    three = definitions[builder.vectors[3][0]]
    assert three.engine == 'valu' and three.code == '+'
    scalar_three = builder.new(1)
    builder.put('load', 'const', [scalar_three, 3], 0, ('scalar_root_three', ()))
    builder.ops.insert(0, builder.ops.pop())
    three.code, three.args, three.kinds, three.time = (
        'vbroadcast', [three.args[0], scalar_three], kinds('valu', 'vbroadcast'), 1)
    ptr = definitions[builder.inptr[1][0]]
    assert ptr.engine == 'load' and ptr.code == 'const' and ptr.args[1] == 2318
    ptr.engine, ptr.code, ptr.args, ptr.kinds, ptr.time = (
        'alu', '|', [ptr.args[0], builder.scalar[2310], builder.scalar[10]],
        kinds('alu', '|'), 1)
    if flow_exchange:
        late = definitions[builder.head_flow_exchange[0]]
        assert late.engine == 'load' and late.code == 'const'
        late.engine, late.code, late.args, late.kinds = (
            'flow', 'add_imm', [late.args[0], builder.scalar[0], late.args[1]],
            kinds('flow', 'add_imm'))
    builder.stats['scalar_initial_root_xors'] = changed * 8
    builder.stats['root_broadcast_to_three_load'] = 1
    builder.stats['root_pointer_flow_exchange'] = int(flow_exchange)


def low_bank_records(builder):
    """Move the first four d7 records, freeing header memory for every bank.

    The d5 anchor for those records is the existing header word 2054. Moving
    both adjacent 8-word images keeps the same four-stride d7 address MAD.
    Only the second image needs a new pointer: 2062 = 2054 | 10. The later
    heap conversion still needs the logical address, so it must mask off
    this 2048 relocation. Those 256 real ALUs make this a negative control.
    """
    frontier.logical_order(builder)
    known,_=family.evaluate(builder)
    five=next((op.args[0] for op in builder.ops if op.engine!='store'
               and len(op.args[0])==1 and known.get(op.args[0][0])==5),None)
    if five is None:
        five=family.Builder.value(builder,'alu','>>',builder.scalar[10],
                                  builder.scalar[1],time=1)
    second=family.Builder.value(builder,'alu','|',builder.scalar[2054],
                                builder.scalar[10],time=1)
    # A table field must not force the complete live header into its native
    # vstore layout. Retain a real standalone anchor copy for that view.
    anchor=family.Builder.value(builder,'alu','|',builder.scalar[2054],
                                builder.scalar[0],time=1)
    low=[builder.scalar[0],builder.scalar[1],builder.scalar[2],builder.scalar[3],
         builder.scalar[4],five,builder.treeptr[0],builder.scalar[7]]
    pointers={old:new for old,new in zip(builder.bankptr,low)}
    changed_anchor=0
    for op in builder.ops:
        name,coords=op.tag
        if name=='d7_image' and coords[0]<2:
            op.args[0]=builder.scalar[2054] if coords[0]==0 else second
        elif name=='image':
            words=list(op.args[1])
            for lane,v in enumerate(words):
                if v==builder.treeptr[0][0]:
                    words[lane]=anchor[0]
                    changed_anchor+=1
            op.args[1]=tuple(words)
        if op.engine=='store' and name in ('constant_stage','anchor_store'):
            op.args[0]=pointers.get(op.args[0],op.args[0])
        elif op.engine=='load' and name in ('constant_reload','anchor'):
            op.args[1]=pointers.get(op.args[1],op.args[1])
    assert changed_anchor==1,changed_anchor
    builder.bankptr=low
    for g in range(32):
        gathers=sorted((op for op in builder.ops if op.tag[0]=='raw_gather'
                        and op.tag[1][:2]==(9,g)),key=lambda op:op.tag[1][2])
        assert len(gathers)==8
        raw=tuple(op.args[1][0] for op in gathers)
        masked=builder.new(8)
        replacement=dict(zip(raw,masked))
        for op in builder.ops:
            for k,arg in enumerate(op.args):
                if op.kinds[k]!='imm' and not op.kinds[k].endswith('w'):
                    op.args[k]=tuple(replacement.get(v,v) for v in arg)
        for lane in range(8):
            builder.put('alu','&',[(masked[lane],),(raw[lane],),builder.scalar[2047]],
                        builder.hashhint[9,g]-4,('relocation_heap_mask',(g,lane)))
    builder.stats['relocated_d7_records']=4
    builder.stats['header_memory_bank']=1
    builder.stats['relocation_heap_mask_lanes']=256


def build(args):
    cfg = dict(layout='shared', small_stage=args.small_stage, extra_stage='',
        coefficients=True, hash_pack=True, replicate_pack=False, native_pack='',
        head4097=True, head_full=True, cache_vector=False, native_cache='',
        sink=0, literals=0, literal_flow=args.literal_flow, header_pack='',
        joint_pack=True, early_input=True, stride16=True)
    _, builder = frontier.build(SimpleNamespace(**cfg))
    if args.cheap_coeff:
        cheap_coeff(builder)
    if args.head_fill:
        frontier.logical_order(builder)
        rederive_head(builder)
    if args.alu_layout:
        alu_layout_roots(builder)
    if args.stage_hash or args.stage_uniform:
        stage_hash(builder, args.stage_hash.split(','),
                   [int(v, 0) for v in args.stage_uniform.split(',') if v])
    if args.tail_transposes:
        tail_transposes(builder, [int(g) for g in args.tail_transposes.split(',')],
                        args.tail_low, args.low_alu, args.tail_single)
    if args.early_small_low_bank:
        assert args.small_stage == '4' and args.tail_low
        early_small_low_bank(builder)
    frontier.logical_order(builder)
    # Delete dormant half-select inputs before ordinary dead-definition CSE.
    for _ in range(3):
        prune_select_inputs(builder)
        builder.sink = builder.literals = builder.literal_flow = 0
        research.clean(builder)
    builder.sink, builder.literals = args.sink, args.literals
    ir = research.clean(builder)
    if args.head_flow:
        head_flow(builder)
        frontier.logical_order(builder)
        builder.sink = builder.literals = 0
        ir = research.clean(builder)
    if args.early_literals:
        for op in builder.ops:
            if op.engine == 'load' and op.code == 'const' and op.time > 20:
                op.time = 2
        ir = research.clean(builder)
    if args.bootstrap_flow:
        assert args.head_flow
        bootstrap_flow(builder, args.bootstrap_flow, min(args.bootstrap_early,args.bootstrap_flow))
        frontier.logical_order(builder)
        ir = research.clean(builder)
    if args.scalar_leaves:
        scalar_leaf_reads(builder)
        frontier.logical_order(builder)
        ir = research.clean(builder)
    if args.scalar_root_head:
        assert args.head_flow
        scalar_root_head(builder, args.root_flow_exchange)
        frontier.logical_order(builder)
        ir = research.clean(builder)
    if args.fold_raw_hash:
        fold_raw_hash(builder)
        frontier.logical_order(builder)
        ir = research.clean(builder)
        ir.raw_hash_swaps = builder.raw_hash_swaps
    if args.low_bank_records:
        low_bank_records(builder)
        frontier.logical_order(builder)
        ir=research.clean(builder)
    return ir, builder


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix', type=Path, required=True)
    ap.add_argument('--small-stage', default='')
    ap.add_argument('--early-small-low-bank', action='store_true')
    ap.add_argument('--stage-hash', default='')
    ap.add_argument('--stage-uniform', default='')
    ap.add_argument('--cheap-coeff', action='store_true')
    ap.add_argument('--head-fill', action='store_true')
    ap.add_argument('--head-flow', action='store_true')
    ap.add_argument('--tail-transposes', default='')
    ap.add_argument('--tail-single', action='store_true')
    ap.add_argument('--tail-low', action='store_true')
    ap.add_argument('--low-alu', action='store_true')
    ap.add_argument('--alu-layout', action='store_true')
    ap.add_argument('--early-literals', action='store_true')
    ap.add_argument('--bootstrap-flow', type=int, choices=(0, 1, 2, 3, 4), default=0)
    ap.add_argument('--bootstrap-early', type=int, choices=(1, 2), default=2)
    ap.add_argument('--scalar-leaves', action='store_true')
    ap.add_argument('--fold-raw-hash', action='store_true')
    ap.add_argument('--scalar-root-head', action='store_true')
    ap.add_argument('--root-flow-exchange', action='store_true')
    ap.add_argument('--low-bank-records', action='store_true')
    ap.add_argument('--sink', type=int, default=4)
    ap.add_argument('--literals', type=int, default=0)
    ap.add_argument('--literal-flow', type=int, default=7)
    ap.add_argument('--serial', action='store_true')
    args = ap.parse_args()
    assert set(args.stage_hash.split(',')) <= {'', 'coeff', 'add', 'final'}
    ir, builder = build(args)
    counts = Counter(op.engine for op in ir.ops)
    print({'C': counts['alu'] + 8 * counts['valu'], 'counts': dict(counts),
           'changes': dict(builder.stats)}, flush=True)
    if args.serial:
        degree = list(map(len, ir.deps))
        ready = [j for j, d in enumerate(degree) if not d]
        heapq.heapify(ready)
        times = [-1] * len(ir.ops)
        step = 0
        while ready:
            j = heapq.heappop(ready)
            times[j] = step
            step += 1
            for q, _ in ir.succ[j]:
                degree[q] -= 1
                if not degree[q]:
                    heapq.heappush(ready, q)
        assert step == len(ir.ops), 'cyclic experimental IR'
        program, words, _ = allocate(ir, times)
        assert words <= 1536, words
        print({'scratch': words, 'serial_verified': verify(program, 3)}, flush=True)
    with Path(str(args.prefix) + '.ir.pkl').open('wb') as stream:
        pickle.dump(ir, stream)


if __name__ == '__main__':
    main()
