"""Experimental constant, pointer, and terminal-table record-kernel variants.

Emits research IR only. Native header operations execute all eight lanes, and
only independent standalone scalar definitions may be replaced by their words.
"""
from __future__ import annotations

import argparse
from collections import Counter
import heapq
from pathlib import Path
import pickle
from functools import partial

from codex import research_sub863 as previous
from codex import research_sub864 as research
from codex import stride4_family as family
from codex.research_packet_870 import verify
from codex.word_ssa import allocate, kinds


class Builder(previous.Builder):
    def __init__(self, *, joint_pack=False, terminal_count=0,
                 early_input=False, scalar_shifts=0, stride16=False,
                 cheap_coeff=False, pointer_pack=False, head_c5=False, **kwargs):
        self.joint_pack = joint_pack
        self.terminal_count = terminal_count
        self.early_input = early_input
        self.scalar_shifts = scalar_shifts
        self.stride16 = stride16
        self.cheap_coeff = cheap_coeff
        self.pointer_pack = pointer_pack
        self.head_c5 = head_c5
        super().__init__(**kwargs)

    def setup(self):
        if not self.joint_pack:
            return super().setup()
        saved = self.hash_pack
        self.hash_pack = False
        super().setup()
        self.hash_pack = saved
        assert saved
        if 'coeff' in self.native_pack:
            s33 = self.scalar[33]
        else:
            old33 = self.scalar[33]
            insertion = next(j for j, op in enumerate(self.ops)
                if any(k != 'imm' and not k.endswith('w') and a == old33
                    for k, a in zip(op.kinds, op.args)))
            s32 = self.s('+', self.scalar[16], self.scalar[16], 1)
            s33 = self.s('+', s32, self.scalar[1], 2)
            producers = self.ops[-2:]
            del self.ops[-2:]
            self.ops[insertion:insertion] = producers
            for op in self.ops:
                for k, arg in enumerate(op.args):
                    if op.kinds[k] != 'imm' and not op.kinds[k].endswith('w') and arg == old33:
                        op.args[k] = s33
            self.scalar[33] = s33
        ca = (tuple(self.vectors[33][i] for i in (0, 2, 4))
              if 'coeff' in self.native_pack else self.copies(s33, 3, 3))
        cb = self.copies(self.scalar[16896], 2, 4)
        coefficients = (ca[0], cb[0], ca[1], cb[1], ca[2])
        aa = (tuple(self.vectors[3925396509][i] for i in (0, 2, 4))
              if 'addend' in self.native_pack
              else self.copies(self.scalar[3925396509], 3, 3))
        ab = self.copies(self.scalar[2899272192], 2, 4)
        addends = (aa[0], ab[0], aa[1], ab[1], aa[2])
        mask = self.vectors[1][4:] + self.zero[:4]
        mixed = family.Builder.value(self, 'flow', 'vselect', mask,
            coefficients[1:] + self.new(4, undefined=True),
            self.new(4, undefined=True) + addends[:4],
            time=6, tag=('joint_hash_constants', ()))
        cv = coefficients + mixed[:4]
        av = mixed[4:] + addends
        self.hash_coeff_x, self.hash_coeff_y = cv[:8], cv[1:]
        self.hash_add_x, self.hash_add_y = av[:8], av[1:]
        self.hash_final_add = self.vectors[0xfd7046c5]

    def run(self):
        super().run()
        if self.head_c5:
            assert self.head_full and self.cache_vector
            a = next(op for op in self.ops if op.engine == 'flow'
                and op.code == 'add_imm' and op.args[2] == 0xb55a4f09)
            a.engine, a.code, a.args, a.kinds, a.time = (
                'load', 'const', [a.args[0], 0xb55a4f09], kinds('load', 'const'), 0)
            b = next(op for op in self.ops if op.engine == 'load'
                and op.code == 'const' and op.args[1] == 0xc761c23c)
            b.engine, b.code, b.args, b.kinds, b.time = (
                'flow', 'add_imm', [b.args[0], (self.zero[0],), 0xc761c23c],
                kinds('flow', 'add_imm'), 1)
            self.stats['head_c5_load_swap'] = 1
        if self.early_input:
            op = next(op for op in self.ops if op.engine != 'store'
                and op.args[0] == self.inptr[1])
            op.engine, op.code, op.args, op.kinds, op.time = (
                'load', 'const', [op.args[0], 2318], kinds('load', 'const'), 0)
            self.stats['early_input_literal'] = 1
        if self.stride16:
            definitions = {op.args[0]: op for op in self.ops if op.engine != 'store'}
            for pointers in (self.treeptr, self.inptr):
                for i in range(2, len(pointers)):
                    op = definitions[pointers[i]]
                    assert op.engine == 'alu'
                    op.code = '+'
                    op.args = [op.args[0], pointers[i - 2], self.scalar[16]]
                    op.time = 1 + i // 2
            self.stats['stride16_pointer_roots'] = 2
        if self.cheap_coeff:
            assert 3 in self.small_stage and self.joint_pack
            definitions = {v: op for op in self.ops if op.engine != 'store'
                           for v in op.args[0]}
            p270 = definitions[self.treeptr[33][0]]
            p270.code = '+'
            p270.args = [p270.args[0], self.treeptr[1], self.scalar[256]]
            p270.time = 2
            s33 = definitions[self.scalar[33][0]]
            s33.code = '>>'
            s33.args = [s33.args[0], self.treeptr[33], self.scalar[3]]
            s33.time = 3
            s9 = next(op.args[0] for op in self.ops if op.engine == 'alu'
                and op.code == '-' and op.args[1:] == [self.scalar[10], self.scalar[1]])
            scale = definitions[self.scalar[16896][0]]
            scale.code = '<<'
            scale.args = [scale.args[0], self.scalar[33], s9]
            scale.time = 4
            self.stats['cheap_coefficient_roots'] = 2
        if self.scalar_shifts:
            scalar_initial_shifts(self, self.scalar_shifts)
        if self.terminal_count:
            terminal_pointer_tables(self, self.terminal_count)
        if self.pointer_pack:
            pack_startup_pointers(self)
        if self.pointer_pack or self.cheap_coeff:
            logical_order(self)
        return self


def build(args):
    original = previous.Builder
    previous.Builder = partial(Builder, joint_pack=getattr(args, 'joint_pack', False),
        terminal_count=getattr(args, 'terminal_count', 0),
        early_input=getattr(args, 'early_input', False),
        scalar_shifts=getattr(args, 'scalar_shifts', 0),
        stride16=getattr(args, 'stride16', False),
        cheap_coeff=getattr(args, 'cheap_coeff', False),
        pointer_pack=getattr(args, 'pointer_pack', False),
        head_c5=getattr(args, 'head_c5', False))
    try:
        ir, builder = previous.build(args)
    finally:
        previous.Builder = original
    if getattr(args, 'header_pack', ''):
        pack_header(builder, args.header_pack.split(','))
        builder.sink = builder.literals = builder.literal_flow = 0
        ir = research.clean(builder)
    return ir, builder


def logical_order(builder):
    """Order logical producers before uses, preserving every memory epoch."""
    ops = builder.ops
    writer = {v: j for j, op in enumerate(ops) if op.engine != 'store'
              for v in op.args[0]}
    deps = [set() for _ in ops]
    previous_memory = None
    for j, op in enumerate(ops):
        for kind, arg in zip(op.kinds, op.args):
            if kind != 'imm' and not kind.endswith('w'):
                deps[j].update(writer[v] for v in arg if v in writer and writer[v] != j)
        if op.engine in ('load', 'store'):
            if previous_memory is not None:
                deps[j].add(previous_memory)
            previous_memory = j
    succ = [[] for _ in ops]
    degree = list(map(len, deps))
    for j, dd in enumerate(deps):
        for d in dd:
            succ[d].append(j)
    ready = [j for j, d in enumerate(degree) if not d]
    heapq.heapify(ready)
    ordered = []
    while ready:
        j = heapq.heappop(ready)
        ordered.append(ops[j])
        for q in succ[j]:
            degree[q] -= 1
            if not degree[q]:
                heapq.heappush(ready, q)
    assert len(ordered) == len(ops), 'logical ordering introduced a cycle'
    builder.ops = ordered


def pack_startup_pointers(builder):
    assert 3 in builder.small_stage
    ops = builder.ops
    find = lambda code, a, b: next(op.args[0] for op in ops if op.engine == 'alu'
        and op.code == code and op.args[1:] == [a, b])
    s8 = find('>>', builder.scalar[16], builder.scalar[1])
    s9 = find('-', builder.scalar[10], builder.scalar[1])
    s32 = find('+', builder.scalar[16], builder.scalar[16])
    operands = (builder.treeptr[1][0], s8[0], builder.treeptr[0][0], s9[0],
        builder.new(1, undefined=True)[0], builder.scalar[3][0], s32[0],
        builder.new(1, undefined=True)[0])
    out = family.Builder.value(builder, 'valu', '+', builder.header, operands,
        time=2, tag=('pointer_pack', ()))
    targets = {30: out[0], 2055: out[1], 262: out[2], 19: out[3],
               2057: out[5], 2342: out[6]}
    # Match exact known standalone definitions; none of the source words is a
    # target, so this substitution cannot depend on its own result.
    logical_order(builder)
    known, _ = family.evaluate(builder)
    packed = {v for op in builder.ops for kind, arg in zip(op.kinds, op.args)
              if kind != 'imm' and len(arg) > 1 and op.tag[0] != 'pointer_pack' for v in arg}
    replace = {}
    kept = []
    for op in builder.ops:
        if (op.engine == 'alu' and len(op.args[0]) == 1
                and op.args[0][0] not in packed
                and known.get(op.args[0][0]) in targets):
            replace[op.args[0][0]] = targets[known[op.args[0][0]]]
        else:
            kept.append(op)
    for op in kept:
        if op.tag[0] == 'pointer_pack':
            continue
        for k, arg in enumerate(op.args):
            if op.kinds[k] != 'imm' and not op.kinds[k].endswith('w'):
                op.args[k] = tuple(replace.get(v, v) for v in arg)
    builder.ops = kept
    builder.stats['packed_pointer_definitions'] = len(replace)


def scalar_initial_shifts(builder, count):
    assert 0 < count <= 32
    targets = set(sorted(range(32), key=lambda g: builder.hashhint[0, g])[:count])
    first = {op.args[0] for op in builder.ops
        if op.tag[0] == 'hash_mad' and op.tag[1][0] == 0 and op.tag[1][1] in targets}
    result = []
    for op in builder.ops:
        if op.engine == 'valu' and op.code == '>>' and op.args[1] in first:
            for lane in range(8):
                result.append(type(op)(engine='alu', code='>>',
                    args=[(op.args[0][lane],), (op.args[1][lane],), builder.scalar[19]],
                    kinds=kinds('alu', '>>'), time=op.time, tag=('scalar_head_shift', (lane,))))
        else:
            result.append(op)
    builder.ops = result
    builder.stats['scalar_head_shift_groups'] = count


def terminal_pointer_tables(builder, count):
    """Replace selected last-pass address MADs by a constant pointer tree.

    Copies of the eight existing three-word records occupy reclaimed tree
    memory only after every original deep-tree read. A record's five extra
    vstore words are padding; ascending copies preserve each live triplet.
    """
    assert 0 < count <= 32
    assert builder.head4097, 'the table reuses the existing uniform 256'
    addresses = (0, 3, 9, 16, 19, 33, 256, 1026)
    assert all(v in builder.vectors for v in addresses if v)
    targets = sorted(range(32), key=lambda g: builder.hashhint[14, g], reverse=True)[:count]
    original = list(builder.ops)
    stores = {op.tag[1][0]: op for op in original if op.tag[0] == 'image'}
    before = len(builder.ops)
    for q, address in enumerate(addresses):
        b0, b1, b2 = q >> 2 & 1, q >> 1 & 1, q & 1
        old_address = family.Y[b1] + family.X[b0] * (b2 - 2)
        block = builder.image_base + 8 * ((old_address - builder.image_base) // 8)
        words = stores[block].args[1]
        offset = old_address - block
        record = words[offset:offset + 3]
        assert len(record) == 3
        # This extra view fixes the complete eight-word architectural read.
        image = record + builder.new(5, undefined=True)
        ptr = builder.scalar[address]
        builder.put('store', 'vstore', [ptr, image], 650,
            ('terminal_image', (q, address)))
    copies = builder.ops[before:]
    del builder.ops[before:]
    writer = {v: op for op in original if op.engine != 'store' for v in op.args[0]}
    moved = []
    lookup = []
    for g in targets:
        loads = [op for op in original if op.tag[0] == 'gather'
                 and op.tag[1][:2] == (14, g)]
        assert len(loads) == 8
        address_op = writer[loads[0].args[1][0]]
        assert address_op.code == 'multiply_add'
        b0 = writer[address_op.args[1][0]].args[1]
        b1 = writer[address_op.args[3][0]].args[1]
        b2 = address_op.args[2]
        bit_op = writer[b2[0]]
        assert bit_op.code == '|'
        bit_op.code = '&'
        bit_op.args[2] = builder.vectors[1]
        vectors = [builder.zero if v == 0 else builder.vectors[v] for v in addresses]
        start = len(builder.ops)
        t = address_op.time
        pairs = [builder.sel(b2, vectors[i + 1], vectors[i], t - 3,
            ('terminal_address', (g, 0, i // 2))) for i in range(0, 8, 2)]
        halves = [builder.sel(b1, pairs[i + 1], pairs[i], t - 2,
            ('terminal_address', (g, 1, i // 2))) for i in range(0, 4, 2)]
        pointer = builder.sel(b0, halves[1], halves[0], t - 1,
            ('terminal_address', (g, 2, 0)))
        lookup.extend(builder.ops[start:])
        del builder.ops[start:]
        for lane, op in enumerate(loads):
            op.args[1] = (pointer[lane],)
        moved.extend(loads)
    moved_ids = {id(op) for op in moved}
    outputs = [op for op in original if op.tag[0] == 'output']
    builder.ops = [op for op in original if id(op) not in moved_ids
                   and op.tag[0] != 'output'] + copies + lookup + moved + outputs
    builder.stats['terminal_pointer_groups'] = count
    builder.stats['terminal_pointer_image_stores'] = len(copies)


def pack_header(builder, specs):
    """Share useful header arithmetic without changing any old packed view."""
    old = list(builder.ops)
    old_known, _ = family.evaluate(builder)
    introduced = []
    for spec in specs:
        code, value = spec.split(':')
        value = int(value, 0) & family.MASK
        assert value in builder.vectors, (spec, 'requires an existing vector')
        out = family.Builder.value(builder, 'valu', code, builder.header,
            builder.vectors[value], time=1, tag=('header_pack', (spec,)))
        introduced.append(out)
    added = builder.ops[len(old):]
    # The construction order must also be a valid constant-evaluation order.
    builder.ops = old + added
    known, _ = family.evaluate(builder)
    packed = {v for op in old for kind, arg in zip(op.kinds, op.args)
              if kind != 'imm' and len(arg) > 1 for v in arg}
    sources = {}
    for out in introduced:
        for word in out:
            if word in known:
                sources.setdefault(known[word], word)
    replaced = {}
    kept = []
    for op in old:
        if (op.engine in ('alu', 'load', 'flow') and op.code != 'vload'
                and op.kinds[0] == 'w' and len(op.args[0]) == 1):
            dest = op.args[0][0]
            if dest not in packed and dest in old_known and old_known[dest] in sources:
                replaced[dest] = sources[old_known[dest]]
                continue
        kept.append(op)
    for op in kept:
        for k, arg in enumerate(op.args):
            if op.kinds[k] != 'imm' and not op.kinds[k].endswith('w'):
                op.args[k] = tuple(replaced.get(v, v) for v in arg)
    required = {v for op in added for arg in op.args[1:] for v in arg}
    insertion = max(j + 1 for j, op in enumerate(kept)
        if op.engine != 'store' and any(v in required for v in op.args[0]))
    builder.ops = kept[:insertion] + added + kept[insertion:]
    builder.stats['header_scalar_definitions_removed'] += len(replaced)
    return builder


def serial_control(ir, seeds=3):
    degree = list(map(len, ir.deps))
    ready = [j for j, d in enumerate(degree) if d == 0]
    heapq.heapify(ready)
    times = [-1] * len(ir.ops)
    step = 0
    while ready:
        j = heapq.heappop(ready)
        times[j] = step
        step += 1
        for q, _ in ir.succ[j]:
            degree[q] -= 1
            if degree[q] == 0:
                heapq.heappush(ready, q)
    assert step == len(ir.ops), 'cyclic experimental graph'
    program, words, _ = allocate(ir, times)
    assert words <= 1536, words
    return {'scratch': words, 'verified': verify(program, seeds)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix', type=Path, required=True)
    ap.add_argument('--layout', choices=('tight', 'shared'), default='shared')
    ap.add_argument('--small-stage', default='3,4,-2')
    ap.add_argument('--extra-stage', default='')
    ap.add_argument('--coefficients', action='store_true')
    ap.add_argument('--hash-pack', action='store_true')
    ap.add_argument('--replicate-pack', action='store_true')
    ap.add_argument('--native-pack', default='')
    ap.add_argument('--head4097', action='store_true')
    ap.add_argument('--head-full', action='store_true')
    ap.add_argument('--cache-vector', action='store_true')
    ap.add_argument('--native-cache', default='')
    ap.add_argument('--sink', type=int, default=0)
    ap.add_argument('--literals', type=int, default=0)
    ap.add_argument('--literal-flow', type=int, default=6)
    ap.add_argument('--header-pack', default='', help='e.g. >>:1,+:16,-:2')
    ap.add_argument('--joint-pack', action='store_true')
    ap.add_argument('--terminal-count', type=int, default=0)
    ap.add_argument('--early-input', action='store_true')
    ap.add_argument('--scalar-shifts', type=int, default=0)
    ap.add_argument('--stride16', action='store_true')
    ap.add_argument('--cheap-coeff', action='store_true')
    ap.add_argument('--pointer-pack', action='store_true')
    ap.add_argument('--head-c5', action='store_true')
    ap.add_argument('--serial', action='store_true')
    args = ap.parse_args()
    if args.replicate_pack or args.joint_pack:
        args.hash_pack = True
    assert set(args.native_pack.split(',')) <= {'', 'coeff', 'addend', 'final'}
    assert not args.native_pack or args.replicate_pack or args.joint_pack
    ir, builder = build(args)
    counts = Counter(op.engine for op in ir.ops)
    print({'C': counts['alu'] + 8 * counts['valu'], 'counts': dict(counts),
        'changes': dict(builder.stats)}, flush=True)
    if args.serial:
        print(serial_control(ir), flush=True)
    with Path(str(args.prefix) + '.ir.pkl').open('wb') as stream:
        pickle.dump(ir, stream)


if __name__ == '__main__':
    main()
