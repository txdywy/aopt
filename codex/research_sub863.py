"""Explore shared hash constants and startup changes below the 865 kernel.

This compiler writes experimental IR only. All candidates need a new calendar
and frozen-machine validation before they can replace the standalone source.
"""
from __future__ import annotations

import argparse
from collections import Counter
import heapq
from pathlib import Path
import pickle

from codex import stride4_family as family
from codex import research_sub864 as research
from codex.research_bootstrap864 import Builder as BootstrapBuilder
from codex.research_packet_870 import verify
from codex.word_ssa import allocate


class Builder(BootstrapBuilder):
    def __init__(self, *, hash_pack=False, head4097=False, head_full=False,
                 cache_vector=False, native_cache=(), replicate_pack=False,
                 native_pack=(), **kwargs):
        self.hash_pack = hash_pack
        self.head4097 = head4097
        self.head_full = head_full
        self.cache_vector = cache_vector
        self.native_cache = set(native_cache)
        self.cache_sources = {}
        self.encoded_cache = None
        self.stage33_source = self.stage33_out = None
        self.replicate_pack = replicate_pack
        self.native_pack = set(native_pack)
        super().__init__(**kwargs)

    def literal(self, value, time=1):
        value &= family.MASK
        if self.head4097:
            if value == 0xb55a4f09 and value not in self.scalar:
                self.scalar[value] = family.Builder.value(
                    self, 'flow', 'add_imm', (self.zero[0],), value,
                    time=1 if self.head_full else 0)
            if value == 3925396509:
                time = 0
        return super().literal(value, time)

    def value(self, engine, code, *inputs, time=0, tag=('setup', ())):
        if ('coeff' in self.native_pack and engine == 'valu' and code == 'multiply_add'
                and inputs == (self.vectors.get(16), self.vectors.get(2), self.vectors.get(1))):
            return family.Builder.value(self, engine, code, *inputs, time=time, tag=tag)
        if engine == 'valu' and code == 'vbroadcast':
            if 'final' in self.native_pack and inputs[0] == self.scalar.get(0xfd7046c5):
                return family.Builder.value(self, engine, code, *inputs, time=2, tag=tag)
            node = self.cache_sources.get(inputs[0])
            if node in self.native_cache:
                self.cache_count += 1
                return family.Builder.value(self, engine, code, *inputs,
                    time=4, tag=tag)
            if inputs[0] == self.scalar.get(16896) and 16896 in self.stage_values:
                out = self.new(8)
                self.pending.append((inputs[0], out, time))
                self.stats['stage'] += 1
                return out
        if (engine == 'valu' and code == 'multiply_add'
                and inputs == (self.vectors.get(16), self.vectors.get(2), self.vectors.get(1))
                and 33 in self.stage_values):
            s32 = self.s('+', self.scalar[16], self.scalar[16], 1)
            s33 = self.s('+', s32, self.scalar[1], 2)
            out = self.new(8)
            self.pending.append((s33, out, 3))
            self.stats['stage'] += 1
            self.stage33_source, self.stage33_out = s33, out
            return out
        if self.head_full and engine == 'flow' and code == 'add_imm':
            if inputs[1] == 0x7ed55d16:
                time = 0
            elif inputs[1] == 0xc761c23c:
                return family.Builder.value(self, 'load', 'const', inputs[1], time=0)
        if (self.head4097 and engine == 'valu' and code == 'vbroadcast'
                and inputs[0] == self.scalar.get(4097)):
            v256 = self.uniform(256, 1)
            return family.Builder.value(self, 'valu', 'multiply_add',
                self.vectors[16], v256, self.vectors[1], time=2, tag=tag)
        return super().value(engine, code, *inputs, time=time, tag=tag)

    def s(self, code, a, b, time=1):
        if self.stage33_out is not None:
            if a == self.stage33_out[:1]:
                a = self.stage33_source
            if b == self.stage33_out[:1]:
                b = self.stage33_source
        if (code == '^' and hasattr(self, 'rawnode')
                and b == self.scalar.get(0xb55a4f09)):
            node = next((n for n in range(7) if a == self.rawnode[n]), None)
            if node is not None:
                if self.cache_vector:
                    if self.encoded_cache is None:
                        raw = next(op.args[0] for op in self.ops
                            if op.engine == 'load' and op.code == 'vload'
                            and self.rawnode[0][0] in op.args[0])
                        self.encoded_cache = family.Builder.value(self, 'valu', '^',
                            raw, self.vectors[0xb55a4f09], time=3, tag=('cache_xor', ()))
                    source = (self.encoded_cache[node + 1],)
                else:
                    source = super().s(code, a, b, time)
                self.cache_sources[source] = node
                return source
        return super().s(code, a, b, time)

    def copies(self, source, count, time):
        block = list(source)
        while len(block) < count:
            block.extend(family.Builder.value(self, 'alu', '|', source,
                (self.zero[0],), time=time))
        return tuple(block)

    def setup(self):
        super().setup()
        if not self.hash_pack:
            return
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
        mask = self.vectors[1][4:] + self.zero[:4]

        # Prefix ABABA + one selector result BABAEEEE supplies both shifted
        # eight-word addend windows and a full uniform E vector. All eight
        # architectural selector destinations remain in the allocation model.
        a, b, e = (self.scalar[v] for v in (3925396509, 2899272192, 0xfd7046c5))
        av = (tuple(self.vectors[3925396509][i] for i in (0, 2, 4))
              if 'addend' in self.native_pack else self.copies(a, 3, 3))
        bv = self.copies(b, 2, 4)
        prefix = (av[0], bv[0], av[1], bv[1], av[2])
        if self.replicate_pack:
            ca = (tuple(self.vectors[33][i] for i in (0, 2, 4))
                  if 'coeff' in self.native_pack else self.copies(s33, 3, 3))
            cb = self.copies(self.scalar[16896], 2, 4)
            coef_prefix = (ca[0], cb[0], ca[1], cb[1], ca[2])
            ev = (self.new(4, undefined=True) if 'final' in self.native_pack
                  else self.copies(e, 4, 5))
            coef_out = family.Builder.value(self, 'flow', 'vselect', mask,
                ev + self.new(4, undefined=True),
                self.new(4, undefined=True) + coef_prefix[:4], time=6)
            coefficients = coef_out[4:] + coef_prefix
            self.hash_coeff_x, self.hash_coeff_y = coefficients[:8], coefficients[1:]
            self.hash_final_add = (self.vectors[0xfd7046c5] if 'final' in self.native_pack
                                   else ev + coef_out[:4])
            add_out = family.Builder.value(self, 'flow', 'vselect', mask,
                prefix[1:] + self.new(4, undefined=True),
                self.new(8, undefined=True), time=6)
            addends = prefix + add_out[:4]
            self.hash_add_x, self.hash_add_y = addends[:8], addends[1:]
            return
        ev = self.copies(e, 4, 5)
        mixed = family.Builder.value(self, 'flow', 'vselect', mask,
            prefix[1:] + self.new(4, undefined=True),
            self.new(4, undefined=True) + ev, time=6)
        self.hash_add_x = prefix + mixed[:3]
        self.hash_add_y = prefix[1:] + mixed[:4]
        self.hash_final_add = mixed[4:] + ev
        # Two overlapping windows of nine alternating scalar words match
        # the addend swap pattern without a second selector.
        ca = self.copies(s33, 5, 3)
        cb = self.copies(self.scalar[16896], 4, 4)
        coefficients = tuple(v for pair in zip(ca, cb) for v in pair) + ca[4:]
        self.hash_coeff_x = coefficients[:8]
        self.hash_coeff_y = coefficients[1:]

    def hash(self, state, r, g):
        if not self.hash_pack:
            return super().hash(state, r, g)
        t = self.hashhint[r, g]
        c = lambda v: self.vectors[v & family.MASK]
        a = self.mad(state, c(4097), c(0x7ed55d16), t, ('hash_mad', (r, g)))
        shift = self.v('>>', a, c(19), t + 1)
        mix = self.v('^', a, c(0xc761c23c), t + 1)
        b = self.v('^', shift, mix, t + 2)
        x = self.mad(b, self.hash_coeff_x, self.hash_add_x, t + 3)
        y = self.mad(b, self.hash_coeff_y, self.hash_add_y, t + 3)
        z = self.v('^', x, y, t + 4)
        e = self.mad(z, c(9), self.hash_final_add, t + 5)
        shift = self.v('>>', e, c(16), t + 6)
        if self.tail and r == 15:
            e = self.v('^', e, c(0xb55a4f09), t + 6)
        return self.v('^', e, shift, t + 7)


def build(args):
    previous = family.X, family.Y, family.V
    if args.layout == 'shared':
        family.X, family.Y, family.V = (-60, -40), (2310, 2430), (4, 16)
        flow = (-60 & family.MASK, -40 & family.MASK, 2899272192, 0xfd7046c5)
        staged = (0xb55a4f09, 2430, 1026, 1027, 1028, 1029)
    else:
        family.X, family.Y, family.V = (-72, -52), (2266, 2378), (-4, 4)
        flow = (-72 & family.MASK, -52 & family.MASK, 2899272192, 0xfd7046c5)
        staged = (0xb55a4f09, 2266, 2378, 1026, 1027, 1028, 1029)
    staged = (*staged, *(int(v, 0) & family.MASK
        for v in args.extra_stage.split(',') if v))
    if args.cache_vector:
        staged = tuple(v for v in staged if v != 0xb55a4f09)
    try:
        builder = Builder(small_stage=tuple(int(v) for v in args.small_stage.split(',') if v),
            relocate=True, layout=args.layout, banks=1, cache_stage=True, early_cache=True,
            pair_flow=False, pack_order=True, sink=args.sink, literals=args.literals,
            literal_flow=args.literal_flow, parallel_bank=True, head_fill=True,
            coefficients=args.coefficients, thirds=True, flow=flow, staged=staged,
            hash_pack=args.hash_pack, head4097=args.head4097,
            head_full=args.head_full, cache_vector=args.cache_vector,
            native_cache=tuple(int(v) for v in args.native_cache.split(',') if v),
            replicate_pack=args.replicate_pack,
            native_pack=tuple(v for v in args.native_pack.split(',') if v)).run()
        ir = research.clean(builder)
    finally:
        family.X, family.Y, family.V = previous
    return ir, builder


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
    ap.add_argument('--serial', action='store_true')
    args = ap.parse_args()
    if args.replicate_pack:
        args.hash_pack = True
    assert set(args.native_pack.split(',')) <= {'', 'coeff', 'addend', 'final'}
    assert not args.native_pack or args.replicate_pack
    ir, builder = build(args)
    counts = Counter(op.engine for op in ir.ops)
    print({'C': counts['alu'] + 8 * counts['valu'], 'counts': dict(counts),
        'changes': dict(builder.stats)}, flush=True)
    if args.serial:
        degrees = list(map(len, ir.deps))
        ready = [j for j, d in enumerate(degrees) if d == 0]
        heapq.heapify(ready)
        times = [-1] * len(ir.ops)
        step = 0
        while ready:
            j = heapq.heappop(ready)
            times[j] = step
            step += 1
            for q, _ in ir.succ[j]:
                degrees[q] -= 1
                if degrees[q] == 0:
                    heapq.heappush(ready, q)
        assert step == len(ir.ops), 'cyclic experimental IR'
        program, words, _ = allocate(ir, times)
        assert words <= 1536, words
        print({'scratch': words, 'serial_verified': verify(program, 3)}, flush=True)
    with Path(str(args.prefix) + '.ir.pkl').open('wb') as stream:
        pickle.dump(ir, stream)


if __name__ == '__main__':
    main()
