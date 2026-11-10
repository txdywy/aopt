"""Pass-2 resident-node selection through conditional-address stores.

Research IR only. For chosen groups, rounds 13-15 take the d2/d3/d4 nodes
from scratch-resident encoded words. Shared conjunction address vectors
conditionally overwrite memory, and vloads return the selected fields. The
r14 record gather and its address MAD disappear. Two variants:

  chunk  one 8-word vstore per candidate copies a whole d2 subtree chunk
         (stride-7 staging), then two small bank selections   (80 STORE)
  wide   scalar stores with a three-bit conjunction            (224 STORE)

Never edits the submission.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

from codex import research_frontier863 as frontier
from codex import research_stage864 as stage
from codex import stride4_family as family

X = (-60, -40)
Y = (2310, 2430)


def op(engine, code, args, kinds, time, tag):
    return SimpleNamespace(engine=engine, code=code, args=args, kinds=kinds, time=time, tag=tag)


def convert(builder, chunk_groups, wide_groups):
    ops = builder.ops
    writer = {v: o for o in ops if o.engine != 'store' for v in o.args[0]}
    img, blocks = {}, {}
    for o in ops:
        if o.tag[0] == 'image':
            blocks[o.tag[1][0]] = o
            for lane, v in enumerate(o.args[1]):
                img[o.tag[1][0] + lane] = v
    c5 = builder.scalar[0xb55a4f09]
    assert any(o.args[0] == c5 for o in ops if o.engine != 'store')
    enc = {n: next(o.args[0][0] for o in ops if o.engine == 'alu' and o.code == '^'
                   and o.args[1] == builder.rawnode[n] and o.args[2] == c5)
           for n in (3, 4, 5, 6)}

    def address(n3):
        q = 14 - n3
        return Y[q >> 1 & 1] + X[q >> 2 & 1] * ((q & 1) - 2)

    def record(n3):
        a3 = address(n3)
        return img[a3], img[a3 + 1], img[a3 + 2]      # R4, L4, n3

    hm = {o.tag[1]: o for o in ops if o.tag[0] == 'hash_mad'}
    known, _ = family.evaluate(builder)
    live = {}
    for o in ops:
        if o.engine != 'store':
            for v in o.args[0]:
                if v in known:
                    live.setdefault(known[v], (v,))
    scalar = lambda value: live[value & 0xffffffff]
    print('live small', sorted(k for k in live if k < 64), flush=True)
    ones, zero = builder.vectors[1], builder.zero
    setup = []
    copy = builder.new(1)
    setup.append(op('alu', '|', [copy, builder.bankptr[0], (zero[0],)], ('w', 'r', 'r'), 2,
                    ('scatter_bank', ())))
    zvec = copy + tuple(p[0] for p in builder.bankptr[1:])
    bank = [(v,) for v in zvec]

    chunks = z7 = readback = None
    if chunk_groups:
        chunks = {}
        for c1 in (0, 1):
            for c2 in (0, 1):
                m = 3 + 2 * (1 - c1) + (1 - c2)
                left, right = 2 * m + 1, 2 * m + 2
                al, ar = address(left), address(right)
                tail = blocks[2310 + 8 * ((ar - 2310) // 8)]
                assert (ar - 2310) % 8 == 0
                if (al - 2310) % 8 == 4:
                    head = blocks[al - 4]
                    words = list(head.args[1])
                    assert words[7] in builder.undef
                    words[7] = enc[m]
                    head.args[1] = tuple(words)
                    first = tuple(words[4:])
                else:
                    first = builder.new(3)
                    for k, node in enumerate((2 * left + 2, 2 * left + 1, left)):
                        setup.append(op('alu', '^', [(first[k],), builder.rawnode[node], c5],
                                        ('w', 'r', 'r'), 40, ('scatter_copy', (m, k))))
                    first += (enc[m],)
                chunks[c1, c2] = first + tuple(tail.args[1][:4])
        if 8 not in live:
            live[8] = builder.new(1)
            setup.append(op('alu', '>>', [live[8], scalar(16), scalar(1)], ('w', 'r', 'r'), 2,
                            ('scatter_pointer', (-1,))))
        z7 = [builder.new(1)]
        setup.append(op('alu', '+', [z7[0], scalar(2054), scalar(16)],
                        ('w', 'r', 'r'), 3, ('scatter_pointer', (0,))))
        readback = [z7[0]]
        for i in range(1, 8):
            z7.append(builder.new(1))
            setup.append(op('alu', '+', [z7[i], z7[i - 1], scalar(7)],
                            ('w', 'r', 'r'), 3 + i, ('scatter_pointer', (i,))))
        for i in range(1, 7):
            readback.append(builder.new(1))
            setup.append(op('alu', '+', [readback[i], readback[i - 1], scalar(8)],
                            ('w', 'r', 'r'), 3 + i, ('scatter_readback', (i,))))
        z7vec = tuple(v[0] for v in z7)
        # Unselected chunk copies land on the bank, which is dead between packs.
        sink = zvec

    dead, inserts, split = set(), {}, {}
    for g in list(chunk_groups) + list(wide_groups):
        t13, t14, t15 = (builder.hashhint[r, g] for r in (13, 14, 15))
        x13 = writer[hm[13, g].args[1][0]]
        outer = writer[x13.args[2][0]]
        b11 = outer.args[1]
        b12 = writer[outer.args[2][0]].args[1]
        gathers = [o for o in ops if o.tag[0] == 'gather' and o.tag[1][:2] == (14, g)]
        p3 = writer[gathers[0].args[1][0]]
        bit13 = writer[p3.args[2][0]]
        assert bit13.code == '|'
        bit13.code = '&'
        bit13.args[2] = ones
        b13 = bit13.args[0]
        child = sorted((o for o in ops if o.tag[0] == 'child_select'
                        and o.tag[1][:2] == (15, g)), key=lambda o: o.tag[1][2])
        b14 = writer[child[0].args[1][0]].args[0]
        new = []
        counter = [0]

        def sel(cond, yes, no, time, name):
            out = builder.new(8)
            new.append(op('flow', 'vselect', [out, cond, yes, no], ('vw', 'vr', 'vr', 'vr'),
                          time, (name, (g,))))
            return out

        def store(pack, ptr, source, time, vector=False):
            new.append(op('store', 'vstore' if vector else 'store', [ptr, source],
                          ('r', 'vr' if vector else 'r'), time,
                          ('scatter_store', (g, pack, counter[0]))))
            counter[0] += 1

        def reload(pack, ptr, time, k=0):
            out = builder.new(8)
            new.append(op('load', 'vload', [out, ptr], ('vw', 'r'), time,
                          ('scatter_reload', (g, pack, k))))
            return out

        def select(pack, time, candidates):
            """candidates: [(pointer vector or None for the default, source per lane)]"""
            for lane in range(8):
                for pointers, source in candidates:
                    ptr = bank[lane] if pointers is None else (pointers[lane],)
                    store(pack, ptr, (source[lane] if isinstance(source, tuple) else source,), time)
            return reload(pack, bank[0], time + 9)

        if g in chunk_groups:
            v1 = sel(b11, z7vec, sink, t13 - 30, 'scatter_conj')
            v2 = sel(b12, z7vec, sink, t13 - 30, 'scatter_conj')
            v3 = sel(b11, v2, sink, t13 - 29, 'scatter_conj')
            for lane in range(8):
                for pointers, key in ((z7vec, (0, 0)), (v1, (1, 0)), (v2, (0, 1)), (v3, (1, 1))):
                    store(0, (pointers[lane],), chunks[key], t13 - 25, vector=True)
            packets = [reload(0, readback[k], t13 - 8, k) for k in range(7)]
            word = lambda j: tuple(packets[(7 * e + j) // 8][(7 * e + j) % 8] for e in range(8))
            n2 = word(3)
            ab = sel(b13, zvec, zero, t14 - 12, 'scatter_conj')
            d3 = select(1, t14 - 10, [(None, word(6)), (ab, word(2))])
            ag = sel(b14, zvec, zero, t15 - 14, 'scatter_conj')
            abg = sel(b13, ag, zero, t15 - 13, 'scatter_conj')
            d4 = select(2, t15 - 12, [(None, word(4)), (ag, word(5)), (ab, word(0)), (abg, word(1))])
        else:
            v = {0: None}
            v[1] = sel(b11, zvec, zero, t13 - 30, 'scatter_conj')
            v[2] = sel(b12, zvec, zero, t13 - 30, 'scatter_conj')
            v[3] = sel(b11, v[2], zero, t13 - 29, 'scatter_conj')
            v[4] = sel(b13, zvec, zero, t14 - 30, 'scatter_conj')
            for k in (1, 2, 3):
                v[4 + k] = sel(b13, v[k], zero, t14 - 29, 'scatter_conj')
            order = sorted(range(8), key=lambda k: (bin(k).count('1'), k))
            node2 = lambda k: 3 + 2 * (1 - (k & 1)) + (1 - (k >> 1 & 1))
            node3 = lambda k: 2 * node2(k) + (1 if k & 4 else 2)
            n2 = select(0, t13 - 25, [(v[k], enc[node2(k)]) for k in order if k < 4])
            d3 = select(1, t14 - 25, [(v[k], record(node3(k))[2]) for k in order])
            g0 = select(2, t15 - 25, [(v[k], record(node3(k))[0]) for k in order])
            g1 = select(3, t15 - 25, [(v[k], record(node3(k))[1]) for k in order])
            d4 = sel(b14, g1, g0, t15 - 3, 'scatter_select')
        if g in chunk_groups:
            # The staged d2 words are not contiguous: use eight scalar XORs.
            split[id(x13)] = [op('alu', '^', [(x13.args[0][lane],), (x13.args[1][lane],),
                                              (n2[lane],)], ('w', 'r', 'r'), x13.time,
                                 ('scatter_xor', (g, lane))) for lane in range(8)]
        else:
            x13.args[2] = n2
        for o in ops:
            if o.tag[0] == 'node_xor' and o.tag[1][:2] == (14, g):
                o.args[2] = (d3[o.tag[1][2]],)
        selected = {child[part].args[0][2 * i]: part + 2 * i for part in range(2) for i in range(4)}
        for o in ops:
            if o.tag == ('node_xor', (15, g)):
                o.args[2] = (d4[selected[o.args[2][0]]],)
        dead.update(id(o) for o in gathers + child)
        inserts[id(x13)] = new
    result = []
    for o in ops:
        if id(o) in dead:
            continue
        result.extend(inserts.get(id(o), ()))
        result.extend(split.get(id(o), (o,)))
    builder.ops = setup + result
    builder.stats['scatter_chunk_groups'] = len(chunk_groups)
    builder.stats['scatter_wide_groups'] = len(wide_groups)


def main():
    parse = lambda text: [int(g) for g in text.split(',') if g]
    chunk_groups, wide_groups = parse(sys.argv.pop(1)), parse(sys.argv.pop(1))
    original = frontier.build

    def build(args):
        ir, builder = original(args)
        convert(builder, chunk_groups, wide_groups)
        frontier.logical_order(builder)
        return ir, builder
    stage.frontier = SimpleNamespace(**{k: getattr(frontier, k) for k in dir(frontier)})
    stage.frontier.build = build
    stage.main()


if __name__ == '__main__':
    main()
