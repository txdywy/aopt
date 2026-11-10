"""Pass-2 resident-node selection through conditional-address stores.

Research IR only. For chosen groups, rounds 13-15 take the d2/d3/d4 nodes
from scratch-resident encoded scalars: shared conjunction address vectors
overwrite a memory bank, and one vload returns each selected field. The r14
record gather and its address MAD disappear. Never edits the submission.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

from codex import research_frontier863 as frontier
from codex import research_stage864 as stage

X = (-60, -40)
Y = (2310, 2430)


def convert(builder, groups):
    ops = builder.ops
    writer = {v: op for op in ops if op.engine != 'store' for v in op.args[0]}
    img = {}
    for op in ops:
        if op.tag[0] == 'image':
            for lane, v in enumerate(op.args[1]):
                img[op.tag[1][0] + lane] = v
    c5 = builder.scalar[0xb55a4f09]
    enc = {}
    for n in (3, 4, 5, 6):
        enc[n] = next(op.args[0][0] for op in ops if op.engine == 'alu'
                      and op.code == '^' and op.args[1] == builder.rawnode[n]
                      and op.args[2] == c5)

    def record(n3):
        q = 14 - n3
        b0, b1, b2 = q >> 2 & 1, q >> 1 & 1, q & 1
        a3 = Y[b1] + X[b0] * (b2 - 2)
        return img[a3], img[a3 + 1], img[a3 + 2]      # R4, L4, n3

    # candidate fields per raw (c1, c2): n2, C0, C1, G00, G01, G10, G11
    fields = {}
    for c1 in (0, 1):
        for c2 in (0, 1):
            m = 3 + 2 * (1 - c1) + (1 - c2)
            right, left = record(2 * m + 2), record(2 * m + 1)
            fields[c1, c2] = (enc[m], right[2], left[2],
                              right[0], right[1], left[0], left[1])
    hm = {op.tag[1]: op for op in ops if op.tag[0] == 'hash_mad'}
    ones = builder.vectors[1]
    zero = builder.zero
    copy = builder.new(1)
    setup = [SimpleNamespace(engine='alu', code='|', args=[copy, builder.bankptr[0], (zero[0],)],
                             kinds=('w', 'r', 'r'), time=2, tag=('scatter_bank', ()))]
    zvec = copy + tuple(p[0] for p in builder.bankptr[1:])
    bank = [(v,) for v in zvec]
    dead = set()
    inserts = {}
    for g in groups:
        t13, t14, t15 = (builder.hashhint[r, g] for r in (13, 14, 15))
        x13 = writer[hm[13, g].args[1][0]]
        outer = writer[x13.args[2][0]]
        b11 = outer.args[1]
        b12 = writer[outer.args[2][0]].args[1]
        gathers = [op for op in ops if op.tag[0] == 'gather' and op.tag[1][:2] == (14, g)]
        p3 = writer[gathers[0].args[1][0]]
        bit13 = writer[p3.args[2][0]]
        assert bit13.code == '|'
        bit13.code = '&'
        bit13.args[2] = ones
        b13 = bit13.args[0]
        child = sorted((op for op in ops if op.tag[0] == 'child_select'
                        and op.tag[1][:2] == (15, g)), key=lambda op: op.tag[1][2])
        b14 = writer[child[0].args[1][0]].args[0]
        new = []

        def put(engine, code, args, time, tag):
            builder.put(engine, code, args, time, tag)
            new.append(builder.ops.pop())

        def sel(cond, yes, no, time, name):
            out = builder.new(8)
            put('flow', 'vselect', [out, cond, yes, no], time, (name, (g,)))
            return out

        v1 = sel(b11, zvec, zero, t13 - 30, 'scatter_conj')
        v2 = sel(b12, zvec, zero, t13 - 30, 'scatter_conj')
        v3 = sel(b11, v2, zero, t13 - 29, 'scatter_conj')
        vectors = []
        for f in range(7):
            t = (t13 if f == 0 else t14 if f < 3 else t15) - 25
            for lane in range(8):
                for rank, (key, ptr) in enumerate((((0, 0), bank[lane]), ((1, 0), (v1[lane],)),
                                                   ((0, 1), (v2[lane],)), ((1, 1), (v3[lane],)))):
                    put('store', 'store', [ptr, (fields[key][f],)], t + lane,
                        ('scatter_store', (g, f, lane, rank)))
            out = builder.new(8)
            put('load', 'vload', [out, bank[0]], t + 9, ('scatter_reload', (g, f)))
            vectors.append(out)
        n2, c0, c1v, g00, g01, g10, g11 = vectors
        d3 = sel(b13, c1v, c0, t14 - 3, 'scatter_select')
        p = sel(b13, g10, g00, t15 - 5, 'scatter_select')
        q = sel(b13, g11, g01, t15 - 5, 'scatter_select')
        d4 = sel(b14, q, p, t15 - 3, 'scatter_select')
        x13.args[2] = n2
        for op in ops:
            if op.tag[0] == 'node_xor' and op.tag[1][:2] == (14, g):
                op.args[2] = (d3[op.tag[1][2]],)
        rows = [op for op in ops if op.tag == ('node_xor', (15, g))]
        selected = {child[part].args[0][2 * i]: part + 2 * i for part in range(2) for i in range(4)}
        for op in rows:
            op.args[2] = (d4[selected[op.args[2][0]]],)
        dead.update(id(op) for op in gathers + child)
        inserts[id(x13)] = new
    result = []
    first = True
    for op in ops:
        if id(op) in dead:
            continue
        if id(op) in inserts:
            result.extend(inserts[id(op)])
        result.append(op)
    builder.ops = setup + result
    builder.stats['scatter_groups'] = len(groups)


def main():
    groups = [int(g) for g in sys.argv.pop(1).split(',') if g]
    original = frontier.build

    def build(args):
        ir, builder = original(args)
        convert(builder, groups)
        frontier.logical_order(builder)
        return ir, builder
    stage.frontier = SimpleNamespace(**{k: getattr(frontier, k) for k in dir(frontier)})
    stage.frontier.build = build
    stage.main()


if __name__ == '__main__':
    main()
