"""Integrate a fully verified restore-absorbing macro into the real 864 IR.

Only rounds 8, 9 and 15 need raw hash outputs. The verified baseline already
places C5 on e in parallel with e>>16 in these rounds. Replace its four middle
ops by a macro for F(b XOR I) XOR PRE_C5, change C1 to C1 XOR I locally,
remove the C5 op and shift the new encoded e. Nothing executes on the host
with the benchmark's runtime values.

For input key I=C1 the old head XOR becomes XOR zero: delete that instruction
and redirect its sole consumer to the old head input. This permits a future
five-op joint C1+C5 macro. --joint-control uses the known six-op expression
to verify this integration, without claiming an instruction saving.

The built-in five-op macro is a correctness control, not a speedup: it moves
the same XOR across the linear tail. This generic emitter also materializes
uniform views for coefficients packed in alternating lanes in the baseline.
A four-op formula would save 768 lane operations before initialization changes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import pickle

import z3

from codex import research_flow_stage863 as flow
from codex import research_frontier863 as frontier
from codex import research_sub864 as research
from codex import stride4_family as family
from codex.search_restore_wirings863 import expression
from codex.search_cross_encoding863 import C1, C5, PRE_C5
from codex.cegis_c1_absorb863 import middle


CONTROL = dict(input_key=0, order=['mad', 'mad', '^', 'mad', '^'],
               wiring=[[0, 1, 2], [0, 2, 3], [1, 2], [3, 4, 5], [4, 5]],
               constants=[[33, 0xE9F8CC1D], [16896, 0xACCF6200], [0, 0],
                          [9, 0xFD7046C5], [PRE_C5, 0]])

JOINT_CONTROL = dict(input_key=C1, order=['^', 'mad', 'mad', '^', 'mad', '^'],
                     wiring=[[0, 1], [1, 2, 3], [1, 3, 4], [2, 3], [4, 5, 6], [5, 6]],
                     constants=[[C1, 0], [33, 0xE9F8CC1D], [16896, 0xACCF6200],
                                [0, 0], [9, 0xFD7046C5], [PRE_C5, 0]])


def verify_macro(macro, seconds=30):
    assert len(macro['order']) == len(macro['wiring']) == len(macro['constants'])
    assert len(macro['order']) in (4, 5, 6)
    for i, (op, wiring, kk) in enumerate(zip(macro['order'], macro['wiring'], macro['constants'])):
        assert op in ('mad', '^', '&', '|')
        assert len(wiring) == (3 if op == 'mad' else 2)
        assert len(kk) == 2 and all(0 <= j < i+3 for j in wiring)
    x = z3.BitVec('macro_input', 32)
    constants = [[z3.BitVecVal(v & 0xFFFFFFFF, 32) for v in kk] for kk in macro['constants']]
    got = expression(macro['order'], macro['wiring'], constants, x)
    want = middle(x ^ macro['input_key'], 0xFFFFFFFF) ^ PRE_C5
    checker = z3.SolverFor('QF_BV')
    checker.set(timeout=int(seconds*1000))
    checker.add(got != want)
    result = checker.check()
    if result != z3.unsat:
        detail = checker.reason_unknown() if result == z3.unknown else hex(checker.model().eval(x).as_long())
        raise ValueError(f'macro is not proved equivalent: {result}, {detail}')


def build(macro, rounds=(8, 9, 15)):
    verify_macro(macro)
    builder = deepcopy(flow.original())
    before = Counter(o.engine for o in builder.ops)
    ops = list(builder.ops)
    known, _ = family.evaluate(builder)
    # Earlier packing rewrites can leave builder.vectors pointing to retired
    # SSA names. Reuse the actual live uniform views, not the stale dictionary.
    live_uniforms = {}
    for o in ops:
        for kind, arg in zip(o.kinds, o.args):
            if kind != 'imm' and len(arg) == 8 and arg[0] in known:
                value = known[arg[0]]
                if all(known.get(v) == value for v in arg):
                    live_uniforms.setdefault(value, arg)
    live_scalars = {}
    for v, value in known.items():
        live_scalars.setdefault(value, (v,))
    writer = {v: o for o in ops if o.engine != 'store' for v in o.args[0]}
    users = {}
    for o in ops:
        for kind, arg in zip(o.kinds, o.args):
            if kind != 'imm' and not kind.endswith('w'):
                for v in arg:
                    users.setdefault(v, set()).add(id(o))
    const = builder.vectors[C5]
    old_c1 = builder.vectors[C1]
    shifts = builder.vectors[16]
    matched = []
    for restore in ops:
        if restore.engine != 'valu' or restore.code != '^' or restore.args[2] != const:
            continue
        e = writer.get(restore.args[1][0])
        if e is None or e.engine != 'valu' or e.code != 'multiply_add':
            continue
        z = writer[e.args[1][0]]
        if z.code != '^':
            continue
        a, b = [writer[v[0]] for v in z.args[1:]]
        if a.code != 'multiply_add' or b.code != 'multiply_add' or a.args[1] != b.args[1]:
            continue
        mixed = writer[a.args[1][0]]
        candidates = [writer.get(arg[0]) for arg in mixed.args[1:]]
        head = next((o for o in candidates if o is not None and o.code == '^' and o.args[2] == old_c1), None)
        assert head is not None
        coord = writer[head.args[1][0]].tag
        assert coord[0] == 'hash_mad', coord
        r, g = coord[1]
        if r not in rounds:
            continue
        shift = [o for o in ops if o.code == '>>' and o.args[1:] == [e.args[0], shifts]]
        assert len(shift) == 1
        assert users[e.args[0][0]] == {id(restore), id(shift[0])}
        assert users[a.args[0][0]] == users[b.args[0][0]] == {id(z)}
        assert users[z.args[0][0]] == {id(e)}
        matched.append((r, g, restore, e, z, a, b, mixed, head))
    assert len(matched) == 32*len(rounds), len(matched)

    # Fresh literals use real LOAD instructions. Put independent literals before
    # the existing memory epochs so logical_order cannot introduce a cycle.
    constants = []
    macros = []

    def uniform(value):
        value &= 0xFFFFFFFF
        if value == 0:
            return builder.zero
        if value in live_uniforms:
            return live_uniforms[value]
        if value in live_scalars:
            source = live_scalars[value]
        else:
            source = builder.new(1)
            builder.put('load', 'const', [source, value], 100, ('restore_macro_constant', (value,)))
            constants.append(builder.ops.pop())
            builder.scalar[value] = source
            live_scalars[value] = source
        out = builder.new(8)
        builder.put('valu', 'vbroadcast', [out, source], 101, ('restore_macro_uniform', (value,)))
        constants.append(builder.ops.pop())
        builder.vectors[value] = out
        live_uniforms[value] = out
        known.update({v: value for v in out})
        return out

    new_c1 = uniform(C1 ^ macro['input_key'])
    remove = set()
    replace = {}
    folded_heads = 0
    for r, g, restore, e, z, a, b, mixed, head in matched:
        remove.update(id(o) for o in (restore, e, z, a, b))
        if ((C1 ^ macro['input_key']) & 0xFFFFFFFF) == 0:
            # A joint C1+C5 macro consumes a XOR (a>>19), so this old C1
            # instruction becomes x XOR 0 and must really disappear.
            assert all(users[v] == {id(mixed)} for v in head.args[0])
            remove.add(id(head))
            replace.update(zip(head.args[0], head.args[1]))
            folded_heads += 1
        else:
            head.args[2] = new_c1
        values = [mixed.args[0]]
        for i, (op, wiring, kk) in enumerate(zip(macro['order'], macro['wiring'], macro['constants'])):
            pool = values + [None, None]
            for k in set(wiring):
                if k >= len(values):
                    pool[k] = uniform(kk[k-len(values)])
            args = [pool[k] for k in wiring]
            out = e.args[0] if i == len(macro['order'])-1 else builder.new(8)
            builder.put('valu', 'multiply_add' if op == 'mad' else op, [out, *args],
                        a.time+i, ('restore_macro', (r, g, i)))
            macros.append(builder.ops.pop())
            values.append(out)
        replace.update(zip(restore.args[0], e.args[0]))
    kept = [o for o in ops if id(o) not in remove]
    for o in kept:
        for k, (kind, arg) in enumerate(zip(o.kinds, o.args)):
            if kind != 'imm' and not kind.endswith('w'):
                o.args[k] = tuple(replace.get(v, v) for v in arg)
    builder.ops = constants + kept + macros
    builder.sink = builder.literals = builder.literal_flow = 0
    frontier.logical_order(builder)
    # Removing every C5 consumer makes its constant vload dead. The staging
    # protocol owns eight stores per reload; remove this now-unobserved pack
    # before asking order_packs to reorder the remaining complete packs.
    pack_order = builder.pack_order
    builder.pack_order = False
    research.clean(builder)
    live_packs = {o.tag[1][0] for o in builder.ops if o.tag[0] == 'constant_reload'}
    orphaned = [o for o in builder.ops if o.tag[0] == 'constant_stage' and o.tag[1][0] not in live_packs]
    orphan_counts = Counter(o.tag[1][0] for o in orphaned)
    assert all(n == 8 for n in orphan_counts.values()), orphan_counts
    orphan_ids = {id(o) for o in orphaned}
    builder.ops = [o for o in builder.ops if id(o) not in orphan_ids]
    builder.pack_order = pack_order
    ir = research.clean(builder)
    after = Counter(o.engine for o in ir.ops)
    return ir, dict(rounds=list(rounds), replaced_hashes=len(matched),
                    macro_instructions=len(macro['order']), before=dict(before), after=dict(after),
                    removed_head_xors=folded_heads,
                    removed_constant_packs=len(orphan_counts),
                    new_uniforms=[o.tag[1][0] for o in constants if o.tag[0] == 'restore_macro_uniform'],
                    lane_work_before=before['alu']+8*before['valu'],
                    lane_work_after=after['alu']+8*after['valu'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--macro', type=Path)
    ap.add_argument('--joint-control', action='store_true')
    ap.add_argument('--rounds', default='8,9,15')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--ir', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert not (args.macro and args.joint_control)
    macro = json.loads(args.macro.read_text()) if args.macro else JOINT_CONTROL if args.joint_control else CONTROL
    ir, report = build(macro, tuple(map(int, args.rounds.split(','))))
    print(json.dumps(report), flush=True)
    control = flow.serial_control(ir, args.seeds)
    report.update(macro=macro, full32_macro_verified=True, full32_random_seeds=args.seeds,
                  scratch=control['scratch'], serial_cycles=control['verified'],
                  serial_cycles_are_not_a_score=True, new_executable=False)
    if args.ir:
        args.ir.write_bytes(pickle.dumps(ir))
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('scratch', 'serial_cycles', 'full32_random_seeds')}), flush=True)


if __name__ == '__main__':
    main()
