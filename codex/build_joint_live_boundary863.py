"""Integrate a joint boundary macro, removing the original e MAD as well.

Only runtime-root macros are accepted. The five-op control is the exact
original algebra and must preserve actual engine work and frozen output.
Early values are real definitions, not host-computed runtime inputs.
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
from codex.search_joint_live_boundary863 import inputs, wanted, expression, MASK, C0, C4


CONTROL = dict(live='joint_early', runtime_node=True,
               order=['mad', 'shr16', '^', '^', 'mad'],
               wiring=[[0, 9, 10], [9], [9, 10], [11, 8], [12, 13, 14]],
               constants=[[9, C4], [0, 0], [0, 0], [0, 0], [4097, C0]])


def verify(macro, seconds=30):
    if not macro['runtime_node']:
        raise ValueError('Fixed-root coefficients must be generalized before IR integration')
    assert macro['live'] == 'joint_early'
    assert len(macro['order']) == len(macro['wiring']) == len(macro['constants'])
    for i, (op, w, cc) in enumerate(zip(macro['order'], macro['wiring'], macro['constants'])):
        assert op in ('mad', '^', '&', '|', 'shr16')
        assert len(w) == (3 if op == 'mad' else 1 if op == 'shr16' else 2)
        assert len(cc) == 2 and all(0 <= j < 11+i for j in w)
    b = z3.BitVec('joint_ir_primary_b', 32); root = z3.BitVec('joint_ir_raw_root', 30)
    node = z3.ZeroExt(2, root) ^ 0xB55A4F09
    cc = [[z3.BitVecVal(k & MASK, 32) for k in pair] for pair in macro['constants']]
    for swapped in (False, True):
        values = inputs(b, node, True)
        if swapped:
            values[1], values[2] = values[2], values[1]
        s = z3.SolverFor('QF_BV'); s.set(timeout=int(seconds*1000))
        s.add(expression(macro, cc, values) != wanted(values, node))
        status = s.check()
        if status != z3.unsat:
            raise ValueError(f'Runtime-root macro equivalence not proved for swapped={swapped}: {status}')


def build(macro, base=None):
    verify(macro)
    b = deepcopy(base if base is not None else flow.original())
    ops = list(b.ops); before = Counter(o.engine for o in ops)
    known, _ = family.evaluate(b)
    writer = {v:o for o in ops if o.engine != 'store' for v in o.args[0]}
    users = {}
    for o in ops:
        for kind, arg in zip(o.kinds, o.args):
            if kind != 'imm' and not kind.endswith('w'):
                for v in arg:
                    users.setdefault(v, set()).add(id(o))
    def uniform_is(arg, value):
        return all(known.get(v) == value for v in arg)
    matches = []
    for nxt in ops:
        if nxt.tag[0] != 'hash_mad' or nxt.tag[1][0] != 11:
            continue
        assert nxt.code == 'multiply_add' and uniform_is(nxt.args[2], 4097) and uniform_is(nxt.args[3], C0)
        mix = writer[nxt.args[1][0]]; assert mix.code == '^'
        found = []
        for i, arg in enumerate(mix.args[1:]):
            tail = writer.get(arg[0])
            if tail is None or tail.code != '^':
                continue
            for earg, sarg in (tail.args[1:], tail.args[1:][::-1]):
                e, shift = writer.get(earg[0]), writer.get(sarg[0])
                if e is None or shift is None or e.code != 'multiply_add' or shift.code != '>>':
                    continue
                if shift.args[1] != e.args[0] or not uniform_is(shift.args[2], 16):
                    continue
                if uniform_is(e.args[2], 9) and uniform_is(e.args[3], C4):
                    found.append((e, shift, tail, mix.args[2-i]))
        assert len(found) == 1
        e, shift, tail, node = found[0]
        for out, allowed in ((e.args[0], {id(shift), id(tail)}),
                             (shift.args[0], {id(tail)}), (tail.args[0], {id(mix)}),
                             (mix.args[0], {id(nxt)})):
            assert all(users[v] == allowed for v in out)
        z = writer[e.args[1][0]]; assert z.code == '^'
        arm_a, arm_b = [writer[arg[0]] for arg in z.args[1:]]
        assert arm_a.code == arm_b.code == 'multiply_add' and arm_a.args[1] == arm_b.args[1]
        for lane in range(8):
            coeffs = [known[o.args[2][lane]] for o in (arm_a, arm_b)]
            assert set(coeffs) == {33, 16896}
            for o, coeff in zip((arm_a, arm_b), coeffs):
                assert known[o.args[3][lane]] == (0xE9F8CC1D if coeff == 33 else 0xACCF6200)
        bword = writer[arm_a.args[1][0]]; assert bword.code == '^'
        parts = [writer[arg[0]] for arg in bword.args[1:]]
        s19 = next(o for o in parts if o.code == '>>' and uniform_is(o.args[2], 19))
        m19 = next(o for o in parts if o is not s19)
        head = writer[s19.args[1][0]]
        assert head.code == 'multiply_add' and head.tag == ('hash_mad', (10, nxt.tag[1][1]))
        assert m19.code == '^' and head.args[0] in m19.args[1:]
        other = next(arg for arg in m19.args[1:] if arg != head.args[0])
        assert uniform_is(other, 0xC761C23C)
        initial = [z.args[0], arm_a.args[0], arm_b.args[0], bword.args[0],
                   head.args[1], head.args[0], s19.args[0], m19.args[0], node]
        matches.append((nxt, mix, tail, shift, e, initial))
    assert len(matches) == 32
    vectors, scalars = {}, {}
    for v, value in known.items():
        scalars.setdefault(value, (v,))
    for o in ops:
        for kind, arg in zip(o.kinds, o.args):
            if kind != 'imm' and len(arg) == 8 and arg[0] in known:
                value = known[arg[0]]
                if uniform_is(arg, value):
                    vectors.setdefault(value, arg)
    setup, emitted, remove = [], [], set()
    def uniform(value):
        value &= MASK
        if value == 0:
            return b.zero
        if value in vectors:
            return vectors[value]
        if value not in scalars:
            out = b.new(1); b.put('load', 'const', [out, value], 100, ('joint_boundary_constant', (value,)))
            setup.append(b.ops.pop()); scalars[value] = out
        out = b.new(8); b.put('valu', 'vbroadcast', [out, scalars[value]], 101, ('joint_boundary_uniform', (value,)))
        setup.append(b.ops.pop()); vectors[value] = out
        return out
    for nxt, mix, tail, shift, e, initial in matches:
        remove.update(id(o) for o in (nxt, mix, tail, shift, e)); values = list(initial)
        for i, (op, w, cc) in enumerate(zip(macro['order'], macro['wiring'], macro['constants'])):
            pool = values+[None, None]
            for j in set(w):
                if j >= len(values):
                    pool[j] = uniform(cc[j-len(values)])
            args = [pool[j] for j in w]
            if op == 'shr16':
                args.append(shift.args[2])
            out = nxt.args[0] if i == len(macro['order'])-1 else b.new(8)
            b.put('valu', 'multiply_add' if op == 'mad' else '>>' if op == 'shr16' else op,
                  [out, *args], nxt.time+i, ('joint_live_boundary', (nxt.tag[1][1], i)))
            emitted.append(b.ops.pop()); values.append(out)
    b.ops = setup+[o for o in ops if id(o) not in remove]+emitted
    b.sink = b.literals = b.literal_flow = 0
    frontier.logical_order(b); ir = research.clean(b)
    after = Counter(o.engine for o in ir.ops)
    return ir, dict(replaced_boundaries=len(matches), removed_ops_per_boundary=5,
                    macro_instructions=len(macro['order']), before=dict(before), after=dict(after),
                    lane_work_before=before['alu']+8*before['valu'],
                    lane_work_after=after['alu']+8*after['valu'],
                    new_setup_operations=len(setup), both_arm_orientations_checked=True,
                    free_e=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--macro', type=Path)
    ap.add_argument('--base-ir', type=Path)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    macro = json.loads(args.macro.read_bytes()) if args.macro else CONTROL
    base = pickle.loads(args.base_ir.read_bytes()) if args.base_ir else None
    ir, report = build(macro, base)
    result = flow.serial_control(ir, args.seeds)
    report.update(macro=macro, scratch=result['scratch'], frozen_seeds=args.seeds,
                  serial_cycles_not_a_score=result['verified'], new_executable=False)
    args.output.write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
