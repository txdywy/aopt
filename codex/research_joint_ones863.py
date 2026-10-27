"""Construct ones and nonzero uniform constants in a self-masked FLOW chain.

Every temporary zero is an initially zero scratch word whose live range ends
at its in-place overwrite. Results are research graphs, not contest scores.
"""
from collections import Counter
from copy import deepcopy
import argparse
import json
from pathlib import Path

from codex import research_flow_stage863 as flow
from codex import research_frontier863 as frontier
from codex import research_stage864 as stage
from codex import research_sub864 as research
from codex.bound_boundary_no_solver863 import prefix, suffix
from codex.packet_budget_bound import stream_bounds
from codex.packet_modes import expand
from codex.word_ssa import rebuild

NAMES = {'c0': 0x7ed55d16, 'c1': 0xc761c23c, '3': 3, '9': 9, '19': 19}


def pack(builder, names):
    if not names or len(names) != len(set(names)) or any(n not in NAMES for n in names):
        raise ValueError('select distinct constants from c0,c1,3,9,19')
    oldones = builder.vectors[1]
    vectors = [builder.vectors[NAMES[name]] for name in names]
    writer = {v: op for op in builder.ops if op.engine != 'store' for v in op.args[0]}
    producers = [writer[oldones[0]], *(writer[v[0]] for v in vectors)]
    assert producers[0].engine == 'valu' and producers[0].code == '=='
    assert producers[0].args[1:] == [builder.zero, builder.zero]
    assert all(op.engine == 'valu' and op.args[0] == vec
               for op, vec in zip(producers[1:], vectors))
    created = []
    def emit(engine, code, args, time, tag):
        builder.put(engine, code, args, time, tag)
        created.append(builder.ops.pop())
    def initzeros(n):
        words = builder.new(n)
        builder.initial.update({v: frozenset([0]) for v in words})
        return words
    one = builder.new()
    emit('alu', '==', [one, (builder.zero[0],), (builder.zero[0],)],
         0, ('joint_ones_root', ()))
    roots = {}
    for name, op in zip(names, producers[1:]):
        if name in ('c0', 'c1'):
            assert op.code == 'vbroadcast'
            roots[name] = op.args[1]
    if '3' in names:
        roots['3'] = builder.new()
        emit('alu', 'cdiv', [roots['3'], builder.scalar[16], builder.scalar[7]],
             1, ('joint_ones_const', ('3',)))
    if '9' in names:
        roots['9'] = builder.new()
        emit('alu', '-', [roots['9'], builder.scalar[10], one],
             1, ('joint_ones_const', ('9',)))
    if '19' in names:
        roots['19'] = builder.new()
        emit('alu', '+', [roots['19'], builder.scalar[16],
                          roots.get('3', builder.scalar[3])],
             2, ('joint_ones_const', ('19',)))
    def copies(root, count, label):
        result = list(root)
        for i in range(count):
            dest = builder.new()
            emit('alu', '|', [dest, root, (builder.zero[0],)], 2,
                 ('joint_ones_seed', (label, i)))
            result.extend(dest)
        return tuple(result)
    previous = copies(one, 3, 'ones')
    final_seed = copies(roots[names[-1]], 3, names[-1])
    output_vectors = []
    for rank, name in enumerate(names):
        output = builder.new(8)
        if rank == len(names)-1:
            condition = previous + initzeros(4)
            no = builder.new(4, undefined=True) + final_seed
        else:
            seed = copies(roots[name], 1, name)
            condition = previous + seed + initzeros(2)
            no = builder.new(2, undefined=True) + previous + seed
        emit('flow', 'vselect', [output, condition, condition, no], 4 + rank,
             ('joint_ones_pack', (rank,)))
        output_vectors.append(previous + output[:4])
        previous = output[4:]
    output_vectors.append(previous + final_seed)
    mapping = {a:b for old, new in zip([oldones, *vectors], output_vectors)
               for a,b in zip(old, new)}
    # Mixed masks are used solely as FLOW predicates. The final constant is
    # nonzero, so its tail can replace ones next to the existing zero vector.
    mask_mapping = dict(zip(oldones, output_vectors[-1]))
    oldones_set, zero_set = set(oldones), set(builder.zero)
    kept = [op for op in builder.ops if all(op is not p for p in producers)]
    for op in kept:
        for k, (kind, arg) in enumerate(zip(op.kinds, op.args)):
            if kind == 'imm' or kind.endswith('w'):
                continue
            mixed = len(arg) == 8 and bool(set(arg) & oldones_set) and bool(set(arg) & zero_set)
            if mixed:
                assert op.engine == 'flow' and op.code == 'vselect' and k == 1
                op.args[k] = tuple(mask_mapping.get(v, v) for v in arg)
            else:
                op.args[k] = tuple(mapping.get(v, v) for v in arg)
    insertion = next(i for i, op in enumerate(kept) if op.tag[0] == 'hash_mad')
    builder.ops = kept[:insertion] + created + kept[insertion:]
    builder.vectors[1] = output_vectors[0]
    builder.scalar[1] = (output_vectors[0][0],)
    for name, vector in zip(names, output_vectors[1:]):
        builder.vectors[NAMES[name]] = vector
        builder.scalar[NAMES[name]] = (vector[0],)
    builder.stats['joint_ones_uniforms'] = len(names)+1
    builder.stats['joint_ones_flows'] = len(names)
    return dict(names=['ones', *names], vectors=[list(v) for v in output_vectors])


def build(names):
    builder = deepcopy(flow.original())
    layout = pack(builder, tuple(names))
    frontier.logical_order(builder)
    builder.sink = builder.literals = builder.literal_flow = 0
    for _ in range(3):
        stage.prune_select_inputs(builder)
        ir = research.clean(builder)
    return ir, builder, layout


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--names', default='c0')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--controls', action='store_true')
    args = ap.parse_args()
    ir, builder, layout = build(args.names.split(','))
    counts = Counter(op.engine for op in ir.ops)
    work = counts['alu'] + 8*counts['valu']
    streams = stream_bounds(ir)
    expanded = expand(deepcopy(ir))
    expanded.views = [a for _, operands, _, _ in expanded.groups for a in operands]
    rebuild(expanded)
    head, tail = prefix(expanded), suffix(expanded)
    lower = max((work + head['waste_lower_bound'] + tail['waste_lower_bound'] + 59)//60,
                *(v['cycle_lower_bound'] for v in streams.values()))
    result = dict(layout=layout, counts=dict(counts), arithmetic_lanes=work,
                  prefix=head, suffix=tail, streams=streams, cycle_lower_bound=lower,
                  new_executable=False,
                  scope='This concrete self-masked chain; bounds and serial replay are not scores.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('arithmetic_lanes','cycle_lower_bound','prefix')}), flush=True)
    if args.controls:
        control=flow.serial_control(deepcopy(ir), 2)
        result['serial_control']=dict(scratch=control['scratch'],
            serialized_cycles=control['verified'],seeds=2)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'serial_control':result['serial_control']}), flush=True)


if __name__ == '__main__':
    main()
