"""Frozen-ISA permutation transport microbenchmark, NOT a full kernel.

Ranks are supplied as input, not computed. Uses the actual index buffer for
values and optionally the original value buffer for IDs after inputs have
been loaded. Preserves the tree; no extra memory or simulator changes.
"""
import json
import random
import argparse
from tests.frozen_problem import Tree, Input, Machine, DebugInfo, build_mem_image


def program(with_ids=False):
    result = []
    constants = [('const',1024+g,2054+8*g) for g in range(32)]
    constants += [('const',1056+g,2310+8*g) for g in range(32)]
    constants += [('const',1100,2054)]
    if with_ids:
        constants += [('const',1114,256)]
    for i in range(0,len(constants),2):
        result.append({'load':constants[i:i+2]})
    result.append({'valu':[('vbroadcast',1104,1100)]})
    if with_ids:
        result[-1]['valu'].append(('vbroadcast',1120,1114))
    for g in range(32):
        result.append({'load':[('vload',g*8,1056+g),
                               ('vload',256+g*8,1024+g)]})
    ready = len(result)
    for g in range(0,32,6):
        result.append({'valu':[('+',256+8*h,256+8*h,1104)
                               for h in range(g,min(g+6,32))]})
    for lane in range(0,256,2):
        result.append({'store':[('store',256+lane,lane),
                                ('store',257+lane,lane+1)]})
    if with_ids:
        for g in range(0,32,6):
            result.append({'valu':[('+',256+8*h,256+8*h,1120)
                                   for h in range(g,min(g+6,32))]})
        result.append({'load':[('const',1112,0),('const',1113,1)]})
        for lane in range(0,256,2):
            packet = {'store':[('store',256+lane,1112),('store',257+lane,1113)]}
            if lane+2 < 256:
                # Stores read the old IDs; newly loaded IDs become visible
                # next cycle, exactly matching frozen-machine semantics.
                packet['load'] = [('const',1112,lane+2),('const',1113,lane+3)]
            result.append(packet)
    for g in range(0,32,2):
        result.append({'load':[('vload',512+g*8,1024+g),
                               ('vload',520+g*8,1025+g)]})
    if with_ids:
        for g in range(0,32,2):
            result.append({'load':[('vload',768+g*8,1056+g),
                                   ('vload',776+g*8,1057+g)]})
    return result, ready


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-ids', action='store_true')
    args = parser.parse_args()
    code, ready = program(args.with_ids)
    timings = []
    for seed in range(40):
        rng = random.Random(seed)
        ranks = list(range(256))
        if seed == 1:
            ranks.reverse()
        elif seed > 1:
            rng.shuffle(ranks)
        values = [rng.getrandbits(32) for _ in ranks]
        memory = build_mem_image(Tree(10,[0]*2047), Input(ranks,values,16))
        machine = Machine(memory, code, DebugInfo({}))
        machine.run()
        expected = [None]*256
        for rank,value in zip(ranks,values):
            expected[rank] = value
        assert machine.mem[2054:2310] == expected
        assert machine.cores[0].scratch[512:768] == expected
        assert machine.mem[:2054] == memory[:2054]
        if args.with_ids:
            expected_ids = [None]*256
            for lane,rank in enumerate(ranks):
                expected_ids[rank] = lane
            assert machine.mem[2310:] == expected_ids
            assert machine.cores[0].scratch[768:1024] == expected_ids
            restored = [None]*256
            for value,lane in zip(machine.cores[0].scratch[512:768],expected_ids):
                restored[lane] = value
            assert restored == values
        else:
            assert machine.mem[2310:] == memory[2310:]
        assert len(machine.mem) == 2566
        timings.append(machine.cycle)
    assert len(set(timings)) == 1
    print(json.dumps({'verified_permutations':40, 'cycles_including_input_setup':timings[0],
                      'setup_packets':ready, 'transport_packets_after_inputs_ready':len(code)-ready,
                      'instructions':len(code), 'memory_words':2566,
                      'scratch_high_water':1128 if args.with_ids else 1112,
                      'with_ids':args.with_ids,
                      'excluded_costs':['rank computation','node distribution','hash','ISA output unpermutation'] + ([] if args.with_ids else ['original-ID transport'])},indent=2))
