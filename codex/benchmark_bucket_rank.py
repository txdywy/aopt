"""Frozen-ISA stable16-bucket ranks, no supplied ranks and no extra memory.

Two chained dispatch passes implement dynamic scratch counter/base lookup.
This is a ranking microbenchmark, not a full hash kernel submission.
"""
import json
import random
from tests.frozen_problem import Tree,Input,Machine,DebugInfo,build_mem_image


def build():
    constants = [('const',1100+g,2054+8*g) for g in range(32)]
    constants += [('const',1132,1),('const',1133,128),('const',1134,0)]
    constants += [('const',768+i,0) for i in range(8)]
    constants += [('const',1024+i,0) for i in range(16)] + [('const',1040,0)]
    code = [{'load':constants[i:i+2]} for i in range(0,len(constants),2)]
    code.append({'valu':[('vbroadcast',1144,1133),('vbroadcast',1152,1134)]})
    for g in range(0,32,2):
        code.append({'load':[('vload',8*g,1100+g),('vload',8*g+8,1101+g)]})
    for g in range(1,32):
        code.append({'valu':[('+',768+8*g,760+8*g,1144)]})
    for g in range(0,32,6):
        code.append({'valu':[('+',512+8*h,8*h,768+8*h)
                             for h in range(g,min(g+6,32))]})
    code.append({'flow':[('jump_indirect',512)]})
    first = len(code)
    middle = first+4096
    for lane in range(256):
        for bucket in range(16):
            code.append({'alu':[('|',256+lane,1024+bucket,1024+bucket),
                                ('+',1024+bucket,1024+bucket,1132)],
                         'flow':[('jump_indirect',513+lane) if lane < 255
                                 else ('jump',middle)]})
    assert len(code) == middle
    for bucket in range(1,16):
        code.append({'alu':[('+',1040+bucket,1039+bucket,1023+bucket)]})
    for g in range(0,32,6):
        code.append({'valu':[('+',512+8*h,512+8*h,1152)
                             for h in range(g,min(g+6,32))]})
    code.append({'flow':[('jump_indirect',512)]})
    second = len(code)
    for lane in range(256):
        for bucket in range(16):
            code.append({'alu':[('+',256+lane,256+lane,1040+bucket)],
                         'flow':[('jump_indirect',513+lane) if lane < 255 else ('halt',)]})
    for packet in code[:first]:
        if 'load' in packet:
            for i,slot in enumerate(packet['load']):
                if slot[0] == 'const' and slot[1] == 1134:
                    packet['load'][i] = ('const',1134,second-first)
                elif slot[0] == 'const' and 768 <= slot[1] < 776:
                    packet['load'][i] = ('const',slot[1],first+16*(slot[1]-768))
    return code,first,second-middle


if __name__ == '__main__':
    code,setup,middle = build()
    cycles = []
    for seed in range(43):
        rng = random.Random(seed)
        keys = ([0]*256 if seed == 40 else [15]*256 if seed == 41
                else [i%2*15 for i in range(256)] if seed == 42
                else [rng.randrange(16) for _ in range(256)])
        memory = build_mem_image(Tree(10,[0]*2047),Input(keys,[0]*256,16))
        machine = Machine(memory,code,DebugInfo({}))
        machine.run()
        expected = [None]*256
        for rank,lane in enumerate(sorted(range(256),key=lambda i:keys[i])):
            expected[lane] = rank
        assert machine.cores[0].scratch[256:512] == expected
        assert machine.mem == memory
        cycles.append(machine.cycle)
    assert len(set(cycles)) == 1
    print(json.dumps({'verified_inputs':43,'cycles':cycles[0],
                      'setup_cycles':setup,'histogram_cycles':256,
                      'prefix_and_retarget_cycles':middle,'rank_cycles':256,
                      'instructions':len(code),'scratch_high_water':1160,
                      'warning':'Ranking only; no value transport/hash/output recovery.'},indent=2))
