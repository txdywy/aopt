"""Two-key stable ranking dispatch. Microbenchmark, not a submission kernel."""
import json
import random
from tests.frozen_problem import Tree, Input, Machine, DebugInfo, build_mem_image


def build():
    constants = [('const',1100+g,2054+8*g) for g in range(32)]
    constants += [('const',1132,1),('const',1133,2),('const',1134,4),
                  ('const',1135,0)]
    constants += [('const',768+i,0) for i in range(128)]
    constants += [('const',1024+i,0) for i in range(16)]
    constants += [('const',1040,0)]
    code = [{'load':constants[i:i+2]} for i in range(0,len(constants),2)]
    code.append({'valu':[('vbroadcast',1152,1135)]})
    for g in range(0,32,2):
        code.append({'load':[('vload',8*g,1100+g),('vload',8*g+8,1101+g)]})
    for op in range(3):
        for start in range(0,128,12):
            slots = []
            for i in range(start,min(start+12,128)):
                slots.append(('<<',512+i,2*i,1134) if op == 0 else
                             ('+',512+i,512+i,2*i+1) if op == 1 else
                             ('+',512+i,512+i,768+i))
            code.append({'alu':slots})
    code.append({'flow':[('jump_indirect',512)]})
    first = len(code)
    middle = first+128*256
    for pair in range(128):
        for a in range(16):
            for b in range(16):
                slots = [('|',256+2*pair,1024+a,1024+a)]
                if a == b:
                    slots += [('+',257+2*pair,1024+a,1132),
                              ('+',1024+a,1024+a,1133)]
                else:
                    slots += [('|',257+2*pair,1024+b,1024+b),
                              ('+',1024+a,1024+a,1132),
                              ('+',1024+b,1024+b,1132)]
                code.append({'alu':slots,'flow':[
                    ('jump_indirect',513+pair) if pair < 127 else ('jump',middle)]})
    assert len(code) == middle
    # Inclusive parallel scan, then copy shifted counts to exclusive bases.
    for distance in (1,2,4,8):
        # VALU not suitable: scan edges cross vector boundaries.
        # Split into <=12 slots in descending order, preserving old sources.
        indices = list(range(15,distance-1,-1))
        for start in range(0,len(indices),12):
            code.append({'alu':[('+',1024+i,1024+i,1024+i-distance)
                                for i in indices[start:start+12]]})
    for start in range(1,16,12):
        code.append({'alu':[('|',1040+i,1023+i,1023+i)
                            for i in range(start,min(start+12,16))]})
    for start in range(0,16,6):
        code.append({'valu':[('+',512+8*i,512+8*i,1152)
                             for i in range(start,min(start+6,16))]})
    code.append({'flow':[('jump_indirect',512)]})
    second = len(code)
    for pair in range(128):
        for a in range(16):
            for b in range(16):
                code.append({'alu':[('+',256+2*pair,256+2*pair,1040+a),
                                    ('+',257+2*pair,257+2*pair,1040+b)],
                             'flow':[('jump_indirect',513+pair) if pair < 127
                                     else ('halt',)]})
    for packet in code[:first]:
        for i,slot in enumerate(packet.get('load',[])):
            if slot[0] == 'const' and slot[1] == 1135:
                packet['load'][i] = ('const',1135,second-first)
            elif slot[0] == 'const' and 768 <= slot[1] < 896:
                packet['load'][i] = ('const',slot[1],first+256*(slot[1]-768))
    return code,first,second-middle


if __name__ == '__main__':
    code,setup,middle = build()
    cycles = []
    for seed in range(60):
        rng = random.Random(seed)
        keys = ([seed-40]*256 if 40 <= seed < 56 else
                [i%2*15 for i in range(256)] if seed == 56 else
                [rng.randrange(16) for _ in range(256)])
        memory = build_mem_image(Tree(10,[0]*2047),Input(keys,[0]*256,16))
        machine = Machine(memory,code,DebugInfo({}))
        machine.run()
        expected = [None]*256
        for rank,lane in enumerate(sorted(range(256),key=lambda i:keys[i])):
            expected[lane] = rank
        assert machine.cores[0].scratch[256:512] == expected, seed
        assert machine.mem == memory
        cycles.append(machine.cycle)
    assert len(set(cycles)) == 1
    print(json.dumps(dict(verified_inputs=60,cycles=cycles[0],setup_cycles=setup,
                         histogram_cycles=128,prefix_and_retarget_cycles=middle,
                         rank_cycles=128,instructions=len(code),
                         json_bytes=len(json.dumps(code).encode()),
                         warning='Ranking only, not a full kernel.'),indent=2))
