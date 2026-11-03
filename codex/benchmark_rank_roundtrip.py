"""Real-ISA rank-map transport and inverse recovery, not a challenge kernel.

The ranks are supplied. A mock XOR computation changes every sorted value
to ensure recovery cannot accidentally reuse the original input values.
Rank scratch is erased to force restoration from the saved memory map.
"""
import json
import random
from codex.benchmark_rank_scatter import program
from tests.frozen_problem import Tree, Input, Machine, DebugInfo, build_mem_image


def build():
    code, setup = program()
    # Save absolute sorted-value addresses in ORIGINAL lane order.
    for g in range(0,32,2):
        code.append({'store':[('vstore',1056+g,256+8*g),
                              ('vstore',1057+g,264+8*g)]})
    transport_end = len(code)
    code.append({'load':[('const',1112,0xA5A5A5A5)]})
    code.append({'valu':[('vbroadcast',1120,1112)]})
    for g in range(0,32,6):
        code.append({'valu':[('^',512+8*h,512+8*h,1120)
                             for h in range(g,min(g+6,32))]})
    for g in range(0,32,2):
        code.append({'store':[('vstore',1024+g,512+8*g),
                              ('vstore',1025+g,520+8*g)]})
    for g in range(0,32,6):
        code.append({'valu':[('^',256+8*h,256+8*h,256+8*h)
                             for h in range(g,min(g+6,32))]})
    recovery_start = len(code)
    for g in range(0,32,2):
        code.append({'load':[('vload',256+8*g,1056+g),
                             ('vload',264+8*g,1057+g)]})
    for lane in range(0,256,2):
        code.append({'load':[('load',768+lane,256+lane),
                             ('load',769+lane,257+lane)]})
    for g in range(0,32,2):
        code.append({'store':[('vstore',1056+g,768+8*g),
                              ('vstore',1057+g,776+8*g)]})
    return code, {'setup':setup,'forward_transport':transport_end-setup,
                  'mock_compute_and_rank_clobber':recovery_start-transport_end,
                  'inverse_recovery':len(code)-recovery_start}


if __name__ == '__main__':
    code, phases = build()
    for seed in range(40):
        rng = random.Random(seed)
        ranks = list(range(256))
        if seed == 1:
            ranks.reverse()
        elif seed > 1:
            rng.shuffle(ranks)
        values = [rng.getrandbits(32) for _ in ranks]
        memory = build_mem_image(Tree(10,[0]*2047),Input(ranks,values,16))
        machine = Machine(memory,code,DebugInfo({}))
        machine.run()
        expected = [v ^ 0xA5A5A5A5 for v in values]
        assert machine.mem[2310:] == expected
        assert machine.cores[0].scratch[768:1024] == expected
        assert machine.mem[:2054] == memory[:2054]
        assert len(machine.mem) == 2566
        assert machine.cycle == len(code)
    print(json.dumps({'verified_permutations':40,'total_cycles':len(code),
                      'phase_cycles':phases,
                      'transport_plus_recovery':phases['forward_transport']+phases['inverse_recovery'],
                      'scratch_high_water':1128,'memory_words':2566,
                      'warning':'Ranks supplied; mock computation is not hash. No full-kernel score.'},indent=2))
