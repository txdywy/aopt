"""Executed resource accounting for the paired-ranking grouping proposal.

This is an additive-design screen, NOT a lower bound on all grouped kernels.
It generously omits ranking setup, mock computation, and node routing costs.
"""
from collections import Counter
import json
from pathlib import Path
import random
from codex.benchmark_bucket_rank_pair import build as rank_build
from codex.benchmark_rank_roundtrip import build as transport_build
from tests.frozen_problem import Tree, Input, Machine, DebugInfo, build_mem_image


class CountMachine(Machine):
    def step(self, instr, core):
        if self.include(core.pc-1):
            self.slots.update({engine:len(slots) for engine,slots in instr.items()})
        return super().step(instr,core)


def count(code, keys, include):
    memory = build_mem_image(Tree(10,[0]*2047),Input(keys,[0]*256,16))
    machine = CountMachine(memory,code,DebugInfo({}))
    machine.slots = Counter()
    machine.include = include
    machine.run()
    return machine.slots


if __name__ == '__main__':
    baseline = json.loads(Path(__file__).with_name('resource_profile_983.json').read_text())['slots']
    code,setup,_ = rank_build()
    reports = []
    transport,phases = transport_build()
    recovery = phases['setup']+phases['forward_transport']+phases['mock_compute_and_rank_clobber']
    transport_slots = count(transport,list(range(256)),lambda pc:
                            phases['setup'] <= pc < phases['setup']+phases['forward_transport']
                            or pc >= recovery)
    for seed in range(20):
        rng = random.Random(seed)
        keys = [rng.randrange(16) for _ in range(256)]
        added = count(code,keys,lambda pc:pc >= setup)+transport_slots
        work = baseline['alu']+8*baseline['valu']+added['alu']+8*added['valu']
        reports.append(dict(seed=seed,added_slots=dict(added),
                            additive_arithmetic_floor=(work+59)//60,
                            arithmetic_work_to_remove_for_899=work-899*60,
                            flow_slots_to_remove_for_899=baseline['flow']+added['flow']-899))
    print(json.dumps(dict(transport_slots=dict(transport_slots),cases=reports,
                         caveat='Retains baseline arithmetic/control; all ranking setup and new node routing omitted. '
                         'A redesign must remove existing work, not merely save tree loads.'),indent=2))
