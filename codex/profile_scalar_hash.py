"""Read-only executed operand census; values are observations, not constants proof."""
from collections import Counter
import json
import random
from codex import perf_takehome_983
from tests.frozen_problem import Tree, Input, Machine, DebugInfo, build_mem_image, reference_kernel


class Census(Machine):
    def step(self, instr, core):
        if not hasattr(self,'writers'):
            self.writers = {}
            self.signatures = Counter()
            self.constant_uses = Counter()
            self.mad_sources = {}
            self.subtract_mads = Counter()
            self.vector_mad_coefficients = Counter()
            self.other_mad_shapes = Counter()
        pending = {}
        new_mads = {}
        for engine in ('alu','valu'):
            for slot in instr.get(engine,[]):
                self.ops[engine+':'+slot[0]] += 1
                if engine == 'alu' and slot[0] in ('&','^','-'):
                    parents = [self.writers.get(a,'initial') for a in slot[2:4]]
                    if slot[0] != '-':
                        parents.sort()
                    self.signatures[slot[0]+' <- '+', '.join(parents)] += 1
                    for pos,addr in enumerate(slot[2:4]):
                        if self.writers.get(addr) == 'load:const':
                            self.constant_uses[f'{slot[0]} operand{pos}={hex(core.scratch[addr])}'] += 1
                    if slot[0] == '-' and self.writers.get(slot[3]) == 'valu:multiply_add':
                        self.subtract_mads[str(self.mad_sources[slot[3]])] += 1
                for addr in range(slot[1],slot[1]+(8 if engine == 'valu' else 1)):
                    pending[addr] = engine+':'+slot[0]
                if engine == 'valu' and slot[0] == 'multiply_add':
                    coefficients = core.scratch[slot[3]:slot[3]+8]
                    label = hex(coefficients[0]) if len(set(coefficients)) == 1 else 'nonuniform'
                    self.vector_mad_coefficients[label] += 1
                    if label not in ('0x1001','0x21','0x4200','0x9','0x2'):
                        boolean = all(core.scratch[slot[2]+lane] in (0,1) for lane in range(8))
                        parents = tuple(self.writers.get(a,'initial') for a in slot[2:5])
                        self.other_mad_shapes[str((boolean,parents))] += 1
                    for lane in range(8):
                        # Observed coefficient/addend, not a claim of static constancy.
                        new_mads[slot[1]+lane] = (core.scratch[slot[3]+lane],core.scratch[slot[4]+lane])
                if engine == 'alu' and slot[0] == '*':
                    values = [core.scratch[a] for a in slot[2:4]]
                    matches = [a for a in values if a in (2,9,16,33,4097,16896)]
                    self.mult[str(matches[0]) if matches else 'other'] += 1
        for slot in instr.get('load',[]):
            dest = slot[1]+(slot[3] if slot[0] == 'load_offset' else 0)
            for addr in range(dest,dest+(8 if slot[0] == 'vload' else 1)):
                pending[addr] = 'load:'+slot[0]
        for slot in instr.get('flow',[]):
            if slot[0] in ('select','vselect','add_imm','coreid'):
                for addr in range(slot[1],slot[1]+(8 if slot[0] == 'vselect' else 1)):
                    pending[addr] = 'flow:'+slot[0]
        super().step(instr,core)
        self.writers.update(pending)
        self.mad_sources.update(new_mads)


if __name__ == '__main__':
    builder = perf_takehome_983.KernelBuilder()
    builder.build_kernel(10,2047,256,16)
    reports = []
    for seed in range(10):
        rng = random.Random(seed)
        tree = Tree(10,[rng.getrandbits(32) for _ in range(2047)])
        inp = Input([0]*256,[rng.getrandbits(32) for _ in range(256)],16)
        mem = build_mem_image(tree,inp)
        reference_kernel(tree,inp)
        machine = Census(mem,builder.instrs,DebugInfo({}))
        machine.ops = Counter()
        machine.mult = Counter()
        machine.run()
        assert machine.mem[2310:] == inp.values
        reports.append(dict(cycles=machine.cycle,ops=dict(machine.ops),
                            signatures=dict(machine.signatures),
                            constant_uses=dict(machine.constant_uses),
                            subtract_mads=dict(machine.subtract_mads),
                            vector_mad_coefficients=dict(machine.vector_mad_coefficients),
                            other_mad_shapes=dict(machine.other_mad_shapes),
                            scalar_multiply_observed_coefficients=dict(machine.mult)))
    keys = sorted(set().union(*(r['ops'] for r in reports)))
    multkeys = sorted(set().union(*(r['scalar_multiply_observed_coefficients'] for r in reports)))
    print(json.dumps(dict(verified_full32_inputs=10,
                         cycles=sorted(set(r['cycles'] for r in reports)),
                         opcode_ranges={k:[min(r['ops'].get(k,0) for r in reports),
                                           max(r['ops'].get(k,0) for r in reports)] for k in keys},
                         multiplier_ranges={k:[min(r['scalar_multiply_observed_coefficients'].get(k,0) for r in reports),
                                               max(r['scalar_multiply_observed_coefficients'].get(k,0) for r in reports)] for k in multkeys},
                         seed_zero_producer_signatures=reports[0]['signatures'],
                         seed_zero_constant_uses=reports[0]['constant_uses'],
                         seed_zero_subtract_mad_inputs=reports[0]['subtract_mads'],
                         seed_zero_vector_mad_coefficients=reports[0]['vector_mad_coefficients'],
                         seed_zero_other_mad_shapes=reports[0]['other_mad_shapes']),indent=2))
