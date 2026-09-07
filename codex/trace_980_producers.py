"""Trace actual RAW producers. This is not a whole-program scheduling bound."""
import json
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tests'))
from frozen_problem import Machine, Tree, Input, build_mem_image, N_CORES
from codex.perf_takehome_980 import KernelBuilder
from codex.fuse_table_returns import accesses


class TracedMachine(Machine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writers = {}
        self.nodes = []
        self.targets = []

    def step(self, instr, core):
        pending = []
        for engine, slots in instr.items():
            for slot in slots:
                if engine == 'debug' or slot[0] in ('jump','halt','pause'):
                    continue
                reads, writes = accesses(engine, slot)
                # Track independent SIMD lanes separately: unaligned vectors
                # deliberately have dead leading lanes from another group.
                # Aggregating those reads would invent cross-lane dependence.
                lane_accesses = [(None, reads, writes)]
                if engine == 'valu' and slot[0] != 'vbroadcast':
                    lane_accesses = [(lane, {s+lane for s in slot[2:]}, {slot[1]+lane})
                                     for lane in range(8)]
                elif slot[0] == 'vselect':
                    lane_accesses = [(lane, {s+lane for s in slot[2:]}, {slot[1]+lane})
                                     for lane in range(8)]
                for lane, reads, writes in lane_accesses:
                    parents = sorted({self.writers[w] for w in reads if w in self.writers})
                    depth = 1 + max((self.nodes[p]['depth'] for p in parents), default=0)
                    node = dict(pc=core.pc-1, cycle=self.cycle, engine=engine,
                                slot=slot, lane=lane, parents=parents, depth=depth)
                    index = len(self.nodes)
                    self.nodes.append(node)
                    pending.append((writes,index))
                if slot == ('vselect',679,1191,431,447):
                    self.targets = [(word,self.writers.get(word)) for word in (1191,431,437,447,453)]
        for writes,index in pending:
            for word in writes:
                self.writers[word] = index
        super().step(instr,core)


def main():
    random.seed(0)
    tree=Tree.generate(10); inp=Input.generate(tree,256,16)
    builder=KernelBuilder(); builder.build_kernel(10,2047,256,16)
    machine=TracedMachine(build_mem_image(tree,inp),builder.instrs,builder.debug_info(),n_cores=N_CORES)
    machine.enable_pause=machine.enable_debug=False
    machine.run()
    traces=[]
    for word,index in machine.targets:
        chain=[]
        while index is not None:
            node=machine.nodes[index]
            if node['cycle'] < 950: break
            chain.append({k:node[k] for k in ('pc','cycle','engine','slot','lane','depth')})
            index=max(node['parents'],key=lambda p:machine.nodes[p]['depth'],default=None)
        traces.append(dict(word=word,chain=list(reversed(chain))))
    print(json.dumps({'cycles':machine.cycle,'traces':traces,
                      'scope':'executed lane-level RAW chains only; excludes control, WAR/WAW and memory dependencies'}))


if __name__=='__main__': main()
