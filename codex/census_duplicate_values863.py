"""Find arithmetic writes of the 864 program that recompute existing values.

Runs the unchanged frozen machine on several full-32-bit random inputs and
fingerprints each write by its values across seeds. Whole vectors and
high-entropy scalars that repeat an earlier fingerprint are reported.
"""
import sys,random
sys.path.insert(0,'.');sys.path.insert(0,'tests')
from collections import defaultdict,Counter
from frozen_problem import Machine,build_mem_image,Tree,Input
from perf_takehome import KernelBuilder
kb=KernelBuilder();kb.build_kernel(10,2047,256,16);prog=kb.instrs
SEEDS=[11,22,33,44,55]
traces=[]
for sd in SEEDS:
    random.seed(sd)
    f=Tree.generate(10);inp=Input.generate(f,256,16)
    inp.values=[random.getrandbits(32) for _ in inp.values]
    f.values=[random.getrandbits(32) for _ in f.values]
    mem=build_mem_image(f,inp)
    m=Machine(mem,prog,kb.debug_info(),n_cores=1);m.enable_pause=False;m.enable_debug=False
    core=m.cores[0];rec=[]
    for t,ins in enumerate(prog):
        writes=[]
        for eng in ('alu','valu','flow','load'):
            for si,slot in enumerate(ins.get(eng,[])):
                m.scratch_write={};getattr(m,eng)(core,*slot)
                writes.append((eng,si,sorted(m.scratch_write.items())))
        rec.append(writes);m.step(ins,core)
    traces.append(rec)
K=len(SEEDS)
def hi_entropy(fp):
    return len(set(fp))==K and min(fp)>=1<<16
seen_scalar={}   # fp -> first (t,eng,si)
seen_vec={}
dup=Counter();ex=defaultdict(list)
for t in range(len(prog)):
    per=[traces[k][t] for k in range(K)]
    for idx,(eng,si,w) in enumerate(per[0]):
        fps=[tuple(per[k][idx][2][i][1] for k in range(K)) for i in range(len(w))]
        if eng=='valu' or (eng in('flow','load') and len(w)==8):
            vfp=tuple(fps)
            if vfp in seen_vec and seen_vec[vfp][0]<t:
                const=all(len(set(fp))==1 for fp in fps)
                dup[(eng,'vector','const' if const else 'data')]+=1
                if len(ex[(eng,const)])<8: ex[(eng,const)].append((t,si,prog[t][eng][si][0],'first',seen_vec[vfp]))
            seen_vec.setdefault(vfp,(t,eng,si))
        for i,fp in enumerate(fps):
            if eng=='alu':
                if fp in seen_scalar and seen_scalar[fp][0]<t:
                    const=len(set(fp))==1
                    if const or hi_entropy(fp):
                        dup[(eng,'scalar','const' if const else 'data')]+=1
                        if len(ex[('alu',const)])<12: ex[('alu',const)].append((t,si,prog[t][eng][si],'first',seen_scalar[fp]))
            seen_scalar.setdefault(fp,(t,eng,si))
print(dict(dup))
for k,v in ex.items():
    print(k)
    for e in v: print('   ',e)
