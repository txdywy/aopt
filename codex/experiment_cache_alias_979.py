"""Elide scalar cache copies only after auditing the encoded ISA recipes."""
import base64
import json
import random
import sys
from pathlib import Path
import zlib
from codex import perf_takehome_979 as base
from codex.fuse_table_returns import accesses

p,r=json.loads(zlib.decompress(base64.b85decode(base._CODE)))
aliases={303+8*i:166+i for i in range(16)}
targets=set(aliases.values())
copies=[]
for pc,packet in enumerate(p):
    for e,slots in packet.items():
        for s in slots:
            if e=='flow' and s[0] in ('jump','halt','pause'):continue
            rd,wr=accesses(e,s)
            if pc>8:assert not wr & targets,(pc,s)
            if wr & set(aliases):
                assert e=='alu' and s==['|',s[1],aliases[s[1]],aliases[s[1]]],(pc,s)
                copies.append((pc,s))
assert len(copies)==16,len(copies)
for packet,patches in r:
    for e,slots in packet.items():
        for s in slots:
            if e=='flow' and s[0]=='jump':continue
            rd,wr=accesses(e,s)
            assert not wr & (targets|set(aliases)),s
    # Table changes only scalar source operands, never destinations.
    assert all(e=='alu' and ai in (2,3) for e,si,ai,shift,values in patches)
for pc,s in copies:
    assert pc>8,(pc,s)
    p[pc]['alu'].remove(s)
    assert any(p[pc].values())
def rewrite(packet):
    for e,slots in packet.items():
        for s in slots:
            if e=='alu':
                s[2]=aliases.get(s[2],s[2]);s[3]=aliases.get(s[3],s[3])
            else:
                if e=='flow' and s[0] in ('jump','halt','pause'):continue
                rd,wr=accesses(e,s)
                assert not rd & set(aliases),(e,s)
for packet in p:rewrite(packet)
for packet,patches in r:
    rewrite(packet)
    for patch in patches:patch[4]=[aliases.get(v,v) for v in patch[4]]
used=set()
for packet in p+[packet for packet,patches in r]:
    for e,slots in packet.items():
        for s in slots:
            if e=='flow' and s[0] in ('jump','halt','pause'):continue
            rd,wr=accesses(e,s);used.update(rd|wr)
for packet,patches in r:
    for e,si,ai,shift,values in patches:
        # Earlier assertion guarantees every varying operand is a scalar
        # ALU source, so each recipe value names exactly one scratch word.
        used.update(values)
free=sorted(set(range(1536))-used)
ranges=[]
for word in free:
    if ranges and ranges[-1][1]==word:ranges[-1][1]+=1
    else:ranges.append([word,word+1])
base._CODE=base64.b85encode(zlib.compress(json.dumps([p,r]).encode())).decode()
builder=base.KernelBuilder();builder.build_kernel(10,2047,256,16)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from frozen_problem import Machine,Tree,Input,build_mem_image,reference_kernel2,N_CORES
scores=[]
for seed in range(23):
    random.seed(seed);tree=Tree.generate(10);inp=Input.generate(tree,256,16)
    def value(i):
        return random.getrandbits(32) if seed<20 else (0,0xffffffff,0xffffffff if i%2 else 0)[seed-20]
    tree.values=[value(i) for i in range(len(tree.values))]
    inp.values=[value(i) for i in range(256)]
    mem=build_mem_image(tree,inp);machine=Machine(mem,builder.instrs,builder.debug_info(),n_cores=N_CORES)
    machine.enable_pause=machine.enable_debug=False;machine.run()
    for ref in reference_kernel2(mem):pass
    assert machine.mem[ref[6]:ref[6]+256]==ref[ref[6]:ref[6]+256],seed
    scores.append(machine.cycle)
print(json.dumps({'cycles':scores,'scalar_copies_removed':len(copies),
                  'unused_scratch_words':len(free),'unused_half_open_ranges':ranges}))
