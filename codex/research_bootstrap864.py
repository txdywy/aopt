"""Research small-constant staging for the record-family kernel.

All transformations emit the unchanged simulator ISA. The active submission
is deliberately left alone; use packet_modes to schedule and validate a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import pickle

from codex import research_sub864 as research
from codex import stride4_family as family
from codex.word_ssa import allocate, kinds
from codex.research_packet_870 import verify


class Builder(research.Builder):
    def __init__(self, *, small_stage=(), coefficient_only=False, thirds=False, **kwargs):
        self.small_stage=set(small_stage)
        self.coefficient_only=coefficient_only
        self.thirds=thirds
        super().__init__(**kwargs)

    def value(self,engine,code,*inputs,time=0,tag=('setup',())):
        if self.thirds and self.coefficients:
            if (engine=='valu' and code=='multiply_add' and inputs==(
                    self.vectors.get(16),self.vectors.get(2),self.vectors.get(1))):
                s32=self.s('+',self.scalar[16],self.scalar[16],1)
                s33=self.s('+',s32,self.scalar[1],2)
                return self.flow_broadcast(s33,3)
            if (engine=='valu' and code=='vbroadcast'
                    and inputs[0]==self.scalar.get(16896)):
                return self.flow_broadcast(inputs[0],time)
        if (self.coefficient_only and engine=='valu' and code=='vbroadcast'
                and inputs[0]==self.scalar.get(16896)):
            if self.thirds:return self.flow_broadcast(inputs[0],time)
            out=self.new(8)
            self.paired.append((inputs[0],out,time))
            return out
        if (self.thirds and self.early_cache and engine=='valu' and code=='vbroadcast'
                and hasattr(self,'header') and inputs[0]!=(self.header[7],)
                and not any(arg==inputs[0] for arg in self.scalar.values())
                and self.cache_count in (1,2)):
            self.cache_count+=1
            return self.flow_broadcast(inputs[0],time)
        return super().value(engine,code,*inputs,time=time,tag=tag)

    def flow_broadcast(self,source,time):
        if not self.thirds:return super().flow_broadcast(source,time)
        free=any(op.engine!='store' and op.kinds[0]=='w' and op.args[0]==source
                 for op in self.ops)
        block=list(source) if free else []
        while len(block)<3:
            block.extend(family.Builder.value(self,'alu','|',source,(self.zero[0],),
                                             time=max(0,time-1)))
        block=tuple(block)
        # The six useful result words follow the original three-word block.
        # A live eight-word view uses the three originals and five results;
        # the full architectural result still writes all eight words.
        cond=self.vectors[1][5:]+self.zero[:5]
        out=family.Builder.value(self,'flow','vselect',cond,
                block+self.new(5,undefined=True),
                self.new(3,undefined=True)+block+self.new(2,undefined=True),time=time)
        self.stats['third_broadcast']+=1
        self.stats['broadcast_copies']+=2 if free else 3
        return block+out[:5]

    def setup(self):
        super().setup()
        if not self.small_stage:
            return
        # Scalar header words and the c1 vector of twos are already available.
        # Keep scalar users off the staged vector's reload dependency chain.
        old_vectors={v:self.vectors[v&family.MASK] for v in (3,4,-2,9,19)}
        before=len(self.ops)
        s3=self.s('cdiv',self.scalar[16],self.scalar[7],1)
        s4=self.s('+',self.scalar[2],self.scalar[2],2)
        sm2=self.s('-',self.scalar[0],self.scalar[2],2)
        s9=self.s('-',self.scalar[10],self.scalar[1],1)
        s19=self.s('+',self.scalar[16],s3,2)
        scalars={3:s3,4:s4,-2:sm2,9:s9,19:s19}
        added=self.ops[before:]
        del self.ops[before:]
        insertion=next(i+1 for i,op in enumerate(self.ops)
                       if op.engine!='store' and op.args[0]==self.vectors[2])
        self.ops[insertion:insertion]=added
        selected={old_vectors[v]:scalars[v] for v in self.small_stage}
        # The 9 and 19 vector operations remain useful head fillers. Feed them
        # scalar constants instead of waiting for the staged vector of threes.
        if 3 in self.small_stage:
            selected.update({old_vectors[9]:s9,old_vectors[19]:s19})
        kept=[]
        for op in self.ops:
            if op.engine=='valu' and op.args[0] in selected:
                out=op.args[0]
                if out in (old_vectors[9],old_vectors[19]) and 3 in self.small_stage:
                    op.code='vbroadcast';op.kinds=kinds('valu','vbroadcast')
                    op.args=[out,selected[out]];op.time=2 if out==old_vectors[9] else 3
                else:
                    continue
            for k,arg in enumerate(op.args):
                if op.kinds[k]!='imm' and not op.kinds[k].endswith('w') and len(arg)==1:
                    for vec,scalar in selected.items():
                        if arg==(vec[0],):
                            op.args[k]=scalar
                            break
            kept.append(op)
        self.ops=kept
        for v in self.small_stage:
            source=scalars[v];out=old_vectors[v]
            for lane in range(8):
                self.put('store','store',[self.bankptr[lane],source],5+lane//2,
                         ('constant_stage',(source[0],lane)))
            self.put('load','vload',[out,self.bankptr[0]],9,
                     ('constant_reload',(source[0],)))
            self.scalar[v&family.MASK]=source
        self.stats['small_stages']=len(self.small_stage)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix',type=Path,required=True)
    ap.add_argument('--small-stage',default='3,4,-2')
    ap.add_argument('--root-scalar',action='store_true')
    ap.add_argument('--coefficients',action='store_true')
    ap.add_argument('--coefficient-only',action='store_true')
    ap.add_argument('--literals',type=int,default=8)
    ap.add_argument('--literal-flow',type=int,default=11)
    ap.add_argument('--sink',type=int,default=2)
    ap.add_argument('--thirds',action='store_true')
    ap.add_argument('--layout',choices=('shared','tight'),default='shared')
    ap.add_argument('--native-c2',action='store_true')
    ap.add_argument('--serial',action='store_true')
    args=ap.parse_args()
    previous=family.X,family.Y,family.V
    if args.layout=='shared':
        family.X,family.Y,family.V=(-60,-40),(2310,2430),(4,16)
        flow=(-60&family.MASK,-40&family.MASK,2899272192,0xfd7046c5)
        staged=(0xb55a4f09,2430,1026,1027,1028,1029)
    else:
        family.X,family.Y,family.V=(-72,-52),(2266,2378),(-4,4)
        flow=(-72&family.MASK,-52&family.MASK,2899272192,0xfd7046c5)
        if not args.native_c2:flow=(*flow,3925396509)
        staged=(0xb55a4f09,2266,2378,1026,1027,1028,1029)
    try:
        builder=Builder(small_stage=tuple(int(v) for v in args.small_stage.split(',') if v),
            relocate=True,layout=args.layout,banks=1,cache_stage=True,early_cache=True,
            pair_flow=not args.thirds,pack_order=True,sink=args.sink,literals=args.literals,literal_flow=args.literal_flow,
            parallel_bank=True,head_fill=True,root_scalar=args.root_scalar,
            coefficients=args.coefficients,
            coefficient_only=args.coefficient_only,
            thirds=args.thirds,
            flow=flow,staged=staged).run()
        ir=research.clean(builder)
    finally:
        family.X,family.Y,family.V=previous
    counts=Counter(op.engine for op in ir.ops)
    print({'counts':dict(counts),'C':counts['alu']+8*counts['valu'],
           'changes':dict(builder.stats)},flush=True)
    if args.serial:
        program,words,_=allocate(ir,list(range(len(ir.ops))))
        assert words<=1536,words
        print({'serial_scratch':words,'serial_verified':verify(program,3)},flush=True)
    with Path(str(args.prefix)+'.ir.pkl').open('wb') as f:
        pickle.dump(ir,f)


if __name__=='__main__':
    main()
