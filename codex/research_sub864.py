"""Explore initialization and record placement below the preserved 868 kernel.

The default standalone submission is never changed by this research tool.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import pickle
from types import SimpleNamespace

from codex import stride4_family as family
from codex.word_ssa import rebuild, allocate
from codex.research_packet_870 import verify


class Builder(family.Builder):
    def __init__(self, *, flow=(), staged=(), cache_flow=False, cache_stage=False,
                 relocate=False, banks=2, pair_flow=False, imm=False, layout='tight', root_scalar=False,
                 coefficients=False, pack_order=False, early_cache=False, sink=0, parallel_bank=False,
                 literals=0, head_fill=False, literal_flow=0, packed_constants=False):
        self.flow_values=set(flow)
        self.stage_values=set(staged)
        self.cache_flow=cache_flow
        self.cache_stage=cache_stage
        self.relocate=relocate
        self.banks=banks
        self.image_base=2310 if relocate else 2086
        self.image_span=256 if layout=='shared' else 224
        self.layout=layout
        self.imm=imm
        self.root_scalar=root_scalar
        self.coefficients=coefficients
        self.pack_order=pack_order
        self.early_cache=early_cache
        self.cache_count=0
        self.sink=sink
        self.parallel_bank=parallel_bank
        self.literals=literals
        self.head_fill=head_fill
        self.literal_flow=literal_flow
        self.packed_constants=packed_constants
        self.pending=[]
        self.paired=[]
        self.replacements={}
        self.pair_flow=pair_flow
        self.stats=Counter()
        super().__init__(True,True)

    def value(self,engine,code,*inputs,time=0,tag=('setup',())):
        if (self.coefficients and engine=='valu' and code=='multiply_add'
            and inputs==(self.vectors.get(16),self.vectors.get(2),self.vectors.get(1))):
            s32=self.s('+',self.scalar[16],self.scalar[16],1)
            s33=self.s('+',s32,self.scalar[1],2)
            out=self.new(8)
            self.paired.append((s33,out,3))
            return out
        if engine=='valu' and code=='vbroadcast' and hasattr(self,'header'):
            source=inputs[0]
            if self.root_scalar and source==(self.header[7],):
                # The initial node XOR can consume one shared scalar directly.
                return (self.header[7],)*8
            constants=[value for value,arg in self.scalar.items() if arg==source]
            val=constants[0] if constants else None
            if self.coefficients and val==16896:
                out=self.new(8)
                self.paired.append((source,out,4))
                return out
            cache=val is None and source!=(self.header[7],)
            cache_index=self.cache_count if cache else -1
            self.cache_count+=int(cache)
            if self.early_cache and cache_index in (1,2):
                out=self.new(8)
                self.paired.append((source,out,time))
                return out
            if val in self.stage_values or (cache and self.cache_stage):
                out=self.new(8)
                self.pending.append((source,out,time))
                self.stats['stage']+=1
                return out
            if val in self.flow_values or (cache and self.cache_flow):
                if self.pair_flow:
                    out=self.new(8)
                    self.paired.append((source,out,time))
                    return out
                return self.flow_broadcast(source,time)
        if engine=='flow' and code=='add_imm' and self.relocate:
            address=inputs[1]
            if 2310<=address<2566 and (address-2310)%8==0 and hasattr(self,'inptr'):
                self.stats['reused_image_pointer']+=1
                return self.inptr[(address-2310)//8]
        return super().value(engine,code,*inputs,time=time,tag=tag)

    def literal(self,value,time=1):
        value&=family.MASK
        if self.imm and value not in self.scalar and value not in (3925396509,0xfd7046c5):
            priority=4 if value==0xb55a4f09 else max(8,time-1)
            self.scalar[value]=super().value('flow','add_imm',(self.zero[0],),value,time=priority)
        return super().literal(value,time)

    def v(self,code,a,b,time,tag=('math',())):
        if self.root_scalar and code=='^' and a and b==(self.header[7],)*8:
            return self.scalar_xor(a,b,time,tag)
        return super().v(code,a,b,time,tag)

    def block(self,source,time):
        free=any(op.engine!='store' and op.kinds[0]=='w' and op.args[0]==source
                 for op in self.ops)
        block=list(source) if free else []
        while len(block)<4:
            block.extend(super().value('alu','|',source,(self.zero[0],),time=max(0,time-1)))
        self.stats['broadcast_copies']+=3 if free else 4
        return tuple(block)

    def flow_broadcast(self,source,time):
        # A four-word block is viewed twice, shifted by four positions.
        # The known half-mask selects that block into both halves of the result.
        block=self.block(source,time)
        left=self.new(4,undefined=True)
        right=self.new(4,undefined=True)
        cond=self.vectors[1][4:]+self.zero[:4]
        out=super().value('flow','vselect',cond,block+right,left+block,time=time)
        self.stats['flow_broadcast']+=1
        return out

    def setup(self):
        super().setup()
        if self.parallel_bank:
            dests={arg[0]:i for i,arg in enumerate(self.bankptr[1:8],1)}
            old={dests[op.args[0][0]]:op for op in self.ops
                 if op.engine=='alu' and op.args[0][0] in dests}
            base=self.bankptr[0];one=self.scalar[1];seven=self.scalar[7];six=self.treeptr[0]
            definitions={1:('+',base,one),3:('+',self.scalar[2047],self.scalar[10]),
                         7:('+',base,seven),2:('^',self.bankptr[3],one),
                         4:('^',self.bankptr[7],seven),5:('^',self.bankptr[7],six),
                         6:('+',base,six)}
            insertion=min(self.ops.index(op) for op in old.values())
            old_ids={id(op) for op in old.values()}
            self.ops=[op for op in self.ops if id(op) not in old_ids]
            ordered=[]
            for i,(code,a,b) in definitions.items():
                op=old[i];op.code=code;op.args=[op.args[0],a,b];op.time=1 if i in (1,3,7) else 2
                ordered.append(op)
            self.ops[insertion:insertion]=ordered
        if self.head_fill:
            replacements={self.treeptr[1]:('+',self.scalar[7],self.scalar[7]),
                          self.treeptr[30]:('-',self.scalar[256],self.scalar[10]),
                          self.treeptr[36]:('cdiv',self.scalar[2054],self.scalar[7]),
                          self.bankptr[6]:('^',self.scalar[2054],self.scalar[10])}
            for op in self.ops:
                if op.engine=='alu' and op.args[0] in replacements:
                    code,a,b=replacements[op.args[0]]
                    op.code=code;op.args=[op.args[0],a,b];op.time=1
        for i in range(0,len(self.paired),2):
            if i+1==len(self.paired):
                source,out,time=self.paired[i]
                replacement=self.flow_broadcast(source,time)
                self.replacements.update(zip(out,replacement))
                continue
            (sa,oa,ta),(sb,ob,tb)=self.paired[i:i+2]
            aa=self.block(sa,ta);bb=self.block(sb,tb)
            cond=self.vectors[1][4:]+self.zero[:4]
            mixed=super().value('flow','vselect',cond,aa+self.new(4,undefined=True),
                                self.new(4,undefined=True)+bb,time=max(ta,tb))
            self.replacements.update(zip(oa,aa+mixed[:4]))
            self.replacements.update(zip(ob,mixed[4:]+bb))
            self.stats['paired_broadcast']+=1
        if self.pack_order:
            # Hash/node consumers are emitted by run(), so known setup
            # priorities provide a deterministic initial ordering here.
            self.pending.sort(key=lambda item:(item[2],item[0][0]))
        for source,out,time in self.pending:
            for lane in range(8):
                self.put('store','store',[self.bankptr[lane],source],max(3,time)+lane//2,
                         ('constant_stage',(source[0],lane)))
            self.put('load','vload',[out,self.bankptr[0]],max(3,time)+4,
                     ('constant_reload',(source[0],)))

    def run(self):
        super().run()
        for op in self.ops:
            for k,arg in enumerate(op.args):
                if op.kinds[k]!='imm' and not op.kinds[k].endswith('w'):
                    op.args[k]=tuple(self.replacements.get(v,v) for v in arg)
        if self.relocate:
            # All walkers must finish reading the transformed values region
            # before final output stores reuse that same memory.
            outputs=[op for op in self.ops if op.tag[0]=='output']
            self.ops=[op for op in self.ops if op.tag[0]!='output']+outputs
        return self


def clean(builder):
    # Share scalar constants with the existing header/constant registers. No
    # vector view is changed by this CSE: only standalone scalar producers.
    known,_=family.evaluate(builder)
    replace={};seen={};kept=[]
    packed={v for op in builder.ops for kind,arg in zip(op.kinds,op.args)
            if kind!='imm' and len(arg)>1 for v in arg}
    for value in builder.zero:seen.setdefault(0,value)
    for op in builder.ops:
        for k,arg in enumerate(op.args):
            if op.kinds[k]!='imm' and not op.kinds[k].endswith('w'):
                op.args[k]=tuple(replace.get(v,v) for v in arg)
        if op.engine!='store' and op.kinds[0].endswith('w'):
            dest=op.args[0]
            if len(dest)==1 and dest[0] in known and op.engine in ('alu','load','flow') and dest[0] not in packed:
                value=known[dest[0]]
                if value in seen:
                    replace[dest[0]]=seen[value]
                    continue
                seen[value]=dest[0]
            elif op.engine=='load':
                for v in dest:
                    if v in known:seen.setdefault(known[v],v)
        kept.append(op)
    builder.ops=kept
    # Remove pure definitions that became dead, retaining all actual stores.
    while True:
        uses={v for op in builder.ops for kind,arg in zip(op.kinds,op.args)
              if kind!='imm' and not kind.endswith('w') for v in arg}
        kept=[op for op in builder.ops if op.engine=='store' or any(v in uses for v in op.args[0])]
        if len(kept)==len(builder.ops):break
        builder.ops=kept
    if builder.sink or builder.literals:
        from codex.word_ssa import kinds
        uses={}
        for op in builder.ops:
            for kind,arg in zip(op.kinds,op.args):
                if kind!='imm' and not kind.endswith('w'):
                    for v in arg:uses[v]=min(uses.get(v,10**9),op.time)
        candidates=[op for op in builder.ops if op.engine=='alu' and len(op.args[0])==1
                    and op.args[0][0] in known
                    and (builder.packed_constants or op.args[0][0] not in packed)]
        candidates.sort(key=lambda op:uses.get(op.args[0][0],-1),reverse=True)
        for op in candidates[:builder.sink]:
            out=op.args[0];value=known[out[0]]
            op.engine='flow';op.code='add_imm';op.kinds=kinds('flow','add_imm')
            op.args=[out,(builder.zero[0],),value]
            op.time=max(10,uses[out[0]]-1)
        builder.stats['arithmetic_to_flow']=min(len(candidates),builder.sink)
        for op in candidates[builder.sink:builder.sink+builder.literals]:
            out=op.args[0];value=known[out[0]]
            op.engine='load';op.code='const';op.kinds=kinds('load','const')
            op.args=[out,value];op.time=max(3,uses[out[0]]-1)
        builder.stats['arithmetic_to_load']=len(candidates[builder.sink:builder.sink+builder.literals])
    if builder.literal_flow:
        from codex.word_ssa import kinds
        uses={}
        for op in builder.ops:
            for kind,arg in zip(op.kinds,op.args):
                if kind!='imm' and not kind.endswith('w'):
                    for v in arg:uses[v]=min(uses.get(v,10**9),op.time)
        candidates=[op for op in builder.ops if op.engine=='load' and op.code=='const'
                    and op.args[1] not in (3925396509,0xfd7046c5,0xb55a4f09)]
        candidates.sort(key=lambda op:uses.get(op.args[0][0],-1),reverse=True)
        for op in candidates[:builder.literal_flow]:
            out,value=op.args
            op.engine='flow';op.code='add_imm';op.kinds=kinds('flow','add_imm')
            op.args=[out,(builder.zero[0],),value]
            op.time=max(8,uses[out[0]]-1)
        builder.stats['literal_to_flow']=min(len(candidates),builder.literal_flow)
    known,events=family.evaluate(builder)
    rows=[(j,op.engine,(op.code,*(0 for _ in op.kinds)),op.tag) for j,op in enumerate(builder.ops)]
    ir=SimpleNamespace(ops=builder.ops,origin=builder.origin,known=known,
                       undef=builder.undef,rows=rows,memory_events=events)
    if builder.pack_order:order_packs(ir)
    return rebuild(ir)


def order_packs(ir):
    """Order exclusive scratch-memory packs by consumer deadlines.

    Each pack writes all eight fixed buffer words and then reloads them. It
    can commute with another complete pack, but its stores cannot interleave
    with that pack's reload. Table and output memory epochs remain unchanged.
    """
    uses={};packs={}
    for op in ir.ops:
        for kind,arg in zip(op.kinds,op.args):
            if kind!='imm' and not kind.endswith('w'):
                for v in arg:uses[v]=min(uses.get(v,10**9),op.time)
    for j,op in enumerate(ir.ops):
        name,coord=op.tag
        if name=='constant_stage':key=('const',coord[0]);order=coord[1]
        elif name=='constant_reload':key=('const',coord[0]);order=8
        elif name=='anchor_store':key=('anchor',coord[0]);order=coord[1]
        elif name=='anchor':key=('anchor',coord[0]);order=8
        else:continue
        pack=packs.setdefault(key,{'ops':[],'due':10**9})
        pack['ops'].append((order,j))
        if order==8:pack['due']=min(uses.get(v,10**9) for v in op.args[0])
    for rank,pack in enumerate(sorted(packs.values(),key=lambda p:p['due'])):
        assert len(pack['ops'])==9
        for order,j in pack['ops']:
            _,engine,slot,tag=ir.rows[j]
            ir.rows[j]=(100000+16*rank+order,engine,slot,tag)


def build(**kwargs):
    previous=family.X,family.Y,family.V
    if kwargs.get('layout')=='shared':
        assert kwargs.get('relocate')
        family.X,family.Y,family.V=(-60,-40),(2310,2430),(4,16)
    elif kwargs.get('relocate'):family.Y=tuple(y+224 for y in family.Y)
    try:
        builder=Builder(**kwargs).run()
        ir=clean(builder)
    finally:family.X,family.Y,family.V=previous
    return ir,builder.stats


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix',type=Path,default=Path('/tmp/aopt-sub864'))
    ap.add_argument('--flow',default='')
    ap.add_argument('--staged',default='')
    ap.add_argument('--cache-flow',action='store_true')
    ap.add_argument('--cache-stage',action='store_true')
    ap.add_argument('--pair-flow',action='store_true')
    ap.add_argument('--imm',action='store_true')
    ap.add_argument('--layout',choices=('tight','shared'),default='tight')
    ap.add_argument('--root-scalar',action='store_true')
    ap.add_argument('--coefficients',action='store_true')
    ap.add_argument('--pack-order',action='store_true')
    ap.add_argument('--early-cache',action='store_true')
    ap.add_argument('--sink',type=int,default=0)
    ap.add_argument('--parallel-bank',action='store_true')
    ap.add_argument('--literals',type=int,default=0)
    ap.add_argument('--head-fill',action='store_true')
    ap.add_argument('--literal-flow',type=int,default=0)
    ap.add_argument('--packed-constants',action='store_true')
    ap.add_argument('--relocate',action='store_true')
    ap.add_argument('--banks',type=int,choices=(1,2),default=2)
    ap.add_argument('--serial',action='store_true')
    args=ap.parse_args()
    parse=lambda value:tuple(int(x,0)&family.MASK for x in value.split(',') if x)
    ir,stats=build(flow=parse(args.flow),staged=parse(args.staged),cache_flow=args.cache_flow,
                   cache_stage=args.cache_stage,relocate=args.relocate,banks=args.banks,pair_flow=args.pair_flow,
                   imm=args.imm,layout=args.layout,root_scalar=args.root_scalar,coefficients=args.coefficients,
                   pack_order=args.pack_order,early_cache=args.early_cache,sink=args.sink,
                   parallel_bank=args.parallel_bank,literals=args.literals,head_fill=args.head_fill,
                   literal_flow=args.literal_flow,packed_constants=args.packed_constants)
    counts=Counter(op.engine for op in ir.ops)
    print({'ops':len(ir.ops),'counts':dict(counts),'lane_work':counts['alu']+8*counts['valu'],
           'changes':dict(stats)},flush=True)
    if args.serial:
        program,words,_=allocate(ir,list(range(len(ir.ops))))
        assert words<=1536,words
        print({'serial_scratch':words,'serial_verified':verify(program,3)},flush=True)
    with open(str(args.prefix)+'.ir.pkl','wb') as f:pickle.dump(ir,f)


if __name__=='__main__':main()
