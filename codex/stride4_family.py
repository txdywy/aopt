"""Experimental record-family compiler, inspired by Josu's public 864 log.

The host builds instructions only. Every tree transform and hash runs as ISA
operations on the unchanged simulator. This file is not the active submission.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

from codex.research_packet_870 import capture,calc,join,graph,verify
from codex.word_ssa import kinds,rebuild,allocate,condense

MASK=(1<<32)-1
X=(-72,-52);Y=(2042,2154);V=(-4,4)


class Builder:
    def __init__(self,head=False,tail=False):
        self.head,self.tail=head,tail
        self.banks=getattr(self,'banks',2)
        self.image_base=getattr(self,'image_base',2086)
        self.image_span=getattr(self,'image_span',224)
        self.ops=[];self.origin=[];self.undef=set();self.initial={};self.scalar={};self.vectors={}
        self.zero=self.new(8)
        self.initial.update({v:frozenset([0]) for v in self.zero})
        self.scalar[0]=(self.zero[0],)
        rows,_=capture();self.hashhint={}
        for t,e,s,(name,coord) in rows:
            if name.startswith('hash_4097'):
                if name=='hash_4097_5':r,g=7,coord[0]
                elif name=='hash_4097_7':r,g=6,coord[0]
                else:r,g=coord[:2]
                self.hashhint[r,g]=t

    def new(self,width=1,undefined=False):
        value=tuple(range(len(self.origin),len(self.origin)+width));self.origin.extend([-1]*width)
        if undefined:self.undef.update(value)
        return value

    def put(self,e,code,args,time=0,tag=('setup',())):
        self.ops.append(SimpleNamespace(engine=e,code=code,args=list(args),kinds=kinds(e,code),time=int(time),tag=tag))

    def value(self,e,code,*inputs,time=0,tag=('setup',())):
        out=self.new(8 if e=='valu' or code=='vselect' else 1)
        self.put(e,code,[out,*inputs],time,tag);return out

    def literal(self,value,time=1):
        value&=MASK
        if value not in self.scalar:self.scalar[value]=self.value('load','const',value,time=time)
        return self.scalar[value]

    def uniform(self,value,time=2):
        value&=MASK
        if value not in self.vectors:self.vectors[value]=self.value('valu','vbroadcast',self.literal(value,max(0,time-1)),time=time)
        return self.vectors[value]

    def v(self,code,a,b,time,tag=('math',())):return self.value('valu',code,a,b,time=time,tag=tag)
    def mad(self,a,b,c,time,tag=('math',())):return self.value('valu','multiply_add',a,b,c,time=time,tag=tag)
    def sel(self,cond,yes,no,time,tag=('select',())):return self.value('flow','vselect',cond,yes,no,time=time,tag=tag)

    def s(self,code,a,b,time=1):return self.value('alu',code,a,b,time=time)

    def loadvec(self,p,time,tag=('load',())):
        out=self.new(8);self.put('load','vload',[out,p],time,tag);return out

    def scalar_xor(self,state,node,time,tag):
        out=self.new(8)
        for lane in range(8):self.put('alu','^',[(out[lane],),(state[lane],),(node[lane],)],time+lane//4,tag)
        return out

    def setup(self):
        self.header=self.loadvec((self.zero[0],),0)
        for lane,val in enumerate((16,2047,256,10,7,2054,2310)):
            self.scalar[val]=(self.header[lane],)
        one=self.value('valu','==',self.zero,self.zero,time=0)
        self.vectors[1]=one;self.scalar[1]=(one[0],)
        flowtimes={}
        if self.head:
            flowtimes={0x7ed55d16:0,4097:1,0xc761c23c:2,2899272192:3}
            for value,t in flowtimes.items():
                self.scalar[value]=self.value('flow','add_imm',(self.zero[0],),value,time=t)
            self.vectors[16]=self.value('valu','vbroadcast',self.scalar[16],time=1)
            self.rootraw=self.value('valu','vbroadcast',(self.header[7],),time=1)
            def derived(value,code,a,b,t):
                vec=self.v(code,a,b,t);self.vectors[value&MASK]=vec;self.scalar[value&MASK]=(vec[0],);return vec
            two=derived(2,'+',one,one,1)
            three=derived(3,'+',one,two,2)
            four=derived(4,'+',two,two,2)
            derived(9,'*',three,three,3)
            derived(19,'+',self.vectors[16],three,3)
            thirtythree=self.mad(self.vectors[16],two,one,2)
            self.vectors[33]=thirtythree;self.scalar[33]=(thirtythree[0],)
            derived(0xfffffffe,'-',self.zero,two,2)
            derived(-4,'-',self.zero,four,3)
            s512=self.s('<<',self.scalar[256],self.scalar[1],1)
            self.scalar[16896]=self.s('*',self.scalar[33],s512,3)
            s26=self.s('+',self.scalar[16],self.scalar[10],1)
            s52=self.s('+',s26,s26,2)
            self.scalar[-52&MASK]=self.s('-',self.scalar[0],s52,3)
            s8=self.s('>>',self.scalar[16],self.scalar[1],1)
            s72=self.s('*',self.scalar[9],s8,4)
            self.scalar[-72&MASK]=self.s('-',self.scalar[0],s72,5)
            six=self.s('-',self.scalar[16],self.scalar[10],1)
            five=self.s('-',six,self.scalar[1],2)
            self.scalar[2042]=self.s('-',self.scalar[2047],five,3)
            s128=self.s('>>',self.scalar[256],self.scalar[1],1)
            s112=self.s('-',s128,self.scalar[16],2)
            self.scalar[2154]=self.s('+',self.scalar[2042],s112,4)
        for value in (4097,0x7ed55d16,0xc761c23c,19,33,3925396509,16896,2899272192,9,0xfd7046c5,16,0xb55a4f09):
            self.uniform(value,flowtimes.get(value,1)+1)
        for value in (2,3,4,0xfffffffe,*X,*Y,*V,1026,1027,1028,1029):self.uniform(value,7 if self.head else 12)
        self.eight=self.s('>>',self.scalar[16],self.scalar[1],1)
        self.scalar[8]=self.eight
        p6=self.s('-',self.scalar[16],self.scalar[10],1)
        self.treeptr=[p6]
        if self.head:
            s32=self.s('+',self.scalar[16],self.scalar[16],1)
            s64=self.s('*',self.scalar[16],self.scalar[4],3)
            def pointers(base,count):
                out=[base]
                for i in range(1,count):
                    if i>=8:a,b,t=out[i-8],s64,4+i//8
                    elif i==1:a,b,t=out[0],self.eight,2
                    elif i==2:a,b,t=out[0],self.scalar[16],2
                    elif i==3:a,b,t=out[2],self.eight,3
                    elif i==4:a,b,t=out[0],s32,2
                    elif i==5:a,b,t=out[4],self.eight,3
                    elif i==6:a,b,t=out[4],self.scalar[16],3
                    else:a,b,t=out[6],self.eight,4
                    out.append(self.s('+',a,b,t))
                return out
            self.treeptr=pointers(p6,64)
        else:
            for i in range(1,64):self.treeptr.append(self.s('+',self.treeptr[-1],self.eight,2+i//2))
        self.inptr=[self.scalar[2310]]
        if self.head:self.inptr=pointers(self.inptr[0],32)
        else:
            for i in range(1,32):self.inptr.append(self.s('+',self.inptr[-1],self.eight,2+i//2))
        self.bankptr=[self.scalar[2054]]
        for i in range(1,8*self.banks):self.bankptr.append(self.s('+',self.bankptr[-1],self.scalar[1],2+i//4))
        self.inputs=[self.loadvec(p,-1 if self.head and i==0 else 2+i//2,('input',(i,))) for i,p in enumerate(self.inptr)]
        raw=[self.loadvec(p,3+i//2,('tree',(i,))) for i,p in enumerate(self.treeptr)]
        self.rawnode={n:(raw[(n+1)//8][(n+1)%8],) for n in range(511)}
        self.cache=[]
        for n in range(7):
            encoded=self.s('^',self.rawnode[n],self.scalar[0xb55a4f09],8+n//2)
            self.cache.append(self.value('valu','vbroadcast',encoded,time=10+n//2))
        # The 40 bilinear records occupy [2086,2310), disjoint from the two
        # eight-word anchor banks. Missing record slots are observer-dead.
        image={};used=set()
        def record(address,node,anchor=None):
            assert address not in used;used.add(address)
            assert all(address+i not in image for i in range(4))
            for field,n in enumerate((2*node+2,2*node+1,node)):
                image[address+field]=self.s('^',self.rawnode[n],self.scalar[0xb55a4f09],20+(address-self.image_base)//8)
            image[address+3]=anchor if anchor is not None else self.new(undefined=True)
        for q in range(8):
            b0,b1,b2=(q>>2&1,q>>1&1,q&1)
            a3=Y[b1]+X[b0]*(b2-2);record(a3,14-q)
            for bits in range(4):
                b3,b4=bits>>1,bits&1;q5=4*q+bits
                record(a3+V[b3]*(b4-2),62-q5,self.treeptr[2*q5])
        assert len(image)==160 and min(image)>=self.image_base and max(image)<self.image_base+self.image_span
        self.image_addresses=sorted({self.image_base+8*((a-self.image_base)//8) for a in image})
        for i,address in enumerate(self.image_addresses):
            words=[]
            for lane in range(8):words.extend(image.get(address+lane,self.new(undefined=True)))
            if self.head and 0<i<=4 and address==self.image_addresses[i-1]+8:ptr=self.s('+',previous_ptr,self.eight,22+i)
            else:ptr=self.value('flow','add_imm',(self.zero[0],),address,time=20+i)
            previous_ptr=ptr
            self.put('store','vstore',[ptr,tuple(words)],35+i,('image',(address,)))
        # Stride-four d7 records use all 64 original tree-read block pointers.
        for block in range(64):
            fields=[]
            for q in (2*block,2*block+1):
                node=254-q
                for n in (2*node+2,2*node+1,node):
                    fields.extend(self.s('^',self.rawnode[n],self.scalar[0xb55a4f09],25+block))
                fields.extend(self.new(undefined=True))
            self.put('store','vstore',[self.treeptr[block],tuple(fields)],30+block,('d7_image',(block,)))

    def hash(self,state,r,g):
        t=self.hashhint[r,g]
        c=lambda value:self.vectors[value&MASK]
        a=self.mad(state,c(4097),c(0x7ed55d16),t,('hash_mad',(r,g)))
        shift=self.v('>>',a,c(19),t+1)
        mix=self.v('^',a,c(0xc761c23c),t+1)
        b=self.v('^',shift,mix,t+2)
        x=self.mad(b,c(33),c(3925396509),t+3)
        y=self.mad(b,c(16896),c(2899272192),t+3)
        z=self.v('^',x,y,t+4)
        e=self.mad(z,c(9),c(0xfd7046c5),t+5)
        shift=self.v('>>',e,c(16),t+6)
        if self.tail and r==15:
            e=self.v('^',e,c(0xb55a4f09),t+6)
        return self.v('^',e,shift,t+7)

    def gather(self,p,state,r,g,anchors=False):
        t=self.hashhint[r,g]-5
        loads=[];root=self.new(8)
        for lane in range(8):
            packet=self.loadvec((p[lane],),t+lane//2,('gather',(r,g,lane)))
            loads.append(packet)
            self.put('alu','^',[(root[lane],),(state[lane],),(packet[2],)],t+1+lane//2,('node_xor',(r,g,lane)))
            if anchors:
                self.put('store','store',[self.bankptr[(g%self.banks)*8+lane],(packet[3],)],t+1+lane//2,('anchor_store',(g,lane)))
        anchor=None
        if anchors:anchor=self.loadvec(self.bankptr[(g%self.banks)*8],t+5,('anchor',(g,)))
        return root,loads,anchor

    def child(self,packets,bits,state,r,g):
        t=self.hashhint[r,g]-3;selected=[None]*8
        padding=self.new(undefined=True)
        extended=tuple(bits)+padding
        for part in range(2):
            pair=[]
            for lane in range(part,8,2):pair.extend(packets[lane][:2])
            pair.append(packets[6+part][2])
            out=self.sel(extended[part:part+8],tuple(pair[1:9]),tuple(pair[:8]),t+part,('child_select',(r,g,part)))
            for i,lane in enumerate(range(part,8,2)):selected[lane]=out[2*i]
        return self.scalar_xor(state,tuple(selected),t+2,('node_xor',(r,g)))

    def run(self):
        self.setup();c=lambda v:self.vectors[v&MASK]
        for g in range(32):
            state=self.inputs[g];bits={};packets={};p3=p5=p7=p9=p10=anchor=None
            for r in range(16):
                t=self.hashhint[r,g];depth=r%11
                if depth==0:
                    node=(self.rootraw if self.head else tuple([self.header[7]]*8)) if r==0 else self.cache[0]
                    state=self.v('^',state,node,t-1) if self.head else self.scalar_xor(state,node,t-1,('node_xor',(r,g)))
                elif depth==1:
                    n=self.sel(bits[r-1],self.cache[1],self.cache[2],t-2)
                    state=self.v('^',state,n,t-1)
                elif depth==2:
                    a=self.sel(bits[r-1],self.cache[3],self.cache[4],t-4)
                    b=self.sel(bits[r-1],self.cache[5],self.cache[6],t-3)
                    n=self.sel(bits[r-2],a,b,t-2)
                    state=self.v('^',state,n,t-1)
                elif depth==3:
                    base=r-3
                    u=self.sel(bits[base],c(X[1]),c(X[0]),t-9)
                    w=self.sel(bits[base+1],c(Y[1]),c(Y[0]),t-8)
                    p3=self.mad(u,bits[r-1],w,t-6)
                    state,packets[r],_=self.gather(p3,state,r,g)
                elif depth==4:
                    state=self.child(packets[r-1],bits[r-1],state,r,g)
                elif depth==5:
                    u=self.sel(bits[3],c(V[1]),c(V[0]),t-7)
                    p5=self.mad(u,bits[4],p3,t-6)
                    state,packets[r],anchor=self.gather(p5,state,r,g,anchors=True)
                elif depth==6:
                    state=self.child(packets[r-1],bits[r-1],state,r,g)
                elif depth==7:
                    a=self.sel(bits[6],c(3),c(2),t-8)
                    digit=self.sel(bits[5],a,bits[6],t-7)
                    p7=self.mad(digit,c(4),anchor,t-6)
                    state,packets[r],_=self.gather(p7,state,r,g)
                elif depth==8:
                    state=self.child(packets[r-1],bits[r-1],state,r,g)
                else:
                    restored=self.v('^',state,c(0xb55a4f09),t-10)
                    parity=self.v('&',restored,c(1),t-9)
                    if depth==9:
                        k0=self.sel(parity,c(1029),c(1028),t-8)
                        k1=self.sel(parity,c(1027),c(1026),t-7)
                        k=self.sel(bits[7],k1,k0,t-6)
                        p9=self.v('-',k,p7,t-5);ptr=p9
                    else:
                        p10=self.mad(p9,c(2),parity,t-5);ptr=p10
                    nodes=[]
                    for lane in range(8):
                        raw=self.loadvec((ptr[lane],),t-4+lane//2,('raw_gather',(r,g,lane)))
                        nodes.append(raw[6])
                    state=self.scalar_xor(restored,tuple(nodes),t-1,('node_xor',(r,g)))
                state=self.hash(state,r,g)
                if r<15 and depth not in (8,9,10):
                    # Only the chain-end bits used as signed multipliers use
                    # OR~1. Every child selector retains ordinary 0/1 bits.
                    signed=r in (2,4,13)
                    bits[r]=self.v('|' if signed else '&',state,c(0xfffffffe) if signed else c(1),t+8)
            output=state if self.tail else self.v('^',state,c(0xb55a4f09),self.hashhint[15,g]+8)
            self.put('store','vstore',[self.inptr[g],output],self.hashhint[15,g]+(8 if self.tail else 9),('output',(g,)))
        return self


def evaluate(builder):
    values=dict(builder.initial);known={v:0 for v in builder.zero};events={}
    memory={i:frozenset([v]) for i,v in enumerate((16,2047,256,10,7,2054,2310))}
    memory.update({i:frozenset([0]) for i in range(2054,2310)})
    for j,op in enumerate(builder.ops):
        e,code,a=op.engine,op.code,op.args
        val=lambda v:values.get(v)
        if e in ('alu','valu'):
            for lane,v in enumerate(a[0]):
                values[v]=val(a[1][0]) if code=='vbroadcast' else calc(code,[val(arg[lane if len(arg)>1 else 0]) for arg in a[1:]])
        elif e=='flow':
            if code=='add_imm':values[a[0][0]]=calc('+',[val(a[1][0]),frozenset([a[2]&MASK])])
            else:
                for lane,v in enumerate(a[0]):
                    cond=val(a[1][lane]);yes=val(a[2][lane]);no=val(a[3][lane])
                    values[v]=no if cond==frozenset([0]) else yes if cond is not None and 0 not in cond else join(yes,no)
        elif e=='load' and code=='const':values[a[0][0]]=frozenset([a[1]&MASK])
        elif e in ('load','store'):
            bases=val(a[1][0] if e=='load' else a[0][0]);assert bases,('unknown pointer',j,op.tag)
            width=8 if code in ('vload','vstore') else 1
            locations={b+l for b in bases for l in range(width)}
            assert all(0<=p<2566 for p in locations),(j,op.tag,min(locations),max(locations))
            events[j]=(e,bases,locations)
            if e=='load':
                for lane,v in enumerate(a[0]):values[v]=join(*(memory.get(b+lane) for b in bases))
            else:
                for b in bases:
                    for lane,v in enumerate(a[1]):
                        address=b+lane;value=val(v)
                        memory[address]=value if len(bases)==1 else join(memory.get(address),value)
        else:raise ValueError((e,code))
        if e!='store':
            for v in a[0]:
                if values.get(v) is not None and len(values[v])==1:known[v]=next(iter(values[v]))
    return known,events


def build_ir(head=False,tail=False):
    b=Builder(head,tail).run();known,events=evaluate(b)
    rows=[(j,op.engine,(op.code,*(0 for _ in op.kinds)),op.tag) for j,op in enumerate(b.ops)]
    ir=SimpleNamespace(ops=b.ops,origin=b.origin,known=known,undef=b.undef,rows=rows,memory_events=events)
    return rebuild(ir)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prefix',type=Path,default=Path('/tmp/aopt-stride4'))
    ap.add_argument('--head',action='store_true');ap.add_argument('--tail',action='store_true')
    args=ap.parse_args();ir=build_ir(args.head,args.tail);counts=Counter(op.engine for op in ir.ops)
    print({'ops':len(ir.ops),'counts':dict(counts),'lane_work':counts['alu']+8*counts['valu']},flush=True)
    # Serial construction is a correctness control, never a claimed fast score.
    serial_times=list(range(len(ir.ops)))
    program,words,shapes=allocate(ir,serial_times)
    print({'serial_scratch':words,'serial_cycles':len(program)},flush=True)
    assert words<=1536,words
    print({'serial_verified':verify(program,3)},flush=True)
    with open(str(args.prefix)+'.ir.pkl','wb') as f:pickle.dump(ir,f)


if __name__=='__main__':main()
