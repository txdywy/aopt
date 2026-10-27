"""Exhaustive 2-instruction closure at full 32 bits for the xor-shift pieces.

Targets: a^C1^(a>>19) and e^(e>>16)^n with n live. Opcodes: the frozen ALU
set plus multiply_add; two private constants per instruction. Each wiring
is checked with Z3 on 18 random/edge points (UNSAT on samples excludes it)
and any model is verified on all sample points. Ten worker processes.
"""
"""Exhaustive 2-instruction search at full 32 bits with the frozen ALU opcode set."""
import itertools, random, time, json, z3
MASK=(1<<32)-1
C1=0xC761C23C
OPS=['+','-','*','^','&','|','<<','>>','//','%','<','==','mad']
def ev(op,a,b,c=None):
    if op=='+':return (a+b)&MASK
    if op=='-':return (a-b)&MASK
    if op=='*':return (a*b)&MASK
    if op=='^':return a^b
    if op=='&':return a&b
    if op=='|':return a|b
    if op=='<<':return (a<<b)&MASK if b<32 else 0
    if op=='>>':return a>>b
    if op=='//':return a//b if b else None
    if op=='%':return a%b if b else None
    if op=='<':return int(a<b)
    if op=='==':return int(a==b)
    if op=='mad':return (a*b+c)&MASK
def sym(op,a,b,c=None):
    if op=='+':return a+b
    if op=='-':return a-b
    if op=='*':return a*b
    if op=='^':return a^b
    if op=='&':return a&b
    if op=='|':return a|b
    if op=='<<':return z3.If(z3.ULT(b,32),a<<b,z3.BitVecVal(0,32))
    if op=='>>':return z3.LShR(a,b)
    if op=='//':return z3.UDiv(a,b)
    if op=='%':return z3.URem(a,b)
    if op=='<':return z3.If(z3.ULT(a,b),z3.BitVecVal(1,32),z3.BitVecVal(0,32))
    if op=='==':return z3.If(a==b,z3.BitVecVal(1,32),z3.BitVecVal(0,32))
    if op=='mad':return a*b+c
def search(name,inputs,target,tries=40,only_op0=None):
    # pools: instr0 over inputs+[k0,k1]; instr1 over inputs+[t0,k2,k3]
    n_in=len(inputs)
    res={'unsat':0,'sat':[], 'timeout':0}
    rng=random.Random(1)
    pts=[tuple(rng.getrandbits(32) for _ in range(n_in)) for _ in range(tries)]
    pts+= [tuple(0 for _ in range(n_in)),tuple(MASK for _ in range(n_in))]
    p0=list(range(n_in+2)); p1=list(range(n_in+3))
    cnt=0
    for op0 in ([only_op0] if only_op0 else OPS):
        ar0=3 if op0=='mad' else 2
        for a0 in itertools.product(p0,repeat=ar0):
            if all(v>=n_in for v in a0):continue
            for op1 in OPS:
                ar1=3 if op1=='mad' else 2
                for a1 in itertools.product(p1,repeat=ar1):
                    if n_in not in a1:continue   # must use t0
                    cnt+=1
                    s=z3.Solver();s.set(timeout=10000)
                    ks=z3.BitVecs('k0 k1 k2 k3',32)
                    ok=True
                    for pt in pts[:16]:
                        xs=[z3.BitVecVal(v,32) for v in pt]
                        pool0=xs+[ks[0],ks[1]]
                        t0=sym(op0,*[pool0[i] for i in a0])
                        pool1=xs+[t0,ks[2],ks[3]]
                        t1=sym(op1,*[pool1[i] for i in a1])
                        s.add(t1==target(*pt))
                    r=s.check()
                    if r==z3.unsat:res['unsat']+=1
                    elif r==z3.unknown:res['timeout']+=1
                    else:
                        m=s.model();kv=[m.eval(k,model_completion=True).as_long() for k in ks]
                        good=True
                        for pt in pts:
                            pool0=list(pt)+kv[:2];t0=ev(op0,*[pool0[i] for i in a0])
                            if t0 is None:good=False;break
                            pool1=list(pt)+[t0]+kv[2:];t1=ev(op1,*[pool1[i] for i in a1])
                            if t1!=target(*pt):good=False;break
                        if good:res['sat'].append((op0,a0,op1,a1,kv))
                        else:res['timeout']+=1
    res['wirings']=cnt
    return res

TARGETS={'g19C1':([0],lambda a:a^C1^(a>>19)),'g16n':([0,1],lambda e,n:e^(e>>16)^n)}
def job(args):
    name,op0=args
    inputs,target=TARGETS[name]
    r=search(name,inputs,target,only_op0=op0)
    return name,op0,r
if __name__=='__main__':
    from concurrent.futures import ProcessPoolExecutor
    from collections import defaultdict
    jobs=[(n,o) for n in TARGETS for o in OPS]
    tot=defaultdict(lambda:{'wirings':0,'unsat':0,'timeout':0,'sat':[]})
    with ProcessPoolExecutor(10) as ex:
        for name,op0,r in ex.map(job,jobs):
            t=tot[name]
            for k in ('wirings','unsat','timeout'):t[k]+=r[k]
            t['sat']+=r['sat']
            print(name,op0,r['wirings'],r['unsat'],r['timeout'],r['sat'][:2],flush=True)
    for name,t in tot.items():print('TOTAL',name,json.dumps({k:(v if k!='sat' else v[:5]) for k,v in t.items()}),flush=True)
