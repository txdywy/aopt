"""Cross-round fusion with correlated values already computed by the kernel.

ez: [e,z], e=9*z+C4. arms: [e,z,u,v,b], u=33*b+K1,
v=16896*b+K2, z=u^v. All are real preceding hash intermediates.
The target is 4097*(e XOR (e>>16) XOR N)+C0.
Fixed-node searches allow arbitrary prepared coefficients for that node.
--runtime-node instead appends N as a real input, covers all 30-bit raw
roots, and keeps private coefficients independent of the root. A verified
formula remains a macro, not a score, until actual compiler integration.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from itertools import permutations,product
import hashlib
import gzip
import json
from pathlib import Path
import random
import time

import numpy as np
import z3

from codex.cegis_c1_absorb863 import _instr_options
from codex.scout_joint_wirings863 import write_report
from codex.search_root_boundary863 import C0,MASK,target


C4,C5=0xFD7046C5,0xB55A4F09


def inputs(primary,node,live,runtime_node):
    if live=='ez':
        z=primary;e=(9*z+C4)&MASK
        values=[e,z]
    else:
        b=primary
        u,v=(33*b+0xE9F8CC1D)&MASK,(16896*b+0xACCF6200)&MASK
        z=u^v;e=(9*z+C4)&MASK
        values=[e,z,u,v,b]
    return values+[node] if runtime_node else values


def expression(row,constants,values):
    vals=list(values)
    for op,w,cc in zip(row['order'],row['wiring'],constants):
        pool=vals+list(cc);a=pool[w[0]]
        if op in ('shr16', 'shr'):
            count = 16 if op == 'shr16' else cc[0]
            v = z3.LShR(a, count) if z3.is_bv(a) or z3.is_bv(count) else a >> count
        elif op in ('//', '%', 'cdiv'):
            divisor = cc[0]
            if not z3.is_bv(divisor) and divisor == 0:
                raise ZeroDivisionError('prepared divisor must be nonzero')
            if z3.is_bv(a) or z3.is_bv(divisor):
                if op == '//': v = z3.UDiv(a, divisor)
                elif op == '%': v = z3.URem(a, divisor)
                else:
                    aa = a if z3.is_bv(a) else z3.BitVecVal(int(a),32)
                    dd = divisor if z3.is_bv(divisor) else z3.BitVecVal(int(divisor),32)
                    wide_d = z3.ZeroExt(1,dd)
                    v = z3.Extract(31,0,z3.UDiv(z3.ZeroExt(1,aa)+wide_d-1,wide_d))
            else:
                v = a//divisor if op == '//' else a%divisor if op == '%' else (a+divisor-1)//divisor
        else:
            b=pool[w[1]]
            if op=='mad':v=a*b+pool[w[2]]
            elif op=='^':v=a^b
            elif op=='&':v=a&b
            elif op=='|':v=a|b
            else:raise ValueError(op)
        vals.append(v&MASK)
    return vals[-1]


def primary_path_avoiding(row,n,runtime_node,cut):
    """Retain all primary-data bypasses; the runtime root is a parameter."""
    todo=[n+len(row['order'])-1];visited=set()
    while todo:
        v=todo.pop()
        if v==cut or v in visited:continue
        visited.add(v)
        if v<n:
            if v<n-int(runtime_node):return True
            continue
        i=v-n
        todo.extend(j for j in row['wiring'][i] if j<n+i)
    return False


def lsb_conditions(row,ks,live,runtime_node,node):
    """The unique shifted bit must affect output bit0 for each input parity.

    Every unshifted live word is a T-function of the primary input. Target
    bit0 flips on primary bit16, independently for both primary parities.
    Replace the SHR result bit0 by a free bit and retain all other low bits.
    """
    result=[]
    for raw in ((0,1) if runtime_node else (0x12345678,)):
        nn=(raw^C5) if runtime_node else node
        for p in (0,1):
            initial=[z3.Extract(0,0,v) for v in inputs(z3.BitVecVal(p,32),z3.BitVecVal(nn,32),live,runtime_node)]
            outputs=[]
            for shifted in (0,1):
                vv=list(initial)
                for op,w,cc in zip(row['order'],row['wiring'],ks):
                    pool=vv+[z3.Extract(0,0,k) for k in cc]
                    if op=='shr16':value=z3.BitVecVal(shifted,1)
                    else:
                        a,b=pool[w[0]],pool[w[1]]
                        value=(a*b+pool[w[2]] if op=='mad' else a^b if op=='^' else a&b if op=='&' else a|b)
                    vv.append(value)
                outputs.append(vv[-1])
            result.append(outputs[0]!=outputs[1])
    return result


def make_manifest(live,runtime_node,per_order):
    n=(2 if live=='ez' else 5)+int(runtime_node)
    rows=[];stats=[];rng=random.Random(8642816)
    for bit in ('^','&','|','mad'):
        for order in sorted(set(permutations(('mad','shr16',bit)))):
            if order[-1]=='shr16':continue
            opts=[[(k,) for k in range(n+i)] if op=='shr16' else _instr_options(i+n-1,op)
                  for i,op in enumerate(order)]
            selected=[]
            def accept(w):
                return (all(any(n+j in w[i] for i in range(j+1,3)) for j in range(2))
                        and (not runtime_node or any(n-1 in v for v in w)))
            if per_order:
                found=set();attempts=0
                while len(found)<per_order and attempts<100000:
                    attempts+=1;w=tuple(rng.choice(o) for o in opts)
                    if accept(w):found.add(w)
                selected=sorted(found)
            else:
                selected=[w for w in product(*opts) if accept(w)]
            rows.extend(dict(order=list(order),wiring=[list(w) for w in ww]) for ww in selected)
            stats.append(dict(order=list(order),cases=len(selected)))
    return rows,stats


def solve(row,live,runtime_node,node,seconds,pins=()):
    ks=[z3.BitVecs(f'k{i}a k{i}b',32) for i in range(len(row['order']))]
    solver=z3.SolverFor('QF_BV')
    for i,j,v in pins:solver.add(ks[i][j]==(v&MASK))
    n=(2 if live=='ez' else 5)+int(runtime_node)
    cuts=[n+i for i,op in enumerate(row['order']) if op=='shr16'
          and not primary_path_avoiding(row,n,runtime_node,n+i)]
    if cuts:
        return dict(row,live=live,runtime_node=runtime_node,fixed_node=None if runtime_node else node,
                    status='structural_excluded',certificate='SHR16 is a primary-data word cut',
                    cut=cuts[0],cut_image_at_most=65536,target_image=1<<32,
                    full32_semantics=True,new_executable=False,seconds=0)
    needed=lsb_conditions(row,ks,live,runtime_node,node)
    low=z3.SolverFor('QF_BV');low.add(*needed)
    if low.check()==z3.unsat:
        return dict(row,live=live,runtime_node=runtime_node,fixed_node=None if runtime_node else node,
                    status='dependency_excluded',certificate='Shifted bit0 cannot influence output bit0 for both primary parities',
                    full32_semantics=True,new_executable=False,seconds=0)
    solver.add(*needed)
    prefix_cut=not primary_path_avoiding(row,n,runtime_node,n)
    prefix_mod4=prefix_cut and row['order'][0]!='shr16'
    if prefix_mod4:
        # Initial words are T-functions of the primary word. A cut before
        # any shift must preserve bijectivity modulo four as well as uint32.
        roots=(0x12345678,0) if runtime_node else (0x12345678,)
        for root in roots:
            nn=(root ^ C5) if runtime_node else node
            vv=[expression(dict(order=row['order'][:1],wiring=row['wiring'][:1]),ks[:1],
                           inputs(z3.BitVecVal(p,32),z3.BitVecVal(nn,32),live,runtime_node))&3
                for p in range(4)]
            solver.add(z3.Distinct(*vv))
    rng=random.Random(8642817)
    primaries=list(dict.fromkeys([0,1,2,3,7,15,255,256,511,512,0xffff,0x10000,
                                 0xffff0000,MASK,MASK-1,*(1<<i for i in range(32)),
                                 *(rng.getrandbits(32) for _ in range(12))]))
    points=[(p,0x12345678) for p in primaries]
    if runtime_node:
        roots=[0,1,2,3,1<<29,(1<<30)-1,rng.getrandbits(30),rng.getrandbits(30)]
        points=list(dict.fromkeys(points+[(p,n) for p in (0,1,65536,MASK) for n in roots]))
    def add(point):
        p,raw=point
        primary=z3.BitVecVal(p,32)
        nn=(raw ^ C5) if runtime_node else node
        values=inputs(primary,z3.BitVecVal(nn,32),live,runtime_node)
        solver.add(expression(row,ks,values)==target(values[0],nn))
    started=time.monotonic()
    for p in points:add(p)
    tests=points+[(rng.getrandbits(32),rng.getrandbits(30)) for _ in range(2048)]
    xs=np.array([p for p,_ in tests],dtype=np.uint64)
    ns=np.array([n ^ C5 for _,n in tests],dtype=np.uint64) if runtime_node else np.uint64(node)
    values=inputs(xs,ns,live,runtime_node);wanted=target(values[0],ns)
    result=dict(row,live=live,runtime_node=runtime_node,fixed_node=None if runtime_node else node,
                status='unknown',events=[],full32_semantics=True,new_executable=False,
                pinned_constants=[list(p) for p in pins],necessary_first_prefix_mod4_bijection=prefix_mod4,
                necessary_shifted_lsb_dependency=True)
    while time.monotonic()-started<seconds:
        solver.set(timeout=max(1,int((seconds-time.monotonic()+started)*1000)))
        status=solver.check()
        result['events'].append(dict(samples=len(points),status=str(status)))
        if status!=z3.sat:
            result['status']=str(status)
            if status==z3.unknown:result['reason']=solver.reason_unknown()
            break
        model=solver.model()
        cc=[[model.eval(v,model_completion=True).as_long() for v in pair] for pair in ks]
        with np.errstate(over='ignore'):bad=np.flatnonzero(expression(row,cc,values)!=wanted)
        if not len(bad):
            primary=z3.BitVec('live_primary',32);raw=z3.BitVec('live_raw_root',30)
            nn=(z3.ZeroExt(2,raw)^C5) if runtime_node else z3.BitVecVal(node,32)
            vv=inputs(primary,nn,live,runtime_node)
            ck=[[z3.BitVecVal(v,32) for v in pair] for pair in cc]
            check=z3.SolverFor('QF_BV')
            check.set(timeout=max(1,int((seconds-time.monotonic()+started)*1000)))
            check.add(expression(row,ck,vv)!=target(vv[0],nn))
            checked=check.check()
            if checked==z3.unsat:
                result.update(status='proved_equivalent',constants=cc,
                              macro=dict(row,live=live,runtime_node=runtime_node,constants=cc))
                break
            if checked==z3.unknown:
                result.update(status='verification_unknown',reason=check.reason_unknown());break
            m=check.model()
            take=[(m.eval(primary,model_completion=True).as_long(),
                   m.eval(raw,model_completion=True).as_long() if runtime_node else 0x12345678)]
        else:take=[tests[i] for i in bad[:8]]
        for p in take:
            assert p not in points
            points.append(p);add(p)
    result.update(seconds=round(time.monotonic()-started,4),samples=len(points))
    return result


def job(args):return solve(*args)


def controls(live,runtime_node):
    n=(2 if live=='ez' else 5)+int(runtime_node)
    if runtime_node:
        row=dict(order=['shr16','^','^','mad'],wiring=[[0],[0,n],[n+1,n-1],[n+2,n+3,n+4]])
        nodes=(0,)
    else:
        row=dict(order=['shr16','^','mad'],wiring=[[0],[0,n],[n+1,n+2,n+3]])
        nodes=(0,1<<31)
    results=[solve(row,live,runtime_node,node,30) for node in nodes]
    assert all(r['status']=='proved_equivalent' for r in results),results
    return results


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--live',choices=('ez','arms'),default='ez')
    ap.add_argument('--runtime-node',action='store_true')
    ap.add_argument('--node',type=lambda x:int(x,0),default=0x12345678 ^ C5)
    ap.add_argument('--per-order',type=int,default=0)
    ap.add_argument('--seconds',type=float,default=1)
    ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--refine-from',type=Path)
    ap.add_argument('--reuse-ez',type=Path)
    ap.add_argument('--checkpoint',type=int,default=0)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();source={}
    if args.refine_from:
        encoded=args.refine_from.read_bytes()
        data=gzip.decompress(encoded) if args.refine_from.suffix=='.gz' else encoded
        old=json.loads(data)
        assert old['completed']==len(old['results'])==len(old['manifest'])
        args.live,args.runtime_node,args.node=old['live'],old['runtime_node'],old['node']
        args.per_order=old['per_order']
        rows=[{k:r[k] for k in ('order','wiring')} for r in old['results']
              if r['status'] not in ('unsat','structural_excluded','dependency_excluded','prior_excluded','proved_equivalent')]
        stats=old['orders']
        source=dict(source=str(args.refine_from),source_sha256=hashlib.sha256(encoded).hexdigest())
    else:rows,stats=make_manifest(args.live,args.runtime_node,args.per_order)
    reused=[]
    if args.reuse_ez:
        assert args.live=='arms' and not args.runtime_node and not args.refine_from
        data=args.reuse_ez.read_bytes();prior=json.loads(data)
        assert prior['ez_complete_manifest']==6192 and prior['ez_remaining']==0
        for path,sha in prior['evidence_sha256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
        ez=json.loads(Path('codex/live_boundary863_ez.json').read_text())
        assert ez['node']==args.node and not ez['sampled_not_exhaustive']
        ek={(tuple(r['order']),tuple(tuple(w) for w in r['wiring'])) for r in ez['manifest']}
        for row in rows:
            if any(j in (2,3,4) for w in row['wiring'] for j in w):continue
            compact=dict(order=row['order'],wiring=[[j if j<2 else j-3 for j in w] for w in row['wiring']])
            assert (tuple(compact['order']),tuple(tuple(w) for w in compact['wiring'])) in ek
            reused.append(dict(row,live='arms',runtime_node=False,fixed_node=args.node,
                               status='prior_excluded',compact=compact,
                               certificate='Initial inputs are e/z only; prior complete e/z grammar applies'))
        source.update(reused_ez_audit=str(args.reuse_ez),reused_ez_sha256=hashlib.sha256(data).hexdigest())
    report=dict(live=args.live,runtime_node=args.runtime_node,node=args.node,
                per_order=args.per_order,sampled_not_exhaustive=bool(args.per_order),
                prepared_constants_arbitrary=not args.runtime_node,orders=stats,
                manifest=rows,controls=controls(args.live,args.runtime_node),results=list(reused),completed=len(reused),
                seconds_per_case=args.seconds,workers=args.workers,best_executable_cycles=864,
                new_executable=False,scope='Three ops, one literal SHR16 and MAD plus MAD/XOR/AND/OR; '
                'all specified correlated live inputs, full32 semantics. No global kernel bound.',**source)
    print(len(rows),'cases',args.live,'runtime_node',args.runtime_node,flush=True)
    counts=Counter(r['status'] for r in reused);report['counts']=dict(counts)
    write_report(args.output,report);start=time.monotonic()
    reused_keys={(tuple(r['order']),tuple(tuple(w) for w in r['wiring'])) for r in reused}
    pending=[r for r in rows if (tuple(r['order']),tuple(tuple(w) for w in r['wiring'])) not in reused_keys]
    checkpoint=args.checkpoint or (2000 if len(rows)>20000 else 100)
    with ProcessPoolExecutor(args.workers) as pool:
        futures=[pool.submit(job,(r,args.live,args.runtime_node,args.node,args.seconds)) for r in pending]
        for done,f in enumerate(as_completed(futures),len(reused)+1):
            result=f.result();report['results'].append(result);counts[result['status']]+=1
            if result['status']=='proved_equivalent':print(result,flush=True)
            if done%checkpoint==0 or done==len(rows):
                report.update(completed=done,counts=dict(counts),seconds=round(time.monotonic()-start,3))
                write_report(args.output,report);print(done,dict(counts),report['seconds'],flush=True)


if __name__=='__main__':main()
