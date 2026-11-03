"""Try to fuse a round boundary with a prepared, shared root node.

Target: 4097*(x XOR (x>>16) XOR N)+C0 (uint32).
N is fixed for each query, while every private constant is arbitrary and
may depend on N. Thus the search gives node preprocessing every advantage.
It covers three instructions with exactly one literal SHR16, plus either
MAD and XOR/AND/OR, or two MADs. It does not cover arbitrary shift amounts,
other instructions, other live words, or state-reachability restrictions.
All queries and counterexample checks retain full 32-bit semantics.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import permutations, product
from pathlib import Path
import random
import time

import numpy as np
import z3

from codex.cegis_c1_absorb863 import _instr_options
from codex.scout_joint_wirings863 import write_report


MASK, C0 = 0xFFFFFFFF, 0x7ED55D16


def target(x, node):
    shifted = z3.LShR(x,16) if z3.is_bv(x) else x >> 16
    return (4097*(x ^ shifted ^ node)+C0) & MASK


def expression(order, wiring, constants, x):
    vals = [x]
    for op,w,kk in zip(order,wiring,constants):
        pool = vals+list(kk)
        args = [pool[j] for j in w]
        if op == 'shr16':
            value = z3.LShR(args[0],16) if z3.is_bv(args[0]) else args[0] >> 16
        elif op == 'mad':
            value = args[0]*args[1]+args[2]
        else:
            a,b = args
            value = a ^ b if op == '^' else a & b if op == '&' else a | b
        vals.append(value & MASK)
    return vals[-1]


def manifest():
    rows=[]
    for bit in ('^','&','|','mad'):
        for order in sorted(set(permutations(('mad','shr16',bit)))):
            # A final literal SHR16 cannot produce the target permutation.
            if order[-1] == 'shr16':
                continue
            opts = [[(i,) for i in range(j+1)] if op == 'shr16' else _instr_options(j,op)
                    for j,op in enumerate(order)]
            if order[0] == 'mad':
                opts[0] = [(0,1,2),(0,0,1),(0,0,0)]
            for wiring in product(*opts):
                if all(any(j+1 in wiring[i] for i in range(j+1,3)) for j in range(2)):
                    rows.append(dict(order=list(order),wiring=[list(w) for w in wiring]))
    return rows


def joint_target(x,node):
    return target((9*x+0xFD7046C5)&MASK,node)


def solve(row, node, seconds, pins=(), joint=False):
    order,wiring = row['order'],row['wiring']
    constants = [z3.BitVecs(f'k{i}a k{i}b',32) for i in range(len(order))]
    s=z3.SolverFor('QF_BV')
    affine_wordcut=(order[0]=='mad' and tuple(wiring[0])==(0,1,2)
                    and not any(0 in w for w in wiring[1:]))
    if affine_wordcut:
        # The target is a permutation. Retain every raw-input bypass before
        # applying this necessary condition to the first affine result.
        s.add(constants[0][0] & 1 == 1)
    for i,j,v in pins:
        s.add(constants[i][j] == (v & MASK))
    rng=random.Random(8642810)
    reference=joint_target if joint else target
    points=list(dict.fromkeys([0,1,2,3,7,15,255,256,511,512,0xffff,0x10000,
                              0x10001,0x1ffff,0xffff0000,MASK,MASK-1,node,node^MASK,
                              *(1<<i for i in range(32)),
                              *(rng.getrandbits(32) for _ in range(12))]))
    def add(p):
        x=z3.BitVecVal(p,32)
        s.add(expression(order,wiring,constants,x) == reference(x,node))
    start=time.monotonic()
    for p in points:
        add(p)
    xs=np.array(points+[rng.getrandbits(32) for _ in range(2048)],dtype=np.uint64)
    want=reference(xs,np.uint64(node))
    result=dict(row,node=node,status='unknown',events=[],full32_semantics=True,
                joint_last_mad=joint,affine_wordcut_odd_multiplier=affine_wordcut,
                pinned_constants=[list(p) for p in pins],new_executable=False)
    while time.monotonic()-start < seconds:
        s.set(timeout=max(1,int((seconds-time.monotonic()+start)*1000)))
        status=s.check()
        result['events'].append(dict(samples=len(points),status=str(status)))
        if status != z3.sat:
            result['status']=str(status)
            if status == z3.unknown:
                result['reason']=s.reason_unknown()
            break
        model=s.model()
        cc=[[model.eval(k,model_completion=True).as_long() for k in pair] for pair in constants]
        got=expression(order,wiring,cc,xs)
        bad=np.flatnonzero(got!=want)
        if not len(bad):
            x=z3.BitVec('boundary_counterexample',32)
            kk=[[z3.BitVecVal(k,32) for k in pair] for pair in cc]
            check=z3.SolverFor('QF_BV')
            check.set(timeout=max(1,int((seconds-time.monotonic()+start)*1000)))
            check.add(expression(order,wiring,kk,x)!=reference(x,node))
            checked=check.check()
            if checked == z3.unsat:
                result.update(status='proved_equivalent',constants=cc)
                break
            if checked == z3.unknown:
                result.update(status='verification_unknown',reason=check.reason_unknown())
                break
            bad_points=[check.model().eval(x,model_completion=True).as_long()]
        else:
            bad_points=[int(xs[i]) for i in bad[:8]]
        for p in bad_points:
            assert p not in points
            points.append(p)
            add(p)
    result.update(seconds=round(time.monotonic()-start,4),samples=len(points))
    return result


def job(args):
    return solve(*args)


def controls(joint=False):
    row=(dict(order=['mad','shr16','^','mad'],wiring=[[0,1,2],[1],[1,2],[3,4,5]]) if joint else
         dict(order=['shr16','^','mad'],wiring=[[0],[0,1],[2,3,4]]))
    results=[]
    for n in (0,1<<31):
        # Leave the two last affine constants free: recover the real formula.
        pins=[(0,0,9),(0,1,0xFD7046C5)] if joint else ()
        result=solve(row,n,20,pins,joint)
        assert result['status']=='proved_equivalent',result
        results.append(result)
    return results


def joint_manifest(per_order=32):
    """Sample four-op wirings with one SHR16 and one to three MADs."""
    rng=random.Random(8642815)
    rows=[];seen_orders=set()
    for mads in (1,2,3):
        for bits in product(('^','&','|'),repeat=3-mads):
            for order in sorted(set(permutations(('shr16',*('mad',)*mads,*bits)))):
                if order in seen_orders or order[-1]=='shr16':continue
                seen_orders.add(order)
                opts=[[(k,) for k in range(i+1)] if op=='shr16' else _instr_options(i,op)
                      for i,op in enumerate(order)]
                if order[0]=='mad':opts[0]=[(0,1,2),(0,0,1),(0,0,0)]
                selected=set();attempts=0
                while len(selected)<per_order and attempts<50000:
                    attempts+=1
                    w=tuple(rng.choice(oo) for oo in opts)
                    if all(any(j+1 in w[i] for i in range(j+1,4)) for j in range(3)):
                        selected.add(w)
                rows.extend(dict(order=list(order),wiring=[list(w) for w in wiring]) for wiring in sorted(selected))
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--node',type=lambda s:int(s,0),default=0x12345678 ^ 0xB55A4F09)
    ap.add_argument('--seconds',type=float,default=1)
    ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--refine-from',type=Path)
    ap.add_argument('--joint',action='store_true')
    ap.add_argument('--per-order',type=int,default=32)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    node=args.node & MASK
    rows=joint_manifest(args.per_order) if args.joint else manifest()
    if args.refine_from:
        import hashlib,json
        data=args.refine_from.read_bytes()
        previous=json.loads(data)
        assert previous['completed']==len(previous['results'])==len(previous['manifest'])
        node=previous['node']
        args.joint=previous.get('joint_last_mad',False)
        rows=[{k:r[k] for k in ('order','wiring')} for r in previous['results']
              if r['status'] not in ('unsat','proved_equivalent')]
        source=dict(source=str(args.refine_from),source_sha256=hashlib.sha256(data).hexdigest())
    else:
        source={}
    report=dict(node=node,full32_semantics=True,prepared_constants_arbitrary=True,
                joint_last_mad=args.joint,sampled_not_exhaustive=args.joint,
                scope=('Sampled four-op programs with exactly one literal SHR16 and 1..3 MADs; '
                       'remaining ops XOR/AND/OR. ' if args.joint else
                       'Three ops: one literal SHR16 and MAD plus MAD/XOR/AND/OR. ')+
                      'Single input; no global kernel or all-ISA optimality claim.',
                seconds_per_case=args.seconds,workers=args.workers,
                manifest=rows,controls=controls(args.joint),results=[],completed=0,
                best_executable_cycles=864,new_executable=False,**source)
    counts=Counter()
    start=time.monotonic()
    print(f'{len(rows)} cases, node={node:#x}',flush=True)
    write_report(args.output,report)
    with ProcessPoolExecutor(args.workers) as pool:
        futures=[pool.submit(job,(row,node,args.seconds,(),args.joint)) for row in rows]
        for done,f in enumerate(as_completed(futures),1):
            result=f.result()
            report['results'].append(result)
            counts[result['status']]+=1
            if result['status']=='proved_equivalent':
                print(result,flush=True)
            if done%50==0 or done==len(rows):
                report.update(completed=done,counts=dict(counts),seconds=round(time.monotonic()-start,3))
                write_report(args.output,report)
                print(done,dict(counts),report['seconds'],flush=True)


if __name__=='__main__':
    main()
