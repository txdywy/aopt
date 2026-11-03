"""Sample four-op joint windows with one arbitrary prepared DIV/REM/CDIV.

The eight initial words exclude e. Divisors are full32 root-only constants,
not restricted to powers of two. A fixed-root formula is not a kernel.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import gzip
import hashlib
import json
from pathlib import Path
import time

from codex.scout_joint_wirings863 import write_report
from codex.search_focused_joint_boundary863 import edge_points,manifest
from codex.search_joint_live_boundary863 import C0,C4,NAMES,NODE,solve


def division_row(row,op):
    assert op in ('//','%','cdiv') and row['order'].count('shr16')==1
    return dict(order=[op if x=='shr16' else x for x in row['order']],wiring=[list(w) for w in row['wiring']])


def control(encoding='native'):
    row=dict(order=['mad','//','^','mad'],wiring=[[0,8,9],[8],[8,9],[10,11,12]])
    pins=[(0,0,9),(0,1,C4),(1,0,65536),(3,0,4097),(3,1,C0)]
    result=solve(row,0,15,pins,True,edge_points(0),encoding)
    assert result['status']=='proved_equivalent',result
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--per-order',type=int,default=4)
    ap.add_argument('--seed',type=int,default=8642935)
    ap.add_argument('--seconds',type=float,default=2)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--refine-from',type=Path)
    ap.add_argument('--encoding',choices=('native','product'),default='native')
    ap.add_argument('--divisors',default='',help='Optional comma-separated concrete divisors; these do not close the full divisor domain.')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();source={};started=time.monotonic()
    if args.refine_from:
        raw=args.refine_from.read_bytes();decoded=gzip.decompress(raw) if args.refine_from.suffix=='.gz' else raw
        previous=json.loads(decoded)
        assert previous['completed']==len(previous['manifest'])==len(previous['results'])
        closed={'unsat','division_wordcut_excluded','lsb_dependency_excluded','one_high_tail_excluded','polynomial_quotient_excluded','quotient_pair_excluded','proved_equivalent'}
        rows=[{k:r[k] for k in ('order','wiring','divisor_pin') if k in r} for r in previous['results'] if r['status'] not in closed]
        orders=previous['orders'];args.seed=previous['seed'];args.per_order=previous['per_order']
        source=dict(source=str(args.refine_from),source_sha256=hashlib.sha256(raw).hexdigest())
    else:
        old,orders=manifest(args.per_order,args.seed)
        rows=[division_row(r,op) for r in old for op in ('//','%','cdiv')]
    divisors=[int(v,0) for v in args.divisors.split(',') if v]
    assert all(1<=d<=0xffffffff for d in divisors)
    if divisors:
        rows=[dict(row,divisor_pin=d) for row in rows for d in ([row['divisor_pin']] if 'divisor_pin' in row else divisors)]
    extra=edge_points()
    report=dict(node=NODE,input_names=NAMES,free_e=False,sampled_not_exhaustive=True,
                no_complete_four_op_grammar_claim=True,prepared_divisor_domains={'//':[2,0xffffffff],'%':'odd3..MASK','cdiv':[2,0xffffffff]},
                excluded_divisor_proof='Divisor1 is T-only; even REM preserves numerator bit0. Both fail target variation under b16 within each initial parity class.',
                seed=args.seed,per_order=args.per_order,seconds_per_case=args.seconds,
                pinned_divisors=sorted({r['divisor_pin'] for r in rows if 'divisor_pin' in r}),
                edge_primary_points=extra,manifest=rows,orders=orders,division_encoding=args.encoding,
                positive_control=control(args.encoding),results=[],completed=0,counts={},
                best_executable_cycles=864,new_executable=False,**source)
    write_report(args.output,report);print('prepared-division cases',len(rows),flush=True)
    counts=Counter();checkpoint=min(50,max(1,len(rows)//10))
    with ProcessPoolExecutor(args.workers) as pool:
        futures=[pool.submit(solve,r,NODE,args.seconds,
                             [(next(i for i,op in enumerate(r['order']) if op in ('//','%','cdiv')),0,r['divisor_pin'])]
                             if 'divisor_pin' in r else (),True,extra,args.encoding) for r in rows]
        for done,future in enumerate(as_completed(futures),1):
            result=future.result();report['results'].append(result);counts[result['status']]+=1
            if result['status']=='proved_equivalent':print(result,flush=True)
            if done%checkpoint==0 or done==len(rows):
                report.update(completed=done,counts=dict(counts),seconds=round(time.monotonic()-started,3))
                write_report(args.output,report);print(done,dict(counts),report['seconds'],flush=True)


if __name__=='__main__':main()
