"""Joint last-MAD / fold / shared-node / next-MAD affine candidates.

Original cost is five: t=9*x+C4, q=t>>16, u=t^q, v=u^N,
out=4097*v+C0. Three concrete four-op families let every coefficient
depend arbitrarily on the prepared root node N. All checks are full32.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import z3

from codex.search_root_boundary863 import MASK,C0,target as boundary_target


C4=0xFD7046C5
FAMILIES=('fold_affine','raw_xor','separate_affine')


def body(x,c,family):
    u=(c['A']*x+c['B']) & MASK
    source=x if family=='separate_affine' else u
    q=z3.LShR(source,16) if z3.is_bv(source) else source>>16
    v=(x if family=='raw_xor' else u) ^ q
    return (c['G']*v+c['H']) & MASK


def target(x,node):
    return boundary_target((9*x+C4)&MASK,node)


def solve(family,node,seconds,control=False,pin=False):
    c=dict(zip(('A','B','G','H'),z3.BitVecs('A B G H',32)))
    s=z3.SolverFor('QF_BV')
    s.add(c['G'] & 1 == 1)
    if family=='fold_affine':
        # Here the first affine result is a genuine one-word cut.
        s.add(c['A'] & 1 == 1)
    witness=dict(A=9,B=C4,G=4097,H=C0)
    if pin:
        assert control and family=='fold_affine'
        s.add(*(c[k]==v for k,v in witness.items()))
    def want(x):
        return body(x,witness,family) if control else target(x,node)
    rng=random.Random(8642812)
    points=list(dict.fromkeys([*range(16),MASK,MASK-1,0xffff,0x10000,0xffff0000,
                              *(1<<k for k in range(32)),
                              *(rng.getrandbits(32) for _ in range(16))]))
    def add(p):
        x=z3.BitVecVal(p,32)
        s.add(body(x,c,family)==want(x))
    start=time.monotonic()
    for p in points:add(p)
    xs=np.array(points+[rng.getrandbits(32) for _ in range(2048)],dtype=np.uint64)
    expected=want(xs)
    report=dict(family=family,node=node,full32_semantics=True,control=control,
                pinned=pin,status='unknown',events=[],new_executable=False)
    while time.monotonic()-start<seconds:
        s.set(timeout=max(1,int((seconds-time.monotonic()+start)*1000)))
        status=s.check()
        report['events'].append(dict(samples=len(points),status=str(status)))
        if status!=z3.sat:
            report['status']=str(status)
            if status==z3.unknown:report['reason']=s.reason_unknown()
            break
        m=s.model()
        cv={k:m.eval(v,model_completion=True).as_long() for k,v in c.items()}
        bad=np.flatnonzero(body(xs,cv,family)!=expected)
        if not len(bad):
            x=z3.BitVec('affine_boundary_counterexample',32)
            check=z3.SolverFor('QF_BV')
            check.set(timeout=max(1,int((seconds-time.monotonic()+start)*1000)))
            check.add(body(x,cv,family)!=want(x))
            status=check.check()
            if status==z3.unsat:
                report.update(status='proved_equivalent',coefficients=cv)
                break
            if status==z3.unknown:
                report.update(status='verification_unknown',reason=check.reason_unknown())
                break
            bad_points=[check.model().eval(x,model_completion=True).as_long()]
        else:
            bad_points=[int(xs[i]) for i in bad[:8]]
        for p in bad_points:
            assert p not in points
            points.append(p);add(p)
    report.update(seconds=round(time.monotonic()-start,3),samples=len(points))
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--family',choices=FAMILIES,required=True)
    ap.add_argument('--node',type=lambda s:int(s,0),default=0x12345678 ^ 0xB55A4F09)
    ap.add_argument('--seconds',type=float,default=60)
    ap.add_argument('--control',action='store_true')
    ap.add_argument('--pin',action='store_true')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result=solve(args.family,args.node & MASK,args.seconds,args.control,args.pin)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
