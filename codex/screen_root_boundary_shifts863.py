"""Full32 polynomial screen for two literal shifts followed by one MAD.

Both shift directions, independent and chained sources, and every uint32
literal count are covered (>=32 has representative 32). A MAD on x,q,r and
arbitrary constants is contained in the ten-coefficient quadratic space.
The screen evaluates real shifts before forming congruences; it never
narrows the word before a right shift. SAT in this relaxation is not code.
"""
from __future__ import annotations

import argparse
from itertools import combinations, product
import json
from pathlib import Path
import random
import time

from codex.screen_xor_mad_polynomial import consistent
from codex.scout_joint_wirings863 import write_report


MASK, C0 = 0xFFFFFFFF, 0x7ED55D16


def shift(x, direction, count):
    return ((x << count) & MASK) if direction == '<<' else x >> count


def basis(x, q, r):
    return [1,x,q,r,x*x & MASK,x*q & MASK,x*r & MASK,q*q & MASK,q*r & MASK,r*r & MASK]


def sample_inputs():
    rng=random.Random(8642811)
    positions=(0,1,2,3,4,8,12,15,16,17,20,28,31)
    return sorted(set([*range(32),MASK,MASK-1,0xffff,0xffff0000,
                       *(1<<i for i in range(32)),
                       *((1<<a)|(1<<b) for a,b in combinations(positions,2)),
                       *(rng.getrandbits(32) for _ in range(24))]))


def target(x, node):
    return (4097*(x ^ (x>>16) ^ node)+C0) & MASK


def canonical_programs():
    """Exact deduplication: constant shifts are GF(2)-linear on word bits."""
    representatives={}
    members={}
    for d,e,a,b,chained in product(('<<','>>'),('<<','>>'),range(33),range(33),(False,True)):
        qbits=tuple(shift(1<<i,d,a) for i in range(32))
        rbits=tuple(shift(qbits[i] if chained else 1<<i,e,b) for i in range(32))
        key=tuple(sorted((qbits,rbits)))
        program=dict(first=[d,a],second=[e,b],chained=chained)
        representatives.setdefault(key,program)
        members[key]=members.get(key,0)+1
    return [(p,members[k]) for k,p in representatives.items()]


def run(node):
    xs=sample_inputs()
    want=[target(x,node) for x in xs]
    representatives=canonical_programs()
    rows=[]
    started=time.monotonic()
    for program,count in representatives:
        d,a=program['first'];e,b=program['second']
        qs=[shift(x,d,a) for x in xs]
        rs=[shift(q if program['chained'] else x,e,b) for x,q in zip(xs,qs)]
        matrix=[basis(x,q,r) for x,q,r in zip(xs,qs,rs)]
        compatible=consistent(matrix,want,32)
        # Actual one-MAD output, including both prepared shift words.
        positive=[(x*r+q) & MASK for x,q,r in zip(xs,qs,rs)]
        assert consistent(matrix,positive,32)
        rows.append(dict(program,multiplicity=count,
                         status='relaxed_survivor' if compatible else 'inconsistent_full32',
                         positive_control=True))
    return dict(node=node,width=32,samples=xs,independent_coefficients=10,
                represented_shift_pairs=sum(c for _,c in representatives),
                deduplicated_pairs=len(representatives),results=rows,
                excluded=sum(r['status']=='inconsistent_full32' for r in rows),
                survivors=[r for r in rows if r['status']=='relaxed_survivor'],
                seconds=round(time.monotonic()-started,3),
                scope='Two literal shifts (either direction) plus one MAD; arbitrary constants; '
                      'all uint32 input words. No variable shifts or other live words.',
                new_executable=False,best_executable_cycles=864)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--node',type=lambda s:int(s,0),default=0x12345678 ^ 0xB55A4F09)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result=run(args.node & MASK)
    write_report(args.output,result)
    print(json.dumps({k:result[k] for k in ('represented_shift_pairs','deduplicated_pairs','excluded','seconds')}))
    print('survivors',len(result['survivors']))


if __name__=='__main__':
    main()
