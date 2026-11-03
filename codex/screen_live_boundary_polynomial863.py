"""Necessary polynomial screen for live boundary programs with two MADs.

When SHR16 reads an initial e/z/u/v/b word, its output q is concrete on
each full32 input. All other initial words are affine in b and z. Two
arbitrary MADs therefore lie in the 35-coefficient degree-four space in
b,z,q, regardless of their order. UNKNOWN graphs covered by an inconsistent
system can be closed without another nonlinear synthesis query.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import random

from codex.search_live_boundary863 import inputs,target,MASK
from codex.screen_xor_mad_polynomial import consistent


def screen():
    rng=random.Random(8642820)
    positions=(0,1,2,3,4,8,12,15,16,17,20,28,31)
    xs=sorted(set([*range(64),MASK,MASK-1,0xffff,0xffff0000,
                   *(1<<i for i in range(32)),
                   *((1<<a)|(1<<b) for a,b in combinations(positions,2)),
                   *(rng.getrandbits(32) for _ in range(64))]))
    node=0xA76E1971
    initial=[inputs(x,node,'arms',False) for x in xs]
    powers=[(i,j,k) for i in range(5) for j in range(5-i) for k in range(5-i-j)]
    assert len(powers)==35
    rhs=[target(v[0],node) for v in initial]
    rows=[]
    for source,name in enumerate(('e','z','u','v','b')):
        qq=[v[source]>>16 for v in initial]
        matrix=[[(x**i*v[1]**j*q**k)&MASK for i,j,k in powers]
                for x,v,q in zip(xs,initial,qq)]
        compatible=consistent(matrix,rhs,32)
        control=[(((x*q+v[1])&MASK)**2+x)&MASK for x,v,q in zip(xs,initial,qq)]
        assert consistent(matrix,control,32)
        rows.append(dict(source_index=source,source_name=name,
                         status='relaxed_survivor' if compatible else 'inconsistent_full32',
                         positive_two_mad_control=True))
    return dict(node=node,width=32,samples=xs,monomial_powers=powers,
                arbitrary_independent_coefficients=35,results=rows,
                all_sources_excluded=all(r['status']=='inconsistent_full32' for r in rows),
                scope='One SHR16 reads an initial live word; the other two instructions are MADs. '
                      'Does not cover shifts of newly synthesized MAD results.',
                best_executable_cycles=864,new_executable=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();r=screen()
    args.output.write_text(json.dumps(r,indent=2)+'\n')
    print(r['all_sources_excluded'],r['results'])


if __name__=='__main__':main()
