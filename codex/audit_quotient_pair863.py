"""Permutation sums exclude prepared quotient/word affine pairs.

P can be any T-only prefix of the nine live inputs, but must be a true
value cut. The last MAD is G*P+q or P+G*q, q=op(P,D). Fixed root only.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import struct

from codex.audit_denominator_products863 import linear_class
from codex.audit_one_high_tail863 import information_fibre
from codex.screen_middle_division_residual863 import pool
from codex.search_early_boundary863 import MASK,NODE,early_inputs,wanted,legal_point
from codex.search_live_boundary863 import primary_path_avoiding


def sum_scan(width=32):
    modulus=1<<width;mask=modulus-1;result={}
    for op,numerator in (('floor',modulus),('ceil',mask)):
        out={0:[],modulus//2:[]};lo=1;intervals=0;digest=hashlib.sha256()
        while lo<=mask:
            k=numerator//lo;hi=min(mask,numerator//k)
            assert numerator//hi==k
            a=k*(k+1)//2;b=0 if op=='floor' else(k+1)*mask
            for target in out:
                cls=linear_class(a,b-target,width)
                if cls is not None:
                    base,step=cls;first=lo+(base-lo)%step
                    if first<=hi:out[target].extend(range(first,hi+1,step))
            digest.update(struct.pack('<QQQ',lo,hi,k));intervals+=1;lo=hi+1
        result[op]=dict(covered_divisors=mask,exact_quotient_intervals=intervals,
                        interval_sha256=digest.hexdigest(),sum_zero=out[0],sum_half=out[modulus//2])
    return result


@lru_cache(maxsize=1)
def audit():
    sums=sum_scan();half=1<<31
    assert sums['floor']['sum_zero']==[]
    assert sums['floor']['sum_half']==[1<<i for i in range(32)]
    assert sums['ceil']['sum_zero']==[2,MASK-1]
    assert sums['ceil']['sum_half']==[1,half-1]
    points=[b for b in pool() if b&((1<<19)|1)==0];anchor=points[0]
    va=early_inputs(anchor);ya=wanted(va);bits=[];legal={anchor}
    for bit in range(32):
        b=next(b for b in points if ((wanted(early_inputs(b))^ya)>>bit)&1)
        v=early_inputs(b);assert [x&1 for x in v]==[x&1 for x in va]
        bits.append(dict(bit=bit,primary_b=[anchor,b],targets=[ya,wanted(v)]));legal.add(b)
    predicates=[]
    functions=[('ceil_G_P_plus_q',lambda y:int(0<y<=half)),
               ('ceil_P_plus_G_q',lambda y:(y^(y>>31))&1)]
    for name,f in functions:
        b=next(b for b in points if f(wanted(early_inputs(b)))!=f(ya))
        v=early_inputs(b);assert [x&1 for x in v]==[x&1 for x in va]
        predicates.append(dict(family=name,primary_b=[anchor,b],targets=[ya,wanted(v)],predicate=[f(ya),f(wanted(v))]));legal.add(b)
    info=information_fibre();assert info['prepared_ceil_divisor_max']<half-1
    return dict(node=NODE,raw_root=0x12345678,complete_in_stated_scope=True,
                scope='P is a T-only prefix on the nine inputs and a true value cut. The output is G*P+op(P,D) or P+G*op(P,D), with prepared uint32 G/D; op is floor,ceil,REM or prepared SHR. No arbitrary third initial/computed addend or runtime coefficient is included.',
                sums=sums,information_fibre=info,bit_witnesses=bits,ceil_predicate_witnesses=predicates,
                floor_G_P_plus_q='Permutation sums require even G and a power-of-two D, unless D1 which is T-only. Put Q=2^32/D. After multiplication by (G*D+1)^-1, each remainder r produces an interval of length Q. A disjoint tiling containing[0,Q-1] forces the interval starts onto the Q grid, hence G=Q*t with odd t. Output bit log2(Q) equals P bit0, contradicting the corresponding target witness.',
                floor_P_plus_G_q='The target bit0 requires odd G. Permutation sums then require sum floor(P/D)=0 modulo2^32. The complete divisor scan finds none.',
                ceil_G_P_plus_q='Even G requires the ceil sum2^31, leaving D1 or2^31-1; the first is T-only and the second exceeds the information bound. Odd G requires ceil sum0, leaving D2 or2^32-2; the bound leaves D2. Normalizing the two half-length arithmetic intervals forces G=2^31-1. Then P bit0 is the indicator0<Y<=2^31, contradicted by the predicate witness.',
                ceil_P_plus_G_q='Odd G is necessary for target bit0. The sum0 scan and information bound leave D2. Two half-length intervals force G=2^31-1, and P bit0=Y bit0 XOR Y bit31. The predicate witness contradicts this.',
                remainder='Even D cannot transmit target bit0, so D is odd>1. For P+G*REM, G must be odd. Put h=gcd(G+1,2^32)>=2: modulo h the output is an odd multiple of floor(P/D). Uniform output counts force D to divide2^32/h by choosing a quotient residue class without the last partial block, impossible for odd D>1. For G*P+REM, the same argument works if G is odd. If G is even, divide by odd G+1 to obtain P plus an odd multiple of floor(P/D), which fails the floor sum0 condition.',
                prefix_bit0='Every T-only prefix has the same bit0 on two inputs whose nine initial bit0 values agree. The target witnesses violate the necessary inverse bit0 relations above.',
                legal_points=[legal_point(b) for b in sorted(legal)],new_executable=False,best_executable_cycles=864,no_global_optimality_claim=True)


@lru_cache(maxsize=1)
def checked():
    path=Path('codex/quotient_pair863_audit.json')
    if not path.exists():return None
    raw=path.read_bytes();assert json.loads(raw)==json.loads(json.dumps(audit()))
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())


def applicable(row,node):
    if node!=NODE or row['order'][-1]!='mad':return None
    special=[i for i,op in enumerate(row['order']) if op in ('shr16','shr','//','%','cdiv')]
    if len(special)!=1:return None
    if any(op not in ('mad','^','&','|') for i,op in enumerate(row['order']) if i!=special[0]):return None
    i=special[0];q=8+i;p=row['wiring'][i][0];last=8+len(row['order'])-1
    if p>=q:return None
    a,b,c=row['wiring'][-1];family=None
    if c==q and (a==p and b>=last or b==p and a>=last):family='G_P_plus_q'
    if c==p and (a==q and b>=last or b==q and a>=last):family='P_plus_G_q'
    if family is None or primary_path_avoiding(row,8,False,p):return None
    proof=checked()
    return None if proof is None else dict(**proof,family=family,value_cut=p,division=row['order'][i])


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();d=audit();args.output.write_text(json.dumps(d,indent=2)+'\n')
    print('Paired quotient/cut affine tails excluded; complete sum scans:',d['sums']['floor']['covered_divisors'],d['sums']['ceil']['covered_divisors'],flush=True)


if __name__=='__main__':main()
