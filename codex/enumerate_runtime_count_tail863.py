"""Exact paired-rectangle test for Y=q*h+c with q at most sixteen bits.

The vector implementation solves for h's high half. The independent scalar
implementation solves the full-word product congruence directly.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from codex.search_joint_live_boundary863 import inputs, wanted, NODE


@lru_cache(maxsize=4)
def inverses(width):
    half=1<<width
    table=np.zeros(half,dtype=np.uint64)
    table[1::2]=[pow(h,-1,half) for h in range(1,half,2)]
    return table


def vector_scan(targets,rectangles,width=16,counts=range(16,24),return_models=False):
    half=1<<width;mask=half*half-1;inv=inverses(width)
    h=np.arange(1,half,2,dtype=np.uint64)[None,:]
    ih=inv[h];ys=np.array(targets,dtype=np.uint64)[:,None]
    assert len(targets)==8
    results=[];models=[]
    for c in counts:
        q=(ih*((ys&(half-1))+half-c))&(half-1)
        gcd=np.where(q==0,half,q&((half-q)&(half-1)))
        mod=half//gcd
        with np.errstate(over='ignore'):rhs=((ys-np.uint64(c)-q*h)&mask)>>width
        ok=np.all((rhs&(gcd-1))==0,axis=0)
        high=((rhs//gcd)*inv[q//gcd])&(mod-1)
        stages=[int(ok.sum())]
        for order in rectangles:
            delta=(high[order[3]]+high[order[0]]+2*half-high[order[1]]-high[order[2]])
            modulus=np.min(mod[list(order)],axis=0)
            ok&=(delta&(modulus-1))==0
            stages.append(int(ok.sum()))
        results.append(dict(count=int(c),survivors_by_stage=stages))
        if return_models:models.extend((int(x),int(c)) for x in h[0,ok])
    return (results,models) if return_models else results


@lru_cache(maxsize=4)
def scalar_product_inverses(width):
    word=1<<(2*width);half=1<<width
    return [None]+[(q&-q,word//(q&-q),pow(q//(q&-q),-1,word//(q&-q))) for q in range(1,half)]


def scalar_scan(targets,rectangles,width=16,counts=range(16,24)):
    half=1<<width;word=half*half;table=scalar_product_inverses(width)
    inverse_h=[pow(h,-1,half) for h in range(1,half,2)];results=[]
    for c in counts:
        stages=[0]*(1+len(rectangles))
        for h,ih in zip(range(1,half,2),inverse_h):
            representatives=[];moduli=[]
            for y in targets:
                q=((y-c)*ih)&(half-1);rhs=(y-c)&(word-1)
                if q==0:
                    if rhs:break
                    representatives.append(h);moduli.append(half)
                    continue
                gcd,mod,iq=table[q]
                if rhs%gcd:break
                representative=(rhs//gcd*iq)%mod
                if representative&(half-1)!=h:break
                representatives.append(representative);moduli.append(mod)
            else:
                stages[0]+=1
                for stage,order in enumerate(rectangles,1):
                    delta=(representatives[order[3]]+representatives[order[0]]-
                           representatives[order[1]]-representatives[order[2]])
                    if delta%min(moduli[i] for i in order):break
                    stages[stage]+=1
        results.append(dict(count=int(c),survivors_by_stage=stages))
    return results


def audit(rectangle_path):
    encoded=rectangle_path.read_bytes();record=json.loads(encoded)
    frames=list(dict.fromkeys((p[0],tuple(p[1]),tuple(p[2])) for p in record['proofs'].values()))
    started=time.monotonic();results=[]
    for index,(base,left,right) in enumerate(frames,1):
        targets=[wanted(inputs(base|(k<<16)),NODE) for k in range(8)]
        assert all(not 16<=(y&65535)<=23 for y in targets)
        vector=vector_scan(targets,(left,right))
        scalar=scalar_scan(targets,(left,right))
        assert vector==scalar,(base,left,right,vector,scalar)
        assert all(r['survivors_by_stage'][-1]==0 for r in vector)
        results.append(dict(base=base,rectangles=[list(left),list(right)],targets=targets,
                            scan=vector,scalar_full_word_scan_agrees=True))
        print('inverse-count frame',index,'/',len(frames),'closed',flush=True)
    return dict(node=NODE,half_width=16,odd_low_half_parameters=32768,count_parameters=list(range(16,24)),
        distinct_frames=len(frames),evaluated_parameters=len(frames)*32768*8,results=results,
        source=str(rectangle_path),source_sha256=hashlib.sha256(encoded).hexdigest(),
        scanner_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        vector_high_half_and_scalar_full_word_implementations_agree=True,
        proof='If count clamps, q=0 and Y=count, but all eight target low16 words avoid16..23. '
        'Thus every point is active and shares c in16..23. Common low16 h=H must be odd, '
        'since target low3 has eight values. Each q is uniquely (Ylow16-c)*H^-1 mod2^16. '
        'Write h=H+2^16*t. The full product equation gives q*t=(Y-c-q*H)/2^16 mod2^16. '
        'Its solution is a residue class modulo2^16/gcd(q,2^16). A rectangle with Delta(h)=0 '
        'is feasible iff the signed residue sum vanishes modulo the gcd of its four moduli. '
        'Exhaust every odd H and all eight c. All frames have zero survivors. An independent '
        'scalar implementation solves q*h=Y-c modulo2^32, then checks h low16 and both rectangles.',
        complete=True,seconds=round(time.monotonic()-started,4),remaining_models=0,
        new_executable=False,best_executable_cycles=864,no_global_optimality_claim=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rectangles',type=Path,default=Path('codex/runtime_count_tail863_rectangles_20261029.json'))
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();result=audit(args.rectangles)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
