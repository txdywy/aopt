"""Bit-separated rectangle screens for free-key live-input middle programs.

On the legal low-a subdomain, a and prior are known functions. A freely
keyed word a XOR K does not change their axis flip masks. When all source
bits separate between the two axes, every bitwise prefix and subsequent
integer polynomial satisfies the mixed-difference divisibility condition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from codex.search_keyed_live_middle863 import C0,reference


def rectangles(width):
    size,mask=1 << width,(1 << width)-1
    inverse=pow(4097,-1,size)
    for base in range(size):
        for i in range(width):
            for j in range(i+1,width):
                if i+j<width:continue
                h,k=1 << i,1 << j
                if base+h+k>=size:continue
                aa=[base+d for d in (0,h,k,h+k)]
                pp=[((a-C0)*inverse)&mask for a in aa]
                if any(v[0]^v[1]^v[2]^v[3]for v in(aa,pp)):continue
                hm=(aa[0]^aa[1])|(pp[0]^pp[1])
                km=(aa[0]^aa[2])|(pp[0]^pp[2])
                if hm&km or hm&(h-1) or km&(k-1):continue
                yield dict(base=base,h=h,k=k,divisor=min(h*k,size),
                    a_values=aa,prior_values=pp,h_changed_bits=hm,k_changed_bits=km)


def search(width=11,bypass=None,control=False):
    if not 2<=width<=13 or bypass not in(None,'a','prior'):
        raise ValueError('width 2..13 and no bypass, a, or prior required')
    size,mask=1 << width,(1 << width)-1
    keys=np.arange(size,dtype=np.int64)
    inverse=pow(4097,-1,size)

    def target(a):
        p=((a-C0)*inverse)&mask
        if control:
            result=(a*p+17*((a^42)^p)+13)&mask
            if bypass:result ^= a if bypass=='a'else p
            return result
        return reference(a,width)&mask

    history=[]
    started=time.monotonic()
    for rect in rectangles(width):
        aa,pp=rect['a_values'],rect['prior_values']
        values=[target(a)^keys^(a if bypass=='a'else p if bypass=='prior'else 0)
                for a,p in zip(aa,pp)]
        delta=values[3]-values[1]-values[2]+values[0]
        keep=(delta&(rect['divisor']-1))==0
        before=len(keys);keys=keys[keep]
        if len(keys)!=before or not history:
            history.append(dict(rect,before=before,after=len(keys)))
        if not len(keys):break
    return dict(width=width,bypass=bypass,control=control,free_input_key='all K; axis masks are invariant under a XOR K',
        initial_output_keys=size,remaining_output_keys=len(keys),rectangles=history,
        survivor_examples=keys[:16].tolist(),zero_control_retained=bool(np.any(keys==0))if control else None,
        result='all_output_keys_excluded'if not len(keys)else'necessary_conditions_survive',
        seconds=round(time.monotonic()-started,3),new_executable=False,
        scope='Arbitrary bitwise prefixes over a, prior, a XOR K, then integer polynomials; '
              'optional final XOR with a or prior. Restricted low-a full32 subdomain.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result=dict(experiments=[search(bypass=b)for b in(None,'a','prior')],
        controls=[search(6,b,True)for b in(None,'a','prior')])
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([dict(bypass=r['bypass'],result=r['result'],remaining=r['remaining_output_keys'],seconds=r['seconds'])for r in result['experiments']]))


if __name__=='__main__':main()
