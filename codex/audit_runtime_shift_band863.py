"""A global count band for a sole dynamic SHR between T-only computations.

No free e is assumed. Prefixes and suffixes may have arbitrary length and
retain every initial word; there must be exactly one non-T operation, SHR.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import z3

from codex.search_early_boundary863 import legal_point
from codex.search_joint_live_boundary863 import inputs, wanted, NODE, NAMES, MASK, C0


def fibre_counts():
    high=(np.arange(64,dtype=np.uint64)<<19)[:,None]
    mid=(np.arange(1024,dtype=np.uint64)<<9)[None,:]
    counts=[]
    for low in range(512):
        b=high+mid+low
        # All operations before extraction are T-functions. Retaining full32
        # here also makes the independent frozen/SMT comparisons direct.
        u=(33*b+0xe9f8cc1d)&MASK
        v=(16896*b+0xaccf6200)&MASK
        e=(9*(u^v)+0xfd7046c5)&MASK
        # Extract the actual target, including the root XOR and final MAD.
        # Low9 extraction commutes with all preceding uint32 reductions.
        tags=((4097*(e^(e>>16)^NODE)+C0)&511).astype(np.int64)
        seen=np.zeros((64,512),dtype=bool)
        np.put_along_axis(seen,tags,True,axis=1)
        counts.extend(map(int,seen.sum(axis=1)))
    minimum=min(counts);index=counts.index(minimum)
    return dict(complete_cases=len(counts),primary_inputs=len(counts)*1024,
                low9_cases=512,high19_through24_cases=64,middle9_through18_cases=1024,
                minimum_distinct=minimum,maximum_distinct=max(counts),
                histogram=dict(sorted(Counter(counts).items())),
                minimum_signature=dict(low9=index//64,high19_through24=index%64),
                count_table_sha256=hashlib.sha256(np.array(counts,dtype='<u2').tobytes()).hexdigest())


@lru_cache(maxsize=1)
def audit():
    b=z3.BitVec('shift_band_primary_b',32)
    i,j=z3.BitVecs('shift_band_i shift_band_j',3)
    base=b & ~(7<<16)
    bs=[base|(z3.ZeroExt(29,x)<<16) for x in (i,j)]
    values=[inputs(x) for x in bs]
    s=z3.SolverFor('QF_BV');s.set(timeout=5000)
    s.add(z3.Or(*((x&65535)!=(y&65535) for x,y in zip(*values))))
    assert s.check()==z3.unsat
    s=z3.SolverFor('QF_BV');s.set(timeout=5000)
    s.add(i!=j,(wanted(values[0],NODE)&7)==(wanted(values[1],NODE)&7))
    assert s.check()==z3.unsat
    mask=z3.BitVec('shift_band_middle_mask',32)
    s=z3.SolverFor('QF_BV');s.set(timeout=5000)
    s.add(mask & ~(1023<<9)==0,
          z3.Or(*((x&511)!=(y&511) for x,y in zip(inputs(b),inputs(b^mask)))))
    assert s.check()==z3.unsat
    s=z3.SolverFor('QF_BV');s.set(timeout=5000)
    s.add((wanted(inputs(b),NODE)&511)!=(wanted(inputs(b&0x1ffffff),NODE)&511))
    assert s.check()==z3.unsat
    counts=fibre_counts()
    assert counts['complete_cases']==32768 and counts['minimum_distinct']==387
    assert counts['maximum_distinct']==483 and counts['minimum_distinct']>256
    assert counts['count_table_sha256']=='5510056a6dfb3eac428e5b96c3e052c84acd39d41bc009fa0c6e814a09791939'
    lo=counts['minimum_signature']['low9'];hi=counts['minimum_signature']['high19_through24']
    witness=[lo+(hi<<19)+(k<<9) for k in (0,1,255,511,1023)]
    return dict(node=NODE,input_names=list(NAMES),free_e=False,
                scope='Exactly one SHR; its numerator/count are T-only expressions in the eight stated '
                'initials/root-only constants. All other computation is T-only, with all bypasses retained. '
                'No restriction on prefix/suffix length. Fixed allowed root0x12345678.',
                initial_low16_fibre_invariant=True,target_low3_fibre_permutation=True,
                initial_low9_fibre_invariant=True,target_low9_ignores_primary_bits25_through31=True,
                universal_fibres=counts,count_low16_minimum=16,count_low16_maximum=23,
                global_quotient_maximum=65535,prepared_numerator_impossible=True,
                first_band_proof='Vary b16..18 with other bits fixed. Every initial/T-prefix low16 '
                'is fixed, while target low3 has8 different values. If count low16=c<16, q low3 '
                'has at most5 values including clamped0. If c30/31, it has at most4/2; c>=32 '
                'gives only0. Hence c16..29 everywhere. Counts can still be>=65536.',
                second_band_proof='Fix b low9 and bits>=19, vary b9..18. All initial/T-prefix low9 '
                'are fixed. Because the first band lies below512, it makes c=count low16 '
                'constant on this fibre. The exhaustive32768 signature table proves the target '
                'has at least387 low9 values for EVERY signature. If c24..29, q is either '
                'numerator>>c or0, with at most256 values, a contradiction. Thus c16..23 everywhere, '
                'and the actual shift always has count>=16, even on clamped branches.',
                prepared_numerator_proof='For a root-only numerator C, each low9 fibre has a single '
                'c16..23, so q is either C>>c or0: at most2 low9 values, contradicting>=387.',
                legal_example_points=[legal_point(x) for x in witness],
                new_executable=False,best_executable_cycles=864,no_global_optimality_claim=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();result=audit()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Dynamic SHR count low16 is always16..23; q<=65535; all32768 fibres checked.',flush=True)


if __name__=='__main__':main()
