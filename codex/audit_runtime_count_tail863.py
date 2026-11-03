"""Paired-rectangle certificates for three count-aware final-MAD families."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
from itertools import combinations
import json
from pathlib import Path
import random

import z3

from codex.audit_runtime_shift_band863 import audit as band_audit
from codex.enumerate_runtime_count_tail863 import vector_scan
from codex.search_early_boundary863 import legal_point
from codex.search_first_runtime_count863 import count_manifest, fibres
from codex.search_joint_live_boundary863 import inputs, wanted, NODE, MASK
from codex.search_runtime_count_prefix863 import count_word
from codex.search_runtime_count_tail863 import FAMILIES


RECTANGLES=Path('codex/runtime_count_tail863_rectangles_20261029.json')
SCOUT=Path('codex/runtime_count_tail863_scout_20261029.json.gz')
INVERSE=Path('codex/runtime_count_tail863_inverse_20261029.json')


def delta(values,order):
    return (values[order[3]]+values[order[0]]-values[order[1]]-values[order[2]])&MASK


def key(row):
    return json.dumps({k:row[k] for k in ('op','source','other','wiring','family') if k in row},sort_keys=True)


def find(output):
    """Deterministically find finite certificates, without querying any score."""
    rows=count_manifest();remaining=list(range(len(rows)));found={}
    rng=random.Random(8642987)
    bases=[f[0] for f in fibres()]+[rng.getrandbits(32)&~(7<<16) for _ in range(128)]
    partitions=[]
    for left in combinations(range(8),4):
        if 0 not in left:continue
        right=tuple(i for i in range(8) if i not in left)
        def orders(indices):
            return [[p[0],*[i for i in indices if i not in p],p[1]]
                    for p in combinations(indices,2) if indices[0] in p]
        partitions.append((orders(left),orders(right)))
    for base in bases:
        vv=[inputs(base|(k<<16)) for k in range(8)]
        yy=[wanted(v,NODE) for v in vv]
        def ok(order):
            d=delta(yy,order)
            return min(d,(-d)&MASK)>2*23*65535
        candidates=[(a,b) for aa,bb in partitions for a in aa if ok(a) for b in bb if ok(b)]
        pending=[]
        for index in remaining:
            row=rows[index]
            if row['op']=='mad':values=[[v[j] for v in vv] for j in range(8)]
            elif 'other' in row:values=[[count_word(row,[0,0],v) for v in vv]]
            else:values=[[(v[row['source']]>>bit)&1 for v in vv] for bit in range(32)]
            frame=next(((a,b) for a,b in candidates
                        if all(delta(xs,a)==delta(xs,b)==0 for xs in values)),None)
            if frame is None:pending.append(index)
            else:found[index]=(base,*frame)
        remaining=pending
        if not remaining:break
    assert not remaining,remaining
    output.write_text(json.dumps(dict(proofs=found,manifest=rows),indent=2)+'\n')


def audit():
    band=band_audit();encoded=RECTANGLES.read_bytes();rect=json.loads(encoded)
    rows=count_manifest()
    assert rect['manifest']==rows and set(rect['proofs'])==set(map(str,range(552)))
    record=json.loads(gzip.decompress(SCOUT.read_bytes()))
    assert record['completed']==len(record['results'])==1104
    assert record['h_manifest']==rows
    assert record['universal_count_band']==json.loads(json.dumps(band))
    assert {key(r) for r in record['results']}=={
        key(dict(r,family=f)) for r in rows for f in FAMILIES}
    assert Counter(r['status'] for r in record['results'])==dict(range_unsat=910,unknown=194)
    assert all(r['status']=='range_sat' for r in record['positive_controls'])
    checks=[];legal=set()
    for index,row in enumerate(rows):
        base,left,right=rect['proofs'][str(index)]
        assert base&((7<<16))==0
        assert set(left).isdisjoint(right) and set(left+right)==set(range(8))
        assert len(left)==len(set(left))==len(right)==len(set(right))==4
        bs=[base|(k<<16) for k in range(8)]
        vv=[inputs(b) for b in bs];yy=[wanted(v,NODE) for v in vv]
        assert all(len({v[j]&65535 for v in vv})==1 for j in range(8))
        assert len({y&7 for y in yy})==8
        if row['op']=='mad':
            assert all(delta([v[j] for v in vv],order)==0 for j in range(8) for order in (left,right))
            reason='common_low16_and_zero_initial_mixed_differences'
        elif 'other' in row:
            assert all(delta([count_word(row,[0,0],v) for v in vv],order)==0 for order in (left,right))
            reason='concrete_runtime_bitop_mixed_difference'
        else:
            assert all(delta([(v[row['source']]>>bit)&1 for v in vv],order)==0
                       for bit in range(32) for order in (left,right))
            reason='each_source_bit_has_zero_mixed_difference_for_all_root_masks'
        # Independent exact QF_BV replay over ALL root coefficients/masks.
        constants=z3.BitVecs(f'rtail_check_{index}_k0 rtail_check_{index}_k1',32)
        hh=[count_word(row,constants,v) for v in vv]
        solver=z3.SolverFor('QF_BV');solver.set(timeout=5000)
        solver.add(z3.Or(*(delta(hh,order)!=0 for order in (left,right))))
        assert solver.check()==z3.unsat,(index,row)
        distances=[]
        for order in (left,right):
            mixed=delta(yy,order);distance=min(mixed,(-mixed)&MASK)
            assert distance>2*23*65535
            distances.append(distance)
        legal.update(bs)
        checks.append(dict(row,base=base,rectangles=[left,right],target_mixed_distances=distances,
                           symbolic_root_parameter_check='unsat',certificate=reason))
    inverse=json.loads(INVERSE.read_bytes())
    assert inverse['complete'] and inverse['remaining_models']==0
    assert inverse['source_sha256']==hashlib.sha256(encoded).hexdigest()
    assert inverse['scanner_source_sha256']==hashlib.sha256(Path('codex/enumerate_runtime_count_tail863.py').read_bytes()).hexdigest()
    assert inverse['vector_high_half_and_scalar_full_word_implementations_agree']
    frames={(r['base'],tuple(r['rectangles'][0]),tuple(r['rectangles'][1])) for r in checks}
    assert frames=={(r['base'],tuple(r['rectangles'][0]),tuple(r['rectangles'][1])) for r in inverse['results']}
    assert len(frames)==inverse['distinct_frames']==18
    assert inverse['evaluated_parameters']==len(frames)*32768*8
    for row in inverse['results']:
        targets=[wanted(inputs(row['base']|(k<<16)),NODE) for k in range(8)]
        assert targets==row['targets']
        assert all(not 16<=(y&65535)<=23 for y in targets)
        assert row['scalar_full_word_scan_agrees']
        assert vector_scan(targets,row['rectangles'])==row['scan']
        assert all(r['survivors_by_stage'][-1]==0 for r in row['scan'])
    return dict(node=NODE,scope=__doc__+' The h bypass is one T instruction over eight stated '
        'initial words/root-only constants. Numerator/count may be arbitrary T-only expressions; '
        'there is exactly one SHR and no other non-T operation. Only q*count+h, count*h+q and q*h+count '
        'are covered. Fixed allowed root0x12345678; no full four-op or kernel lower bound.',
        first_T_bypass_shapes=552,final_MAD_families=[*FAMILIES,'q_times_h_plus_count'],closed_connections=1656,
        mixed_distance_closed_connections=1104,inverse_product_closed_connections=552,
        original_search_counts=dict(range_unsat=910,unknown=194),
        original_search_seconds=record['seconds'],
        q_times_count_plus_h_mixed_distance_maximum=2*23*65535,
        count_times_h_plus_q_mixed_distance_maximum=2*65535,
        distinct_fibres=len({r['base'] for r in checks}),checks=checks,
        proof='On an eight-point b16..18 fibre the target low3 is a permutation, while all '
        'T-prefix low3 words are fixed. Hence q low3 must distinguish all eight points, '
        'so at most one count can clamp to0. On every other point actual count equals '
        'one common c in16..23, by low16 invariance and the universal count band. The two '
        'disjoint four-point rectangles cover the fibre, so at least one is entirely '
        'unclamped. Both have Delta(h)=0 for every root parameter and target mixed '
        'distance greater than3014610. But q*c+h has mixed distance at most2*c*65535, '
        'and c*h+q has at most2*65535, contradictions on that unclamped rectangle. '
        'For q*h+count, the separate complete low-half inverse scan proves neither rectangle '
        'pair can satisfy the product equations; target low16 avoids the count band, so no '
        'clamped point is possible in this third family.',
        inverse_product_scan=dict(evaluated_parameters=inverse['evaluated_parameters'],remaining_models=0,
                                  vector_replayed=True,scalar_full_word_scan_preserved=True),
        evidence_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (RECTANGLES,SCOUT,INVERSE)},
        legal_points=[legal_point(b) for b in sorted(legal)],
        remaining_in_stated_three_tail_families=0,new_executable=False,best_executable_cycles=864,
        no_global_optimality_claim=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--find',action='store_true')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.find:find(args.output);return
    result=audit();args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('1656 count-aware final MAD connections excluded;194 timeouts preserved; executable864.',flush=True)


if __name__=='__main__':main()
