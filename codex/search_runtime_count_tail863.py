"""Range screens for final MADs that read both count and sole SHR result.

The bypass h is one T instruction over the eight initial words/root
constants. The count/numerator prefixes are otherwise unrestricted T-only.
This covers q*count+h and count*h+q, but not q*h+count or other tails.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import time

import z3

from codex.audit_runtime_shift_band863 import audit as band_audit
from codex.scout_joint_wirings863 import write_report
from codex.search_first_runtime_count863 import count_manifest, fibres
from codex.search_joint_live_boundary863 import inputs, wanted, NODE, MASK
from codex.search_runtime_count_prefix863 import count_word


FAMILIES=('q_times_count_plus_h','count_times_h_plus_q')


def fits(family,h,y):
    if not z3.is_bv(h) and not z3.is_bv(y):
        if family==FAMILIES[0]:return ((y-h)&MASK)<=23*65535
        if family==FAMILIES[1]:return any(((y-c*h)&MASK)<=65535 for c in range(16,24))
        raise ValueError(family)
    if family=='q_times_count_plus_h':return z3.ULE(y-h,23*65535)
    if family=='count_times_h_plus_q':return z3.Or(*(z3.ULE(y-c*h,65535) for c in range(16,24)))
    raise ValueError(family)


def fibre_fits(family,hh,yy,couple_count=False):
    if family==FAMILIES[1] and couple_count:
        # T-prefix count low16 is fixed on this fibre. On every active
        # point its full count must equal this one common c in16..23.
        return z3.Or(*(z3.AtMost(*(z3.Not(z3.ULE(y-c*h,65535))
                                   for h,y in zip(hh,yy)),1) for c in range(16,24)))
    return z3.AtMost(*(z3.Not(fits(family,h,y)) for h,y in zip(hh,yy)),1)


def solve(row,family,seconds=2,control=False,pins=(),couple_count=False):
    if family not in FAMILIES:raise ValueError(family)
    started=time.monotonic();constants=z3.BitVecs('rct_k0 rct_k1',32)
    solver=z3.SolverFor('QF_BV');solver.set(timeout=max(1,int(seconds*1000)))
    for i,value in pins:solver.add(constants[i]==(value&MASK))
    for fibre in fibres():
        hh=[];yy=[]
        for b in fibre:
            h=count_word(row,constants,inputs(b))
            h=h if z3.is_bv(h) else z3.BitVecVal(h,32)
            y=(h+16*(b&65535)) if control and family==FAMILIES[0] else (
                16*h+(b&65535)) if control else z3.BitVecVal(wanted(inputs(b),NODE),32)
            hh.append(h);yy.append(y)
        solver.add(fibre_fits(family,hh,yy,couple_count))
    status=solver.check()
    result=dict(row,family=family,status='range_unsat' if status==z3.unsat else
                'range_sat' if status==z3.sat else 'unknown',control=control,
                seconds=round(time.monotonic()-started,4),new_executable=False,coupled_fibre_count=couple_count)
    if status==z3.unknown:result['reason']=solver.reason_unknown()
    if status==z3.sat:
        model=solver.model();result['constants']=[model.eval(k,model_completion=True).as_long() for k in constants]
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds',type=float,default=2)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--source',type=Path)
    ap.add_argument('--coupled-count',action='store_true')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();band=band_audit();provenance={}
    controls=[solve(dict(op='mad',wiring=[0,8,9]),family,5,True,[(0,9),(1,0xfd7046c5)],args.coupled_count) for family in FAMILIES]
    assert all(r['status']=='range_sat' for r in controls),controls
    rows=count_manifest();connections=[(row,family) for row in rows for family in FAMILIES]
    if args.source:
        raw=args.source.read_bytes();old=json.loads(gzip.decompress(raw) if args.source.suffix=='.gz' else raw)
        assert old['completed']==len(old['results'])==len(old['h_manifest'])*2
        connections=[({k:r[k] for k in ('op','source','other','wiring') if k in r},r['family'])
                     for r in old['results'] if r['status']!='range_unsat']
        provenance=dict(source=str(args.source),source_sha256=hashlib.sha256(raw).hexdigest())
    report=dict(node=NODE,scope=__doc__,h_manifest=rows,families=FAMILIES,
        universal_count_band=band,primary_fibres=fibres(),positive_controls=controls,
        coupled_fibre_count=args.coupled_count,connection_manifest=[dict(row,family=family) for row,family in connections],
        completed=0,results=[],counts={},new_executable=False,best_executable_cycles=864,**provenance)
    started=time.monotonic()
    with ProcessPoolExecutor(args.workers) as pool:
        futures=[pool.submit(solve,row,family,args.seconds,False,(),args.coupled_count) for row,family in connections]
        for done,future in enumerate(as_completed(futures),1):
            report['results'].append(future.result())
            if done%20==0 or done==len(connections):
                report.update(completed=done,counts=dict(Counter(r['status'] for r in report['results'])),
                              seconds=round(time.monotonic()-started,3))
                write_report(args.output,report);print(done,report['counts'],flush=True)


if __name__=='__main__':main()
