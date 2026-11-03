"""Necessary count-only screen, allowing an arbitrary T-computed numerator.

A valid sole SHR has count low16 in16..23 and at most one clamped count
on every eight-point b16..18 fibre. A first-T count that is constant on
the finite probe set is deliberately retained by this screen.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import random
import time

import z3

from codex.audit_runtime_shift_band863 import audit as band_audit
from codex.scout_joint_wirings863 import write_report
from codex.search_joint_live_boundary863 import inputs, NODE
from codex.search_runtime_count_prefix863 import count_word, manifest


@lru_cache(maxsize=1)
def fibres():
    rng=random.Random(8642963)
    bases=list(dict.fromkeys([*range(16),1<<19,(1<<19)|1,
        *(rng.getrandbits(32)&~(7<<16) for _ in range(16))]))
    return [[base|(k<<16) for k in range(8)] for base in bases]


def count_manifest():
    rows=[]
    for family in ('constant','runtime','mad'):
        for row in manifest(family):
            if row['numerator']!=0:continue
            rows.append({k:v for k,v in row.items() if k!='numerator'})
    assert len(rows)==552
    return rows


def solve(row,seconds=2,require_nonconstant=True,pins=()):
    started=time.monotonic()
    constants=z3.BitVecs('frc_k0 frc_k1',32)
    ss=z3.SolverFor('QF_BV');ss.set(timeout=int(seconds*1000))
    for index,value in pins:ss.add(constants[index]==value)
    all_counts=[]
    for fibre in fibres():
        counts=[count_word(row,constants,inputs(b)) for b in fibre]
        counts=[x if z3.is_bv(x) else z3.BitVecVal(x,32) for x in counts]
        for t in counts:ss.add(z3.UGE(t&65535,16),z3.ULE(t&65535,23))
        ss.add(z3.AtMost(*(z3.UGE(t,32) for t in counts),1))
        all_counts.extend(counts)
    if require_nonconstant:ss.add(z3.Or(*(t!=all_counts[0] for t in all_counts[1:])))
    status=ss.check()
    report=dict(row,status='nonconstant_count_unsat' if status==z3.unsat else
                'count_screen_sat' if status==z3.sat else 'unknown',
                nonconstant_on_finite_probes_required=require_nonconstant,
                seconds=round(time.monotonic()-started,4),new_executable=False)
    if status==z3.unknown:report['reason']=ss.reason_unknown()
    if status==z3.sat:
        model=ss.model();report['constants']=[model.eval(k,model_completion=True).as_long() for k in constants]
    return report


def interpolate(row,seconds=5):
    """Prove that constancy on all recorded probes implies full32 constancy.

    Root constants are universally arbitrary: they are solver variables,
    rather than fixed to any model found by the preceding screen.
    """
    constants=z3.BitVecs('frc_interpolate_k0 frc_interpolate_k1',32)
    primary=z3.BitVec('frc_interpolate_b',32)
    baseline=count_word(row,constants,inputs(fibres()[0][0]))
    solver=z3.SolverFor('QF_BV');solver.set(timeout=int(seconds*1000))
    for fibre in fibres():
        for b in fibre:solver.add(count_word(row,constants,inputs(b))==baseline)
    solver.add(count_word(row,constants,inputs(primary))!=baseline)
    started=time.monotonic();status=solver.check()
    result=dict(row,status='finite_constancy_implies_full32_constant' if status==z3.unsat else
                'counterexample' if status==z3.sat else 'unknown',
                seconds=round(time.monotonic()-started,4),root_constants_unrestricted=True)
    if status==z3.unknown:result['reason']=solver.reason_unknown()
    return result


def close(source,output,seconds=5):
    encoded=source.read_bytes()
    record=json.loads(gzip.decompress(encoded) if source.suffix=='.gz' else encoded)
    rows=count_manifest();key=lambda r:json.dumps({k:r[k] for k in ('op','source','other','wiring') if k in r},sort_keys=True)
    assert record['completed']==len(rows)==len(record['results'])
    assert {key(r) for r in record['manifest']}=={key(r) for r in rows}
    assert {key(r) for r in record['results']}=={key(r) for r in rows}
    assert all(r['status']=='nonconstant_count_unsat' and
               r['nonconstant_on_finite_probes_required'] for r in record['results'])
    assert record['primary_fibres']==fibres()
    assert record['universal_count_band']==json.loads(json.dumps(band_audit()))
    assert record['positive_constant16_control']['status']=='count_screen_sat'
    checks=[]
    for i,row in enumerate(rows,1):
        checks.append(interpolate(row,seconds))
        if i%50==0:print('interpolation',i,dict(Counter(r['status'] for r in checks)),flush=True)
    result=dict(node=NODE,source=str(source),source_sha256=hashlib.sha256(encoded).hexdigest(),
        scope='A first count computed by one T=MAD/XOR/AND/OR from the eight initial words '
        'and arbitrary root constants, used as the sole SHR count. Numerator and all remaining '
        'computation may be arbitrary-length T expressions, with all bypasses retained.',
        primary_fibres=fibres(),checks=checks,counts=dict(Counter(r['status'] for r in checks)),
        every_one_T_count_must_be_root_constant=all(r['status']=='finite_constancy_implies_full32_constant' for r in checks),
        constant_count_minimum=16,constant_count_maximum=23,
        proof='Each eight-point fibre has fixed T-prefix/suffix low3 and eight distinct target '
        'low3 values. Therefore all q low3 values must be distinct, so at most one count may '
        'clamp to0. The finite screen proves every first-T count satisfying this condition '
        'and the universal band is constant on the probe set. Separate QF_BV queries quantify '
        'over all full32 b/root parameters and prove finite constancy implies full32 constancy '
        'for every count wiring. A constant count cannot clamp, so it lies in16..23.',
        new_executable=False,best_executable_cycles=864,no_global_optimality_claim=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(result['counts'],flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds',type=float,default=2)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--close-source',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.close_source:
        close(args.close_source,args.output,args.seconds);return
    band=band_audit();rows=count_manifest()
    control=solve(dict(op='mad',wiring=[0,8,9]),5,False,[(0,0),(1,16)])
    assert control['status']=='count_screen_sat',control
    report=dict(node=NODE,scope=__doc__,universal_count_band=band,
                primary_fibres=fibres(),manifest=rows,positive_constant16_control=control,
                completed=0,results=[],counts={},new_executable=False,best_executable_cycles=864)
    started=time.monotonic()
    with ProcessPoolExecutor(args.workers) as pool:
        futures=[pool.submit(solve,row,args.seconds) for row in rows]
        for done,future in enumerate(as_completed(futures),1):
            report['results'].append(future.result())
            if done%20==0 or done==len(rows):
                report.update(completed=done,counts=dict(Counter(r['status'] for r in report['results'])),
                              seconds=round(time.monotonic()-started,3))
                write_report(args.output,report);print(done,report['counts'],flush=True)


if __name__=='__main__':main()
