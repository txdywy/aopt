"""Necessary bit0 conditions for a first T count followed by a dynamic SHR.

The initial pool excludes e. All other instructions, of any length, may be
T-only and can retain every initial/prefix word. In each (b0,b19) class their
output bit0 can only be q0 or its complement, since the real target flips
on b16. This is a prefix screen, never a full program or a cycle score.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import gzip
import hashlib
from itertools import combinations_with_replacement
import json
from pathlib import Path
import random
import time

import numpy as np
import z3

from codex.scout_joint_wirings863 import write_report
from codex.cegis_c1_absorb863 import _instr_options
from codex.search_focused_joint_boundary863 import edge_points
from codex.search_joint_live_boundary863 import inputs, wanted, NODE, NAMES, MASK, sample_data


def count_word(row,constants,values):
    if row['op']=='mad':
        pool=list(values)+list(constants[:2]);a,b,c=[pool[i] for i in row['wiring']]
        return (a*b+c)&MASK
    source=values[row['source']]
    right=values[row['other']] if 'other' in row else constants[0]
    return source & right if row['op']=='&' else source ^ right if row['op']=='^' else source | right


def prefix(row, constants, values):
    count=count_word(row,constants,values)
    numerator = count if row['numerator']=='count' else constants[-1] if row['numerator']=='constant' else values[row['numerator']]
    if z3.is_bv(count) or z3.is_bv(numerator):
        return z3.LShR(numerator if z3.is_bv(numerator) else z3.BitVecVal(int(numerator),32),count)
    if isinstance(count,np.ndarray) or isinstance(numerator,np.ndarray):
        return np.right_shift(np.uint64(numerator),np.minimum(count,32).astype(np.uint64))
    return 0 if count>=32 else numerator >> count


def target_bit(b,control=False):
    if control=='u16':return (inputs(b)[1]>>16)&1
    return ((b >> 16)^1)&1 if control else wanted(inputs(b),NODE)&1


@lru_cache(maxsize=1)
def points():
    rng = random.Random(8642957)
    initial = list(dict.fromkeys([*sample_data(True)[0],*edge_points()]))
    tests = list(dict.fromkeys([*initial,*(rng.getrandbits(32) for _ in range(4096))]))
    return initial,tests


def solve(row,seconds=2,control=False,pins=(),use_count_band=False):
    if row['op'] not in ('mad','^','&','|') or row['op']!='mad' and row['source'] not in range(8):
        raise ValueError('Initial-word bitop count required')
    if row['op']=='mad' and (len(row['wiring'])!=3 or any(i not in range(10) for i in row['wiring'])):
        raise ValueError('Invalid count MAD wiring')
    if row['numerator'] not in (*range(8),'constant','count'):
        raise ValueError('Initial, constant or count numerator required')
    if 'other' in row and row['other'] not in range(8):
        raise ValueError('Invalid second runtime count operand')
    started = time.monotonic()
    kk = z3.BitVecs('rcp_count0 rcp_count1 rcp_numerator' if row['op']=='mad' else
                    'rcp_count_parameter rcp_numerator_parameter',32)
    flips = z3.BitVecs(' '.join(f'rcp_flip{i}' for i in range(4)),1)
    solver = z3.SolverFor('QF_BV')
    if use_count_band:
        assert not control,'The count band is specific to the actual full target'
        from codex.audit_runtime_shift_band863 import audit
        audit()
    for index,value in pins:
        solver.add(kk[index]==(value&MASK))
    initial,tests = points();samples=list(initial)
    def constrain(b):
        q = prefix(row,kk,inputs(b))
        q = q if z3.is_bv(q) else z3.BitVecVal(q,32)
        cls = (b&1)|(((b>>19)&1)<<1)
        solver.add(z3.Extract(0,0,q)==z3.BitVecVal(target_bit(b,control),1)^flips[cls])
        if use_count_band:
            count=count_word(row,kk,inputs(b))&65535
            if not z3.is_bv(count):count=z3.BitVecVal(int(count),32)
            solver.add(z3.UGE(count,16),z3.ULE(count,23))
    for b in samples:
        constrain(b)
    report = dict(row,status='unknown',events=[],control=control,pins=[list(p) for p in pins],
                  initial_words=list(NAMES),free_e=False,node=NODE,new_executable=False,
                  universal_count_band_used=use_count_band)
    with np.errstate(over='ignore'):
        xs=np.array(tests,dtype=np.uint64);vv=inputs(xs)
        want=target_bit(xs,control)
        classes=(xs&1)|(((xs>>19)&1)<<1)
    while time.monotonic()-started < seconds:
        solver.set(timeout=max(1,int(1000*(seconds-time.monotonic()+started))))
        status=solver.check();event=dict(samples=len(samples),status=str(status));report['events'].append(event)
        if status!=z3.sat:
            report['status']='lsb_unsat' if status==z3.unsat else 'unknown'
            if status==z3.unknown:report['reason']=solver.reason_unknown()
            break
        model=solver.model()
        cv=[model.eval(k,model_completion=True).as_long() for k in kk]
        fv=[model.eval(f,model_completion=True).as_long() for f in flips]
        with np.errstate(over='ignore'):
            got=prefix(row,cv,vv)&1
            bad=np.flatnonzero(got!=(want ^ np.array(fv,dtype=np.uint64)[classes]))
        if len(bad):
            take=[tests[i] for i in bad[:8]]
        else:
            b=z3.BitVec('rcp_verify_b',32)
            q=prefix(row,[z3.BitVecVal(k,32) for k in cv],inputs(b))
            ff=[z3.BitVecVal(f,32) for f in fv]
            flip=z3.If(z3.Extract(19,19,b)==0,
                       z3.If(z3.Extract(0,0,b)==0,ff[0],ff[1]),
                       z3.If(z3.Extract(0,0,b)==0,ff[2],ff[3]))
            verify=z3.SolverFor('QF_BV')
            verify.set(timeout=max(1,int(1000*(seconds-time.monotonic()+started))))
            violations=[(q&1)!=(target_bit(b,control)^flip)]
            if use_count_band:
                count=count_word(row,[z3.BitVecVal(k,32) for k in cv],inputs(b))&65535
                violations.extend((z3.ULT(count,16),z3.UGT(count,23)))
            verify.add(z3.Or(*violations))
            checked=verify.check();event['full32_check']=str(checked)
            if checked==z3.unsat:
                report.update(status='lsb_feasible_full32',constants=cv,class_flips=fv)
                break
            if checked==z3.unknown:
                report.update(status='verification_unknown',reason=verify.reason_unknown())
                break
            take=[verify.model().eval(b,model_completion=True).as_long()]
        for b in take:
            assert b not in samples
            samples.append(b);constrain(b)
    report.update(seconds=round(time.monotonic()-started,4),samples=len(samples),primary_samples=samples)
    return report


def manifest(family='constant'):
    if family not in ('constant','runtime','mad'):
        raise ValueError('Unknown first-count family')
    rows=[]
    if family=='mad':
        return [dict(op='mad',wiring=list(w),numerator=n)
                for w in _instr_options(7,'mad') for n in (*range(8),'constant','count')]
    for op in ('^','&','|'):
        pairs=[(i,None) for i in range(8)] if family=='constant' else combinations_with_replacement(range(8),2)
        for source,other in pairs:
            for numerator in (*range(8),'constant','count'):
                row=dict(op=op,source=source,numerator=numerator)
                if other is not None:row['other']=other
                rows.append(row)
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--family',choices=('constant','runtime','mad'),default='constant')
    ap.add_argument('--seconds',type=float,default=2)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--source',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();provenance={}
    if args.source:
        raw=args.source.read_bytes();old=json.loads(gzip.decompress(raw) if args.source.suffix=='.gz' else raw)
        assert old['completed']==len(old['manifest'])==len(old['results'])
        rows=[{k:r[k] for k in ('op','source','other','wiring','numerator') if k in r}
              for r in old['results'] if r['status']!='lsb_unsat']
        args.family=old['family']
        provenance=dict(source=str(args.source),source_sha256=hashlib.sha256(raw).hexdigest())
    else:rows=manifest(args.family)
    control=solve(dict(op='&',source=3,numerator='constant'),10,True,[(0,65536),(1,1)])
    assert control['status']=='lsb_feasible_full32',control
    record=dict(family=args.family,node=NODE,manifest=rows,positive_control=control,
                scope='One first T count then one dynamic SHR, followed by any T-only suffix retaining all eight initial words. '
                'Only necessary bit0 conditions; a feasible prefix is not a full macro or an executable.',
                results=[],completed=0,counts={},new_executable=False,best_executable_cycles=864,**provenance)
    started=time.monotonic()
    with ProcessPoolExecutor(args.workers) as pool:
        futures=[pool.submit(solve,row,args.seconds,False,(),args.family=='mad') for row in rows]
        for done,future in enumerate(as_completed(futures),1):
            record['results'].append(future.result())
            if done%20==0 or done==len(rows):
                record.update(completed=done,counts=dict(Counter(r['status'] for r in record['results'])),
                              seconds=round(time.monotonic()-started,3))
                write_report(args.output,record);print(done,record['counts'],record['seconds'],flush=True)


if __name__=='__main__':main()
