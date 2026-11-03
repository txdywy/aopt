"""A single ordinary tail cannot repair an arbitrary one-high word.

q may be any function of b, but has at most one distinct value above2^31.
No restriction on its prefix is used. Scope: nine stated initial words,
root-only constants, fixed allowed root0x12345678, and the actual target.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from codex.audit_comparison_core863 import fibre
from codex.audit_prepared_numerator863 import low_scalar_filter
from codex.cegis_c1_absorb863 import _instr_options
from codex.close_computed_denominator863 import prepared_bitop_certificates
from codex.screen_middle_division_residual863 import pool
from codex.search_early_boundary863 import MASK,NAMES,NODE,early_inputs,wanted,legal_point


def tail_class(op,w):
    q=9
    if q not in w:return 'T_only_lowbit'
    if op=='mad' and w[0]==w[1]==q:
        return 'square_plus_self_even' if w[2]==q else 'square_mod4'
    if not any(j<9 for j in w):return 'q_only_permutation'
    if op!='mad':return 'bitop_initial'
    a,b,c=w
    if q in (a,b):
        other=b if a==q else a
        if other<9:return 'initial_plus_one_factor' if c==q else 'even_initial_factor'
        assert c<9
        return 'odd_scalar_q_initial_addend'
    assert c==q
    return 'fixed_initial_product_plus_q' if a<9 and b<9 else 'scalar_initial_product_plus_q'


@lru_cache(maxsize=1)
def information_fibre():
    # Chosen by a deterministic exploration, then checked as a finite witness.
    width=9;base=136315184
    bs=np.arange(1<<(19-width),dtype=np.uint64)<<width;bs|=base
    with np.errstate(over='ignore'):vv=early_inputs(bs);yy=wanted(vv)
    mask=(1<<width)-1
    assert all(np.all((v&mask)==(v[0]&np.uint64(mask))) for v in vv)
    witnesses={}
    for b,y in zip(bs,yy):witnesses.setdefault(int(y)&mask,int(b))
    assert len(witnesses)==478
    count=len(witnesses)
    return dict(width=width,base=base,primary_b=list(witnesses.values()),
                initial_low_bits=[int(v[0])&mask for v in vv],target_residues=list(witnesses),
                distinct_target_residues=count,prepared_floor_divisor_max=MASK//(count-1),
                prepared_remainder_divisor_max=MASK//(count-1),
                prepared_ceil_divisor_max=(MASK-1)//(count-2),
                prepared_right_shift_max=23,
                proof='All initial low9 are fixed. T-only prefixes/suffixes can distinguish points only through q low9. Thus q needs478 distinct low9 values. Floor has at most floor(MASK/D)+1 outputs; ceil at most ceil(MASK/D)+1; REM has at most floor(MASK/D)+1 low9 values because P low9 is fixed and P%D=P-D*k. SHR>=24 has at most256 outputs. These yield the stated exact integer bounds.')


def audit():
    evidence={}
    def read(name):
        path=Path('codex')/name;raw=path.read_bytes();d=json.loads(raw)
        assert d['node']==NODE;evidence[str(path)]=hashlib.sha256(raw).hexdigest();return d
    enum=read('one_high_tail863_full_enum_20261029.json')
    prepared=read('prepared_numerator863_audit.json');refine=read('prepared_scalar863_refinement.json')
    assert enum['complete'] and enum['all_one_high_excluded']
    assert enum['scanner_source_sha256']==hashlib.sha256(Path('codex/enumerate_middle_residual863.cpp').read_bytes()).hexdigest()
    assert {r['id'] for r in enum['results']}==set(range(9))
    legal=set(enum['dataset']['points'])
    for row in enum['results']:
        assert row['complete'] and row['width']==32 and row['evaluated_odd_parameters']==1<<31 and not row['models']
    for case in enum['dataset']['cases']:
        assert len(set(case['residuals']))==len(case['residuals'])
        for b,r in zip(case['primary_witnesses'],case['residuals']):
            v=early_inputs(b);assert (wanted(v)-v[case['id']])&MASK==r
    fixed=[]
    for row in prepared['fixed_initial_products']:
        a,b=row['factors'];qs=[]
        for w in row['high_outputs']:
            v=early_inputs(w['primary_b']);q=(wanted(v)-v[a]*v[b])&MASK
            assert q==w['q'] and q>1<<31;qs.append(q);legal.add(w['primary_b'])
        assert len(set(qs))==2;fixed.append(row)
    assert len(fixed)==45
    scalars=[]
    for source in range(9):
        row=refine if source==0 else next(r for r in prepared['scalar_initial_products'] if r['source']==source)
        if row.get('certificate')=='two_distinct_high_targets_at_zero_factor':
            qs=[]
            for w in row['high_outputs']:
                v=early_inputs(w['primary_b']);assert v[source]==0 and wanted(v)==w['q']>1<<31
                qs.append(w['q']);legal.add(w['primary_b'])
            assert len(set(qs))==2
        else:
            lows,history=low_scalar_filter(row['initial_factors'],row['targets'])
            assert not lows and history==row['history']
            for b,x,y in zip(row['primary_b'],row['initial_factors'],row['targets']):
                v=early_inputs(b);assert v[source]==x and wanted(v)==y;legal.add(b)
        scalars.append(dict(source=source,certificate=row))
    parities=[];plus_one=[];classes=(0,1,1<<19,(1<<19)|1)
    for source in range(9):
        even=next(b for b in classes if early_inputs(b)[source]&1==0)
        bs=[even,even^65536];vv=[early_inputs(b) for b in bs];ys=[wanted(v) for v in vv]
        assert all(v[source]&1==0 for v in vv) and (ys[0]^ys[1])&1
        assert [x&1 for x in vv[0]]==[x&1 for x in vv[1]]
        parities.append(dict(source=source,primary_b=bs));legal.update(bs)
        if source!=3:
            odd=next(b for b in classes if early_inputs(b)[source]&1)
            bs=[odd,odd^65536];vv=[early_inputs(b) for b in bs]
            assert all(v[source]&1 for v in vv) and (wanted(vv[0])^wanted(vv[1]))&1
            plus_one.append(dict(source=source,primary_b=bs));legal.update(bs)
    forced={}
    for b in pool():
        v=early_inputs(b);assert v[3]&1==0
        q=(wanted(v)*pow(v[3]+1,-1,1<<32))&MASK
        if q>1<<31:forced.setdefault(q,b)
        if len(forced)==2:break
    assert len(forced)==2;legal.update(forced.values())
    all_bit=prepared_bitop_certificates()
    bit={k:all_bit[k] for k in ('xor_initial','or_initial_zero_witness','and_initial_mask_witness')}
    legal.add(bit['or_initial_zero_witness']['primary_b']);legal.add(bit['and_initial_mask_witness']['primary_b'])
    for row in bit['xor_initial']:
        qs=[]
        for w in row['distinct_high_quotients']:
            v=early_inputs(w['primary_b']);assert wanted(v)^v[row['source']]==w['q']>1<<31
            qs.append(w['q']);legal.add(w['primary_b'])
        assert len(set(qs))==2
    ff=fibre();assert len({y&3 for y in ff['target']})==4
    assert all(len({v[j]&3 for v in ff['initial_values']})==1 for j in range(9))
    legal.update(ff['primary_b']);info=information_fibre();legal.update(info['primary_b'])
    rows=[dict(op=op,wiring=list(w),certificate=tail_class(op,w))
          for op in ('mad','^','&','|') for w in _instr_options(9,op)]
    rows += [dict(op=op,wiring=[9,9],certificate=tail_class(op,[9,9])) for op in ('^','&','|')]
    assert len(rows)==913
    return dict(node=NODE,raw_root=0x12345678,input_names=NAMES,evidence_sha256=evidence,
                complete_in_stated_scope=True,scope='Any arbitrary q(b) with at most one distinct q>2^31, then one MAD/XOR/AND/OR on the nine initial words/q/root-only constants. The prefix of q is unrestricted. Fixed allowed raw root0x12345678.',
                full_odd_inverse_parameters=9*(1<<31),fresh_full_enumeration=True,
                fixed_products=fixed,scalar_initial_products=scalars,even_initial_factor_pairs=parities,
                plus_one_even_factor_pairs=plus_one,v_plus_one_forced_high=[dict(q=q,primary_b=b) for q,b in forced.items()],
                bitop_certificates=bit,square_mod4_fibre=ff,information_fibre=info,
                tail_manifest=rows,tail_class_counts=dict(Counter(r['certificate'] for r in rows)),
                q_only_proof='A function of q/root constants equal to the full target permutation forces q to be a permutation. It would have2^31-1 distinct values above2^31, contradicting the one-high assumption.',
                scalar_q_proof='Even G fails the b16 low-bit pair. For odd G, q=G^-1*(Y-I); the fresh scanner exhausts every odd inverse and finds two distinct high q for every choice.',
                legal_points=[legal_point(b) for b in sorted(legal)],
                new_executable=False,best_executable_cycles=864,no_global_optimality_claim=True)


@lru_cache(maxsize=1)
def checked_certificate():
    path=Path('codex/one_high_tail863_audit.json')
    if not path.exists():return None
    raw=path.read_bytes();record=json.loads(raw)
    assert record==json.loads(json.dumps(audit()))
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())


def ordinary_tail_certificate(row,node):
    """Sole prepared division/shift; last T reads no other computed word."""
    if node!=NODE or row['order'][-1] not in ('mad','^','&','|'):return None
    nonlinear=[i for i,op in enumerate(row['order']) if op in ('shr16','shr','//','%','cdiv')]
    if len(nonlinear)!=1:return None
    if any(op not in ('mad','^','&','|','shr16','shr','//','%','cdiv') for op in row['order']):return None
    q=8+nonlinear[0];last=8+len(row['order'])-1;w=[]
    for j in row['wiring'][-1]:
        if j<8:w.append(j+1)
        elif j==q:w.append(9)
        elif j>=last:w.append(10+j-last)
        else:return None
    proof=checked_certificate()
    if proof is None:return None
    return dict(**proof,tail_class=tail_class(row['order'][-1],w),nine_word_tail_wiring=w,
                reason='DIV/CDIV divisor1 and even REM are T-only in bit0. For other prepared divisors, the478-state fibre bounds REM below2^31; floor/ceil are <=2^31. A prepared sole SHR must use16..23 and is also bounded. The arbitrary one-high-tail certificate excludes the remaining final T, independently of the prefix.')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();d=audit();args.output.write_text(json.dumps(d,indent=2)+'\n')
    print('Arbitrary one-high tail closed:',len(d['tail_manifest']),'forms; fresh inverse parameters',d['full_odd_inverse_parameters'],flush=True)
    print('Prepared divisor maxima:',d['information_fibre']['prepared_floor_divisor_max'],d['information_fibre']['prepared_ceil_divisor_max'],flush=True)


if __name__=='__main__':main()
