"""Close first-T dynamic-count screens using finite witnesses and congruences.

This proves a prefix exclusion with arbitrary T-only suffix length, not a
four-op-grammar or whole-kernel lower bound. Original timeouts are preserved.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path

import z3

from codex.audit_division_first_t863 import full32_class_facts, source_inverse
from codex.audit_runtime_shift_band863 import audit as band_audit
from codex.search_early_boundary863 import legal_point
from codex.search_joint_live_boundary863 import inputs, wanted, NODE, NAMES, MASK
from codex.search_runtime_count_prefix863 import manifest, points


ARCHIVES = {family: Path('codex')/('runtime_count_prefix863_'+name+'_20261029.json.gz')
            for family,name in (('constant','constant'),('runtime','binary'),('mad','mad'))}


def row_key(row):
    return json.dumps({k:row[k] for k in ('op','source','other','wiring','numerator') if k in row},sort_keys=True)


@lru_cache(maxsize=1)
def affine_certificate():
    # A source reaching 0,1,8 and the universal count band force A=0 mod2^16.
    a,b=z3.BitVecs('affine_count_a affine_count_b',16)
    solver=z3.SolverFor('QF_BV')
    for x in (0,1,8):
        t=a*x+b
        solver.add(z3.UGE(t,16),z3.ULE(t,23))
    solver.add(a!=0)
    assert solver.check()==z3.unsat
    # For nonzero A divisible by 2^16, the equation A*I+B=c has at most
    # one possible parity of I. Check its exact low-(r+1)-bit projection
    # for every possible r=v2(A), without enumerating any coefficient.
    gating=[]
    for r in range(16,32):
        aa,bb,cc,i,j=z3.BitVecs(' '.join(f'ac_{r}_{s}' for s in ('a','b','c','i','j')),32)
        s=z3.SolverFor('QF_BV');s.set(timeout=5000)
        s.add(z3.Extract(r,0,aa)==1<<r,(i&1)!=(j&1),
              z3.Extract(r,0,aa*i+bb)==z3.Extract(r,0,cc),
              z3.Extract(r,0,aa*j+bb)==z3.Extract(r,0,cc))
        assert s.check()==z3.unsat,r
        gating.append(dict(valuation=r,opposite_parities_cannot_both_reach_c='unsat'))
    reaches=[]
    for source in range(8):
        if source==2:continue  # v=16896*b+0xACCF6200 uses effective coefficient in b.
        for value in (0,1,8):
            primary=source_inverse(source+1,value)
            assert primary is not None and inputs(primary)[source]==value
            reaches.append(dict(source=source,value=value,primary_b=primary))
    facts=full32_class_facts()
    return dict(count_low16_affine_coefficient_zero='unsat',source_reachability=reaches,
                nonzero_coefficient_parity_gating=gating,full32_class_facts=facts,
                v_rewrite='A*v+B=(16896*A)*b+(0xACCF6200*A+B) modulo2^32.',
                proof='For I!=v, reachable I=0,1,8 and the universal16..23 band force '
                'A divisible by2^16. If A!=0, write r=v2(A). A small count can only be '
                'c=B low16. The congruence A*I+B=c modulo2^(r+1) fixes I parity. '
                'The other I parity occurs in at least one entire b0/b19 class, where '
                'q is always0 but target parity flips on b16. Thus A=0. For v apply '
                'the same proof to effective affine coefficients in b. Every surviving '
                'count is a root-only constant, regardless of numerator.')


@lru_cache(maxsize=1)
def raw_shift_certificates():
    # A T-only suffix must have Y0=q0 XOR one fixed bit per (b0,b19) class.
    # Each pair below contradicts this for a prepared count/source choice.
    samples=points()[1];rows=[]
    for source in range(8):
        for count in range(16,24):
            seen={};pair=None
            for b in samples:
                cls=(b&1)|(((b>>19)&1)<<1)
                delta=((inputs(b)[source]>>count)^wanted(inputs(b),NODE))&1
                if (cls,1-delta) in seen:
                    pair=[seen[cls,1-delta],b];break
                seen.setdefault((cls,delta),b)
            assert pair is not None,(source,count)
            rows.append(dict(source=source,count=count,primary_pair=pair,
                             source_bits=[(inputs(b)[source]>>count)&1 for b in pair],
                             target_bits=[wanted(inputs(b),NODE)&1 for b in pair]))
    return rows


def audit():
    band=band_audit();affine=affine_certificate();raw=raw_shift_certificates()
    evidence={};counts=Counter();closed=[];legal=set()
    for row in affine['source_reachability']:legal.add(row['primary_b'])
    for row in raw:legal.update(row['primary_pair'])
    observations={}
    for family,path in ARCHIVES.items():
        encoded=path.read_bytes();record=json.loads(gzip.decompress(encoded))
        evidence[str(path)]=hashlib.sha256(encoded).hexdigest()
        expected={row_key(row) for row in manifest(family)}
        assert len(expected)==len(record['manifest'])==record['completed']==len(record['results'])
        assert {row_key(row) for row in record['manifest']}==expected
        assert {row_key(row) for row in record['results']}==expected
        assert Counter(row['status'] for row in record['results'])==record['counts']
        control=record['positive_control']
        assert control['status']=='lsb_feasible_full32' and control['constants']==[65536,1]
        observations[family]=dict(completed=record['completed'],counts=record['counts'],
                                  seconds=record['seconds'])
        for row in record['results']:
            assert len(row['events'])==1 and row['samples']==len(points()[0])
            if row['status']=='lsb_unsat':
                assert row['events'][0]['status']=='unsat'
                reason='necessary_bit0_smt_unsat'
            else:
                assert row['status']=='unknown' and row['reason']=='timeout'
                assert family=='mad' and row['wiring']==[row['wiring'][0],8,9]
                assert row['wiring'][0] in range(8)
                reason='prepared_numerator_information' if row['numerator']=='constant' else 'affine_count_then_raw_shift'
            counts[reason]+=1
            closed.append(dict(key=row_key(row),certificate=reason))
    assert len(closed)==5520 and counts==dict(necessary_bit0_smt_unsat=5470,
        prepared_numerator_information=7,affine_count_then_raw_shift=43)
    return dict(node=NODE,input_names=list(NAMES),free_e=False,
                scope='First count word is one MAD, XOR, AND or OR on eight initial words '
                'and arbitrary root-only constants, then SHR of any initial/root-only/count '
                'numerator by that count. Any length of T-only suffix may retain all bypasses. '
                'This does not cover a count prepared by two instructions, an independently '
                'computed numerator, or two non-T instructions.',
                universal_band=band,affine_count_certificate=affine,
                prepared_raw_shift_bit0_pairs=raw,evidence_sha256=evidence,
                original_search_observations=observations,closed_prefixes=closed,
                closed_counts=dict(counts),remaining_in_stated_prefix_family=0,
                self_shift_proof='For every uint32 t, t>>t=0: for t<32, t<2^t; for t>=32, SHR clamps0.',
                toy_clamping_control=dict(count='b & 65536',numerator=1,
                    target_bit='1 XOR b16',full32_verified=True,
                    actual_count_may_exceed31=True,band_must_not_be_applied_to_toy=True),
                legal_points=[legal_point(b) for b in sorted(legal)],
                new_executable=False,best_executable_cycles=864,no_global_optimality_claim=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();result=audit()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('5520 first-count prefixes excluded; original50 timeouts retained; executable864.',flush=True)


if __name__=='__main__':main()
