"""Merge scoped prepared-operand searches, preserving every unresolved case."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import z3

from codex.audit_one_high_tail863 import audit as one_high
from codex.audit_quotient_pair863 import audit as pair_audit
from codex.joint_polynomial_quotient863 import certificate as polynomial
from codex.search_joint_live_boundary863 import NODE,NAMES,prefix_cut_guards


CLOSED={'unsat','shift_wordcut_excluded','division_wordcut_excluded','lsb_dependency_excluded',
        'one_high_tail_excluded','polynomial_quotient_excluded','quotient_pair_excluded'}


def key(row):
    return tuple(row['order']),tuple(tuple(w) for w in row['wiring']),row.get('count_pin'),row.get('divisor_pin')


def normalized(value):return json.loads(json.dumps(value))


def audit():
    sources={}
    def read(name):
        path=Path('codex')/name;raw=path.read_bytes();decoded=gzip.decompress(raw) if path.suffix=='.gz' else raw
        d=json.loads(decoded);assert d['node']==NODE
        sources[str(path)]=dict(sha256=hashlib.sha256(raw).hexdigest(),decoded_sha256=hashlib.sha256(decoded).hexdigest())
        return d
    high=read('one_high_tail863_audit.json');assert high==normalized(one_high())
    pair=read('quotient_pair863_audit.json');assert pair==normalized(pair_audit())
    def checked_run(name):
        d=read(name);assert d['completed']==len(d['manifest'])==len(d['results'])
        assert len({key(r) for r in d['manifest']})==len(d['manifest'])
        assert {key(r) for r in d['manifest']}=={key(r) for r in d['results']}
        assert not any(r['status']=='proved_equivalent' for r in d['results'])
        assert d['positive_control']['status']=='proved_equivalent'
        return d
    def merge(previous,run):
        pending={k for k,r in previous.items() if r['status'] not in CLOSED}
        assert pending=={key(r) for r in run['results']}
        previous.update({key(r):r for r in run['results']})
    s0=checked_run('joint_prepared_shift863_scout_20261029.json.gz')
    s1=checked_run('joint_prepared_shift863_refinement_20261029.json')
    s2=checked_run('joint_prepared_shift863_prefix_20261029.json')
    split=checked_run('joint_prepared_shift863_split_20261029.json.gz')
    shift={key(r):r for r in s0['results']};assert len(shift)==1384
    merge(shift,s1)
    pending_shift={k for k,r in shift.items() if r['status'] not in CLOSED};assert len(pending_shift)==1
    assert {key(r)[:2] for r in split['results']}=={k[:2] for k in pending_shift}
    assert {r['count_pin'] for r in split['results']}==set(range(16,32))
    for row in split['results']:
        assert [row['order'].index('shr'),0,row['count_pin']] in row['pinned_constants']
    # Independent replay of the full-parameter prefix cut, not sixteen timeout relabels.
    row=s2['results'][0];kk=[z3.BitVecs(f'joint_close_{i}a joint_close_{i}b',32) for i in range(4)]
    guards=prefix_cut_guards(row,kk);solver=z3.SolverFor('QF_BV');solver.set(timeout=5000)
    solver.add(*(g for _,g in guards));assert solver.check()==z3.unsat
    assert row['prefix_mod4_guard_cuts']==[i for i,_ in guards]
    merge(shift,s2);assert all(r['status'] in CLOSED for r in shift.values())
    d0=checked_run('joint_prepared_division863_scout_20261029.json.gz')
    d1=checked_run('joint_prepared_division863_refinement_20261029.json.gz')
    d2=checked_run('joint_prepared_division863_product_20261029.json.gz')
    magic=checked_run('joint_prepared_division863_magic_20261029.json.gz')
    division={key(r):r for r in d0['results']};assert len(division)==528
    merge(division,d1);merge(division,d2)
    open_div=[r for _,r in sorted(division.items()) if r['status'] not in CLOSED]
    assert len(open_div)==127 and all(r['status']=='unknown' for r in open_div)
    generic_keys={key(r) for r in open_div}
    assert len(magic['results'])==254
    assert {(key(r)[:3]+(None,),r['divisor_pin']) for r in magic['results']}=={
        (k,d) for k in generic_keys for d in (65535,65537)}
    for row in magic['results']:
        i=next(i for i,o in enumerate(row['order']) if o in ('//','%','cdiv'))
        assert [i,0,row['divisor_pin']] in row['pinned_constants']
    open_magic=[r for r in magic['results'] if r['status'] not in CLOSED]
    assert len(open_magic)==110 and all(r['status']=='unknown' for r in open_magic)
    for row in division.values():
        if row['status']=='one_high_tail_excluded':
            assert row['certificate']['sha256']==sources['codex/one_high_tail863_audit.json']['sha256']
        if row['status']=='polynomial_quotient_excluded':assert row['certificate']==normalized(polynomial())
    def open_report(rows,origin):
        return dict(node=NODE,input_names=list(NAMES),free_e=False,sampled_not_exhaustive=True,
                    no_complete_four_op_grammar_claim=True,seed=origin['seed'],per_order=origin['per_order'],
                    manifest=[{k:r[k] for k in ('order','wiring','divisor_pin') if k in r} for r in rows],
                    results=rows,completed=len(rows),counts=dict(Counter(r['status'] for r in rows)),
                    orders=origin['orders'],positive_control=origin['positive_control'],
                    pinned_divisors=sorted({r['divisor_pin'] for r in rows if 'divisor_pin' in r}),
                    best_executable_cycles=864,new_executable=False)
    result=dict(node=NODE,best_executable_cycles=864,new_executable=False,no_global_optimality_claim=True,
                evidence=sources,prepared_shift_connections=len(shift),
                prepared_shift_final_counts=dict(Counter(r['status'] for r in shift.values())),
                all_sampled_prepared_shifts_excluded=True,independent_prefix_guard_replayed=True,
                split_raw_counts=split['counts'],split_unknowns_superseded_by_full_parameter_prefix_proof=True,
                prepared_division_connections=len(division),prepared_division_final_counts=dict(Counter(r['status'] for r in division.values())),
                prepared_division_complete=False,remaining_prepared_division=len(open_div),
                remaining_division_by_opcode=dict(Counter(next(op for op in r['order'] if op in ('//','%','cdiv')) for r in open_div)),
                pinned_magic_counts=magic['counts'],remaining_magic_branches=len(open_magic),
                pinned_branches_do_not_close_full_divisor_connections=True,
                full_odd_inverse_parameters=high['full_odd_inverse_parameters'],information_fibre=high['information_fibre'],
                complete_divisor_sum_scans=pair['sums'],
                still_open='The complete four-op grammar,127 sampled full-divisor connections and110 concrete magic-divisor branches; runtime denominators/counts, extra live data, larger windows and other kernel algorithms.')
    return result,open_report(open_div,d2),open_report(open_magic,magic)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('output','open-output','magic-open-output'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();record,open_div,open_magic=audit()
    for path,value in ((args.output,record),(args.open_output,open_div),(args.magic_open_output,open_magic)):
        path.write_text(json.dumps(value,indent=2)+'\n')
    print('Prepared shifts:',record['prepared_shift_final_counts'],flush=True)
    print('Prepared division:',record['prepared_division_final_counts'],flush=True)
    print('Open full-divisor / concrete branches:',record['remaining_prepared_division'],record['remaining_magic_branches'],flush=True)


if __name__=='__main__':main()
