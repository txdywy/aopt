"""Replay and combine the continuation without relabeling raw timeouts."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from codex.audit_joint_quotient_bypass863 import audit as bypass_audit
from codex.close_prepared_joint863 import audit as previous_audit
from codex.refine_joint_quotient_range863 import target_quotient_maximum
from codex.search_focused_joint_boundary863 import key
from codex.search_joint_live_boundary863 import NODE
from codex.search_keyed_live_restore863 import manifest, parallel_manifest
from codex.search_mixed_affine_restore863 import search as mixed_search
from codex.search_restore_scaled863 import check_normalization


def normalized(value):
    return json.loads(json.dumps(value))


def audit():
    evidence = {}
    def read(name):
        path = Path('codex')/name
        raw = path.read_bytes()
        decoded = gzip.decompress(raw) if path.suffix=='.gz' else raw
        evidence[str(path)] = dict(sha256=hashlib.sha256(raw).hexdigest(),
                                  decoded_sha256=hashlib.sha256(decoded).hexdigest())
        return json.loads(decoded)
    old,old_open,old_magic = previous_audit()
    assert old == read('prepared_joint863_audit.json')
    assert old_open == read('joint_prepared_division863_open_20261029.json')
    assert old_magic == read('joint_prepared_division863_magic_open_20261029.json')
    proof = read('joint_computed_bypass863_audit_20261029.json')
    assert proof == normalized(bypass_audit(Path('codex/joint_computed_bypass863_enum_20261029.json')))
    read('joint_computed_bypass863_enum_20261029.json')
    pending = {key(r):r for r in old_open['results']}
    closed = {}
    range_reports = []
    previous_source = evidence['codex/joint_prepared_division863_open_20261029.json']['sha256']
    for name in ('joint_quotient_range863_scout_20261029.json.gz',
                 'joint_quotient_range863_refinement_20261029.json.gz',
                 'joint_quotient_range863_min3_20261029.json.gz'):
        run = read(name)
        assert run['source_sha256'] == previous_source
        previous_source = evidence['codex/'+name]['decoded_sha256']
        assert run['node']==NODE and run['completed']==len(run['manifest'])==len(run['results'])
        assert {key(r) for r in run['results']} == set(pending)-set(closed)
        assert Counter(r['status'] for r in run['results']) == run['counts']
        for row in run['results']:
            if row['status']=='range_unsat':
                op = next(o for o in row['order'] if o in ('//','%','cdiv'))
                # Earlier scout ranges were looser; those UNSAT observations
                # remain sound supersets of the final, stronger bounds.
                assert row['quotient_maximum'] >= target_quotient_maximum(op)
                closed[key(row)] = dict(order=row['order'],wiring=row['wiring'],
                                        status='range_unsat',evidence='codex/'+name)
            else:
                assert row['status'] in ('range_sat','unknown')
        range_reports.append(dict(path='codex/'+name,counts=run['counts']))
    assert len(closed)==113
    before = set(closed)
    for row in proof['closed_rows']:
        kk = key(row)
        if kk not in pending or kk in closed:
            continue
        closed[kk] = dict(order=row['order'],wiring=row['wiring'],status=row['family']+'_excluded',
                          evidence='codex/joint_computed_bypass863_audit_20261029.json')
    assert set(closed)==set(pending)
    newly_structural = {k:r for k,r in closed.items() if k not in before}
    assert len(newly_structural)==14
    assert all(key(r) in closed for r in old_magic['results'])

    sampled = read('keyed_live_restore863_scout_20261029.json.gz')
    sample_ref = read('keyed_live_restore863_refinement_20261029.json.gz')
    parallel = read('keyed_live_restore863_parallel_20261029.json.gz')
    parallel_ref = read('keyed_live_restore863_parallel_refinement_20261029.json.gz')
    expected_sample,_ = manifest(sampled['per_order'],sampled['seed'])
    expected_parallel,_ = parallel_manifest()
    assert sampled['manifest']==expected_sample and parallel['manifest']==expected_parallel
    norm = read('mixed_affine_restore863_normalized_20261029.json')
    assert norm['status']=='unsat' and norm['width']==11 and norm['control'] is None
    assert mixed_search(11,10)['status']=='unsat'
    single = read('keyed_live_restore863_single_input_replay_20261029.json')
    assert single['normalization_control']==check_normalization()
    query = single['query']
    assert query['status']=='unsat' and query['width']==11 and query['output_key']==0xb55afa53
    assert query['arbitrary_input_key'] and not query['control'] and not query['pinned_control']
    def checked_restore(run):
        assert run['completed']==len(run['manifest'])==len(run['results'])
        assert {key(r) for r in run['manifest']}=={key(r) for r in run['results']}
        assert Counter(r['status'] for r in run['results'])==run['counts']
        assert run['positive_control']['status']=='proved_full32'
        for row in run['results']:
            assert row['width']==11 and row['output_key']==0xb55afa53
            assert row['key_pin'] is None and not row['pinned_constants'] and row['control'] is None
            assert row['status'] in ('unsat','unknown')
    for run in (sampled,sample_ref,parallel,parallel_ref):
        checked_restore(run)
    assert sample_ref['source_sha256']==evidence['codex/keyed_live_restore863_scout_20261029.json.gz']['decoded_sha256']
    assert parallel_ref['source_sha256']==evidence['codex/keyed_live_restore863_parallel_20261029.json.gz']['decoded_sha256']
    def refine(run,ref):
        rows = {key(r):r for r in run['results']}
        assert {k for k,r in rows.items() if r['status']=='unknown'}=={key(r) for r in ref['results']}
        rows.update({key(r):r for r in ref['results']})
        return rows
    sample_rows = refine(sampled,sample_ref)
    parallel_rows = refine(parallel,parallel_ref)
    assert all(r['status']=='unsat' for r in sample_rows.values())
    open_parallel = {k:r for k,r in parallel_rows.items() if r['status']=='unknown'}
    single_input = dict(order=['mad','mad','^','mad'],wiring=[[2,3,4],[2,4,5],[3,4],[5,6,7]])
    assert set(open_parallel)=={key(single_input)}
    # Set x=a XOR K. The target becomes F(x XOR (K XOR C1)) XOR PRE_C5;
    # both arms are affine in x, hence contained in the replayed P1/Q2 family.
    parallel_rows[key(single_input)] = dict(single_input,status='single_input_polynomial_excluded')
    restore_keys = set(sample_rows)|set(parallel_rows)
    return dict(node=NODE,best_executable_cycles=864,new_executable=False,no_global_optimality_claim=True,
                previous_prepared_division_connections=528,previously_closed=401,
                previous_remaining=127,range_continuation=range_reports,
                newly_closed_counts=dict(Counter(r['status'] for r in closed.values())),
                all528_sampled_full_divisor_connections_excluded=True,remaining_of_the528=0,
                generic_proofs_also_close_previous110_pinned_branches=True,
                final_previously_open_rows=[r for _,r in sorted(closed.items())],
                new_odd_inverse_parameters=proof['inverse_scan']['evaluated_odd_parameters'],
                masked_prior_rectangle=proof['masked_prior_rectangle'],
                keyed_restore=dict(sampled_connections=504,parallel_family_connections=1395,
                    unique_connections=len(restore_keys),intersection=len(sample_rows)+len(parallel_rows)-len(restore_keys),
                    all_selected_connections_excluded=True,normalized_final_family=True,
                    raw_parallel_unknowns=3,refinement_remaining_unknowns=1,
                    final_unknown_closed_by_separate_proof=True,not_complete_four_op_grammar=True),
                evidence=evidence,
                still_open='Complete four-op grammars, different earlier live words/encodings/opcodes, '
                'runtime divisors/counts, larger joint windows and new kernel layouts. No sub864 executable found.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    result = audit()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('All528 sampled prepared divisions closed;',result['newly_closed_counts'],flush=True)
    print('Keyed live restore:',result['keyed_restore'],flush=True)


if __name__ == '__main__':
    main()
