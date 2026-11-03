"""Combine joint four-op searches and independent residual certificates.

The 14,208 broad samples are not complete. The 4,200-case affine-tail
family is complete only for its stated MAD,SHR16,XOR,affine-MAD skeleton.
All original timeouts are preserved, then separately accounted for.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from codex.search_joint_live_boundary863 import (
    make_manifest, affine_tail_manifest, parity_models, NAMES, NODE,
)
from codex.search_live_boundary863 import primary_path_avoiding


def key(row):
    return tuple(row['order']), tuple(tuple(w) for w in row['wiring'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); evidence = {}; decoded_hashes = {}
    def read(path):
        actual = Path(path)
        if not actual.exists():
            actual = Path(path+'.gz')
        encoded = actual.read_bytes()
        decoded = gzip.decompress(encoded) if actual.suffix == '.gz' else encoded
        evidence[str(actual)] = hashlib.sha256(encoded).hexdigest()
        decoded_hashes[path] = hashlib.sha256(decoded).hexdigest()
        return json.loads(decoded)
    def combine(path, refs):
        old = read(path)
        assert old['node'] == NODE and old['input_names'] == list(NAMES) and not old['free_e']
        assert old['completed'] == len(old['manifest']) == len(old['results'])
        rows = {key(r):r for r in old['results']}
        assert len(rows) == len(old['manifest']) and set(rows) == set(map(key, old['manifest']))
        stages = [dict(path=path, counts=dict(Counter(r['status'] for r in rows.values())))]
        previous = path
        for ref in refs:
            other = read(ref)
            assert other['source'] == previous and other['source_sha256'] == decoded_hashes[previous]
            assert other['node'] == NODE and not other['free_e']
            assert other['completed'] == len(other['manifest']) == len(other['results'])
            pending = {k for k, r in rows.items() if r['status'] in ('unknown', 'verification_unknown')}
            assert set(map(key, other['results'])) == set(map(key, other['manifest'])) == pending
            rows.update({key(r):r for r in other['results']})
            stages.append(dict(path=ref, counts=dict(Counter(r['status'] for r in rows.values()))))
            previous = ref
        for r in rows.values():
            if r['status'] == 'lsb_dependency_excluded':
                assert not parity_models(r)
            if r['status'] == 'shift_wordcut_excluded':
                assert any(op == 'shr16' and not primary_path_avoiding(r, 8, False, 8+i)
                           for i, op in enumerate(r['order']))
            if r.get('first_prefix_mod4_guard_used'):
                assert r['order'][0] == 'mad'
                assert all(j >= 8 or j < 4 for j in r['wiring'][0])
                assert not primary_path_avoiding(r, 8, False, 8)
        return old, rows, stages
    scout, sr, scout_stages = combine('codex/joint_live_boundary863_scout.json',
                                     ['codex/joint_live_boundary863_refinement.json'])
    expected, _ = make_manifest(scout['per_order'], scout['seed'])
    assert len(sr) == 14208 and set(sr) == set(map(key, expected))
    assert scout['sampled_not_exhaustive']
    assert all(r['status'] in ('unsat', 'lsb_dependency_excluded', 'shift_wordcut_excluded') for r in sr.values())
    affine, ar, stages = combine('codex/joint_live_boundary863_affine_tail.json', [
        'codex/joint_live_boundary863_affine_refinement.json',
        'codex/joint_live_boundary863_affine_dense.json',
        'codex/joint_live_boundary863_affine_final.json'])
    expected, _ = affine_tail_manifest()
    assert len(ar) == 4200 and set(ar) == set(map(key, expected))
    assert affine['family'] == 'affine-tail' and not affine['sampled_not_exhaustive']
    old = read('codex/root_boundary863_fold_affine.json')
    assert old['node'] == NODE and old['family'] == 'fold_affine' and old['status'] == 'unsat'
    assert old['full32_semantics'] and not old['control'] and not old['pinned']
    sliced = read('codex/joint_live_boundary863_affine_slice.json')
    assert sliced['node'] == NODE
    assert sliced['control']['status'] == 'relaxed_slice_survivor' and sliced['control']['exhaustive_slice_passed']
    unsat_slices = {(r['source'], r['addend']) for r in sliced['results'] if r['status'] == 'unsat'}
    assert unsat_slices == {(1, None), (1, 2), (0, 1)}
    counts = Counter(); certificates = []
    for r in ar.values():
        if r['status'] != 'unknown':
            assert r['status'] in ('unsat', 'shift_wordcut_excluded')
            counts[r['status']] += 1
            continue
        assert r['order'] == ['mad', 'shr16', '^', 'mad']
        assert r['wiring'][1:] == [[8], [8, 9], [10, 11, 12]]
        first = r['wiring'][0]
        if first in ([0, 8, 8], [0, 8, 9]):
            cert = 'prior_full32_fold_affine_z_family'
        elif first == [0, 8, 1]:
            assert (0, 1) in unsat_slices; cert = 'full32_sliced_z_times_K_plus_u'
        elif first in ([1, 8, 8], [1, 8, 9]):
            assert (1, None) in unsat_slices; cert = 'full32_sliced_affine_u'
        else:
            raise AssertionError(first)
        counts[cert] += 1; certificates.append(dict(order=r['order'], wiring=r['wiring'], certificate=cert))
    assert len(certificates) == 5 and sum(counts.values()) == 4200
    high = read('codex/joint_live_boundary863_affine_high.json')
    assert high['node'] == NODE and len(high['results']) == 8
    assert all(r['status'] == 'unsat' for r in high['results'])
    assert all(r['status'] == 'necessary_survivor_proved' for r in high['controls'])
    ir = read('codex/joint_live_boundary863_ir_control.json')
    assert ir['replaced_boundaries'] == 32 and ir['removed_ops_per_boundary'] == ir['macro_instructions'] == 5
    assert ir['lane_work_before'] == ir['lane_work_after'] == 51665 and ir['new_setup_operations'] == 0
    assert ir['frozen_seeds'] == 3 and ir['scratch'] == 1111 and not ir['free_e']
    result = dict(node=NODE, input_names=NAMES, free_e=False,
                  sampled_four_op=dict(cases=14208, stages=scout_stages,
                                      counts=dict(Counter(r['status'] for r in sr.values())), remaining_sampled=0,
                                      not_complete_four_op_grammar=True),
                  complete_affine_tail=dict(cases=4200, stages=stages, counts=dict(counts),
                                           certificates=certificates, remaining=0,
                                           scope='MAD(any8live), SHR16(t), XOR(q,initial/t/privateK), affine MAD'),
                  independent_high_half_sources=8, evidence_sha256=evidence,
                  decoded_sources_sha256=decoded_hashes,
                  best_executable_cycles=864, new_executable=False,
                  scope='The sampled manifest and stated complete affine-tail subfamily only. No all-four-op or global kernel bound.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Sampled14208 closed; affine-tail4200 closed:', dict(counts), flush=True)


if __name__ == '__main__':
    main()
