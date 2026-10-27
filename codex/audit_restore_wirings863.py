"""Apply explicit structural certificates to restore-search timeouts.

No UNKNOWN is silently called UNSAT. Certificates either reduce a wiring to
the recorded <=3-op free-key grammar, use a genuine single-word information
cut, or use an exhaustive carry-free polynomial screen over all endpoint
keys. A first operation need not be injective when the original x bypasses it.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from codex.audit_free_key_frontier863 import audit as three_op_audit
from codex.screen_free_keys_polynomial863 import search as polynomial_screen
from codex.cegis_c1_absorb863 import middle


def input_path(order, wiring, node, avoid):
    """Whether x reaches node without passing through the avoided result."""
    if node == avoid:
        return False
    if node == 0:
        return True
    i = node-1
    return any(input_path(order, wiring, ref, avoid) for ref in wiring[i] if ref <= i)


def polynomial_prefix(order):
    mad_seen = False
    for op in order:
        if op == 'mad':
            mad_seen = True
        elif mad_seen:
            return False
    return True


def classify(row):
    order, wiring = row['order'], row['wiring']
    if polynomial_prefix(order):
        return 'bit_local_prefix_then_polynomial'
    if order[-1] == '^' and polynomial_prefix(order[:-1]):
        translated = {0}
        if order[0] == '^':
            translated.add(1)  # t0=x XOR a constant
        if translated.intersection(wiring[-1]):
            return 'polynomial_with_translated_input_xor_bypass'
    for i, op in enumerate(order):
        variable_args = [r for r in wiring[i] if r <= i]
        cut = not input_path(order, wiring, len(order), i+1)
        if cut and op in ('&', '|') and len(variable_args) == 1:
            return 'constant_mask_word_cut_reduces_to_three_ops'
    first_cut = not input_path(order, wiring, len(order), 1)
    if first_cut and order[0] == '^':
        return 'first_xor_word_cut_absorbed_in_free_input_key'
    if first_cut and order[0] == 'mad' and wiring[0] in ([0, 0, 0], [0, 0, 1]):
        return 'square_word_cut_noninjective'
    if order[-1] in ('&', '|'):
        if 0 in wiring[-1]:
            return 'balanced_lattice_with_original_x'
        if 1 in wiring[-1]:
            if order[0] == 'mad':
                return 'lattice_with_first_affine_or_square'
            if order[0] == '^':
                return 'balanced_lattice_with_xor_translation'
            if order[0] == order[-1]:
                return 'same_lattice_mask_must_be_neutral'
    return None


def certificates():
    old = three_op_audit()
    assert old['excluded_orders'] == 64 and old['unresolved_orders'] == 0
    screens = [polynomial_screen(11, carry_free=True, xor_input=x) for x in (False, True)]
    assert all(s['remaining_pairs'] == 0 for s in screens)
    control = polynomial_screen(6, carry_free=True, affine_control=True)
    assert control['zero_key_control_retained']
    xs = np.arange(2048, dtype=np.int64)
    fs = middle(xs, 2047)
    assert len(np.unique(fs)) == 2048
    assert np.all(fs[xs ^ 1024] == (fs ^ 1024))
    triangular = (xs*xs+xs) & 2047
    assert np.all(triangular == triangular[::-1])
    counts = [int(np.count_nonzero(triangular & (1 << j))) for j in range(11)]
    assert counts == [0]+[1024]*10
    diff = middle(0, 2047) ^ middle(2047, 2047)
    assert diff & ~1
    return dict(three_op_evidence_sha256=old['evidence_sha256'],
                three_op_excluded_orders=64, polynomial_screens=screens,
                polynomial_control=control,
                square_plus_x_bit_counts=counts,
                square_plus_x_complement_invariant=True,
                middle_complement_difference=diff,
                middle_permutation_domain=2048,
                middle_highest_input_bit_flip_verified=True,
                square_high_flip_witness=[0, 32],
                square_high_flip_delta=1024)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reports', type=Path, nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--refinements', type=Path)
    ap.add_argument('--polynomial84', type=Path)
    ap.add_argument('--sat-reports', type=Path, nargs='+')
    args = ap.parse_args()
    result = dict(certificates=certificates(), reports=[], closed=[], unresolved=[], new_executable=False)
    counts = Counter()
    for path in args.reports:
        data = json.loads(path.read_text())
        assert data['completed'] == data['jobs'], path
        result['reports'].append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                      jobs=data['jobs'], raw_counts=data['counts']))
        for row in data['unresolved']:
            if row['status'] != 'unknown':
                result['unresolved'].append(row)
                continue
            cert = classify(row)
            if cert:
                result['closed'].append(dict(order=row['order'], wiring=row['wiring'], certificate=cert))
                counts[cert] += 1
            else:
                result['unresolved'].append(row)
    result['closed_counts'] = dict(counts)
    if args.refinements:
        data = json.loads(args.refinements.read_text())
        assert data['completed'] == data['jobs']
        key = lambda r: (tuple(r['order']), tuple(tuple(w) for w in r['wiring']))
        source_keys = set()
        for p in data['source_reports']:
            source_keys.update(key(r) for r in json.loads(Path(p).read_text())['unresolved'])
        assert source_keys == {key(r) for r in result['unresolved']}
        assert len(source_keys) == data['jobs']
        assert set(data['counts']) <= {'unsat', 'unknown'}
        updated = {key(r): r for r in data['unresolved']}
        assert len(updated) == data['counts'].get('unknown', 0)
        keep = []
        for row in result['unresolved']:
            if key(row) in updated:
                keep.append(updated[key(row)])
            else:
                counts['refined_explicit_wiring_unsat'] += 1
                result['closed'].append(dict(order=row['order'], wiring=row['wiring'],
                                             certificate='refined_explicit_wiring_unsat'))
        result['unresolved'] = keep
        result['refinement_sha256'] = hashlib.sha256(args.refinements.read_bytes()).hexdigest()
    if args.polynomial84:
        data = json.loads(args.polynomial84.read_text())
        assert data['status'] == 'unsat' and not data['control'] and data['width'] == 11
        assert data['degrees'] == [8, 4] and data['family'] == 'xor' and data['bitop'] == '^'
        keep = []
        for row in result['unresolved']:
            if row['order'] == ['mad', 'mad', 'mad', '^']:
                counts['independent_degree8_xor_degree4_unsat'] += 1
                result['closed'].append(dict(order=row['order'], wiring=row['wiring'],
                                             certificate='independent_degree8_xor_degree4_unsat'))
            else:
                keep.append(row)
        result['unresolved'] = keep
        result['polynomial84_sha256'] = hashlib.sha256(args.polynomial84.read_bytes()).hexdigest()
    if args.sat_reports:
        key = lambda r: (tuple(r['order']), tuple(tuple(w) for w in r['wiring']))
        remaining = {key(r): r for r in result['unresolved']}
        closed_sat = {}
        candidates = set()
        result['sat_reports'] = []
        for p in args.sat_reports:
            data = json.loads(p.read_text())
            assert data['completed'] == data['jobs'] == len(data['rows'])
            source = json.loads(Path(data['source']).read_text())
            expected = {key(r) for r in source['unresolved'][:data['jobs']]}
            assert expected == {key(r) for r in data['rows']}
            assert expected <= set(remaining)
            result['sat_reports'].append(dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                               counts=data['counts']))
            for r in data['rows']:
                if r['status'] == 'unsat':
                    closed_sat.setdefault(key(r), data['engine'])
                elif r['status'] == 'reduced_sat':
                    candidates.add(key(r))
        assert not candidates.intersection(closed_sat), 'Solver disagreement on a reduced-domain candidate'
        for k, engine in closed_sat.items():
            row = remaining.pop(k)
            counts['external_cnf_unsat_'+engine] += 1
            result['closed'].append(dict(order=row['order'], wiring=row['wiring'],
                                         certificate='external_cnf_unsat_'+engine))
        result['unresolved'] = list(remaining.values())
        result['reduced_candidates'] = len(candidates)
    result['closed_counts'] = dict(counts)
    result['remaining'] = len(result['unresolved'])
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(closed_counts=dict(counts), remaining=result['remaining'])), flush=True)


if __name__ == '__main__':
    main()
