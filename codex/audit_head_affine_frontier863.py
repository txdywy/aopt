"""Replay affine-prefix tables and audit exact module coverage and open DAGs."""
import argparse
from collections import Counter
import hashlib
from itertools import combinations_with_replacement, product
import json
from pathlib import Path

import numpy as np

from codex.audit_head_inner_product863 import audit as previous_audit
from codex.audit_head_nonlinear863 import audit as initial_audit
from codex.scan_head_keyed_module863 import canonical_pair
from codex.scan_head_final_product863 import XOR_ROLES, last_roles
from codex.scan_head_inner_product863 import roles
from codex.screen_affine_head_keys863 import cube_separated, decode, mixed_key_mask, offsets
from codex.search_cross_encoding863 import C1, middle, necessary_inputs

ROOT = Path(__file__).resolve().parent


def replay_tables(report):
    ids = np.arange(65536, dtype=np.int64)
    aa, bb = ids >> 8, ids & 255
    tables = []
    for residue, row in enumerate(report['residues']):
        assert row['residue'] == residue and row['width'] == 11 and not row['control']
        assert row['input_key'] == C1
        table = np.full((65536, 4), np.iinfo(np.uint64).max, dtype=np.uint64)
        for witness in row['witnesses']:
            base, axes = witness['base_y'], tuple(witness['axes_y'])
            assert 2 <= len(axes) <= 8 and len(set(axes)) == len(axes)
            assert all(v > 0 and v & (v-1) == 0 for v in axes)
            assert not base & sum(axes) and base+sum(axes) < 256
            assert 3*len(axes)+sum(v.bit_length()-1 for v in axes) >= 11
            values = [(middle((8*(base+d)+residue) ^ C1) >> 3) & 255 for d in offsets(axes)]
            assert witness['target_upper_values'] == values
            allowed = mixed_key_mask(values)
            assert witness['allowed_upper_key_mask'] == [f'{int(v):016x}' for v in allowed]
            selected = cube_separated(aa, bb, base, axes)
            table[selected] &= allowed
        np.testing.assert_array_equal(table, decode(row['table']))
        tables.append(table)
    assert len(tables) == 8
    first = np.flatnonzero(np.any(tables[0], axis=1))
    ids = np.repeat(first, 64)
    lifts = np.tile(np.arange(64, dtype=np.int64), len(first))
    a, b = (ids >> 8)+256*(lifts >> 3), 8*(ids & 255)+(lifts & 7)
    masks = tables[0][ids].copy()
    for r, table in enumerate(tables[1:], 1):
        masks &= table[((a & 255) << 8) | (((a*r+b) >> 3) & 255)]
    keep = np.any(masks, axis=1)
    pairs = np.column_stack((a[keep], b[keep]))
    np.testing.assert_array_equal(pairs, decode(report['surviving_pairs']))
    np.testing.assert_array_equal(masks[keep], decode(report['surviving_key_masks']))
    assert report['remaining_full_pairs'] == len(pairs)
    return pairs


def scaled_xor(role):
    a, b, c = role
    return (a == 3 and b >= 4 and (c == 3 or c >= 4)) or (a >= 4 and b >= 4 and c == 3)


def audit(require_keyed=True):
    evidence = {}

    def read(name):
        path = ROOT/name
        evidence[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    previous = previous_audit()
    assert read('head_inner_product863_audit.json') == previous
    rectangles = read('affine_head_cubes863_joint.json')
    pairs = replay_tables(rectangles)
    samples = list(dict.fromkeys(necessary_inputs(11)+[32, 64, 96]))
    module = read('head_bivariate863_results.json')
    assert module['width'] == 11 and module['degree'] == 4 and not module['control']
    assert module['samples'] == samples and module['all_canonical_pairs_tested']
    assert module['prior_evidence']['sha256'] == evidence['affine_head_cubes863_joint.json']
    assert module['accelerator_sha256'] == hashlib.sha256((ROOT/'screen_head_bivariate863.cpp').read_bytes()).hexdigest()
    normalized_input = {(int(a) & 1023, int(b) & 1023) for a, b in pairs}
    assert module['canonical_pairs'] == module['tested_pairs'] == len(normalized_input)
    assert module['result'] == 'sample_module_excluded' and module['survivors'] == []
    control = read('head_bivariate863_control.json')
    assert control['positive_witness_retained'] and control['control'] and control['degree'] == 4
    keyed_pairs = sorted({canonical_pair(int(a), int(b)) for a, b in pairs})
    keyed = []
    for shard in range(3):
        name = f'head_keyed_module863_shard_{shard}.json'
        if not (ROOT/name).exists() and not require_keyed:
            continue
        row = read(name)
        assert row['shard'] == shard and row['shards'] == 3
        assert row['width'] == 11 and row['degree'] == 4 and not row['control']
        assert row['all_shard_pairs_tested'] and row['samples'] == samples
        assert row['total_canonical_pairs'] == len(keyed_pairs)
        subset = keyed_pairs[shard::3]
        assert row['tested_pairs'] == len(subset)
        assert row['tested_triples'] == sum(256 if a % 2 == 0 else 512 for a, b in subset)
        assert row['prior_evidence']['sha256'] == evidence['affine_head_cubes863_joint.json']
        for source, fingerprint in row['source_sha256'].items():
            assert fingerprint == hashlib.sha256((ROOT/source).read_bytes()).hexdigest()
        keyed.append(row)
    keyed_complete = len(keyed) == 3 and all(row['result'] == 'sample_module_excluded'
        and not row['survivors'] for row in keyed)
    if keyed_complete:
        control = read('head_keyed_module863_control.json')
        assert control['control'] and control['positive_witness_retained']
        square = [row for row in initial_audit()['first_families']
            if row['sequence'] == ['mad','^','mad','mad'] and row['first_form'] != 'affine']
        assert len(square) == 2 and all(row['status'] != 'open' for row in square)

    def validate(raw, order, fixed):
        assert raw['sequence'] == order and raw['width'] == 11
        assert raw['fixed_input_key'] == C1 and raw['fixed_output_key'] is None
        assert raw['first_form'] == 'affine' and not raw['constant_multipliers_only']
        assert not raw['pin_original'] and raw['control_program'] is None
        assert all(raw[k] is None for k in ('middle_source', 'tail_form', 'last_operands', 'second_form'))
        assert raw['fixed_operands'] == {str(k): list(v) for k,v in fixed.items()}

    tail = read('head_polynomial_tail863_results.json')
    expected = set(product(('input', 'key'), roles()))
    seen = set()
    for row in tail['results']:
        case = row['source'], tuple(row['third_operands'])
        assert case in expected and case not in seen
        validate(row['raw'], ['mad','^','mad','mad'],
            {0: (0,1,2), 1: (0,1,0) if case[0] == 'input' else (1,2,0), 2: case[1]})
        assert row['result'] == row['raw']['result'] and row['result'] in ('projected_grammar_unsat','unknown')
        seen.add(case)
    assert seen == expected
    final = read('head_final_product863_results.json')
    scaled = read('head_polynomial863_scaled_quadratic_xor.json')
    assert scaled['family'] == 'scaled_quadratic_xor' and scaled['width'] == 11
    assert scaled['input_key'] == C1 and scaled.get('fixed_output_key') is None
    assert not scaled['control'] and scaled['result'] == 'relaxed_family_unsat'
    expected = set(product(XOR_ROLES, last_roles()))
    seen = set()
    classifications = Counter()
    open_templates = set()
    classified = {}
    for row in final['results']:
        source, role = row['source'], tuple(row['last_operands'])
        case = source, role
        assert case in expected and case not in seen
        validate(row['raw'], ['mad','mad','^','mad'],
            {0: (0,1,2), 2: XOR_ROLES[source], 3: role})
        assert row['result'] == row['raw']['result']
        assert row['result'] in ('projected_grammar_unsat','unknown')
        if row['result'] == 'projected_grammar_unsat':
            category = 'direct_unsat'
        elif scaled_xor(role):
            category = 'scaled_polynomial_xor_proof'
        else:
            category = 'unresolved_literal_roles'
            # Each remaining literal role uses at most one final constant.
            assert sum(v >= 4 for v in role) <= 1
            open_templates.add((source, tuple(min(v,4) for v in role)))
        classifications[category] += 1
        classified[case] = category
        seen.add(case)
    assert seen == expected
    wiring_classes = Counter()
    all_last = [(*pair,c) for pair in combinations_with_replacement(range(6),2) for c in range(6)]
    all_second = [(*pair,c) for pair in combinations_with_replacement(range(4),2) for c in range(4)]
    for second, third, last in product(all_second,
            combinations_with_replacement(range(5),2), all_last):
        if (3 not in last or third[0] == third[1] or third[0] >= 3 or
                (1 not in second and 1 not in third and 1 not in last)):
            wiring_classes['three_operations'] += 1
        elif 2 not in third:
            # The XOR is independent of the second MAD. Move it earlier,
            # yielding the complete MAD/XOR/MAD/MAD family just audited.
            wiring_classes['reordered_polynomial_tail' if keyed_complete else 'reordering_still_open'] += 1
        else:
            source = 'input' if third == (0,2) else 'first' if third == (1,2) else 'key'
            assert source != 'key' or third in ((2,3),(2,4))
            wiring_classes[classified[source,last]] += 1
    statuses = previous['order_statuses']
    for row in statuses:
        if row['sequence'] == ['mad','^','mad','mad'] and keyed_complete:
            row.update(status='closed', reason='complete affine-prefix degree-four module '
                'screens for t XOR x and t XOR K; earlier square-prefix and initial-XOR proofs')
    return dict(executable_best_cycles=864, new_executable=False, width=11,
        full_affine_pairs=1 << 22, rectangle_remaining_pairs=len(pairs),
        input_xor_module_pairs=len(normalized_input), input_xor_branch_closed=True,
        keyed_module_pairs=len(keyed_pairs), keyed_shards_complete=len(keyed),
        keyed_module_triples=sum(row['tested_triples'] for row in keyed),
        keyed_branch_closed=keyed_complete,
        raw_polynomial_tail_counts=dict(Counter(row['result'] for row in tail['results'])),
        raw_final_product_counts=dict(Counter(row['result'] for row in final['results'])),
        final_product_classifications=dict(classifications),
        final_product_wiring_patterns=sum(wiring_classes.values()),
        final_product_wiring_classes=dict(wiring_classes),
        remaining_final_templates=[dict(source=source,last_operands=list(role))
                                  for source,role in sorted(open_templates)],
        order_statuses=statuses, closed_orders=sum(row['status']=='closed' for row in statuses),
        open_orders=sum(row['status']=='open' for row in statuses),
        evidence_sha256=evidence, scope=previous['scope'],
        remaining='Remaining order/templates, other opcodes/live inputs, whole-hash and '
                  'layout changes remain open. This is not a whole-kernel optimality proof.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(not args.partial)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('input_xor_branch_closed','keyed_branch_closed',
        'keyed_module_triples','closed_orders','final_product_classifications','remaining_final_templates')}))


if __name__ == '__main__':
    main()
