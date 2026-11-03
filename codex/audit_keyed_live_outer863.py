"""Check evidence and complete wiring coverage for live MAD/MAD/XOR.

The live pool and low-a restriction are exactly those of the search. This
does not exclude other instruction orders or prove the whole kernel optimal.
"""
from collections import Counter
import argparse
import hashlib
from itertools import combinations_with_replacement, product
import json
from pathlib import Path

from codex.scan_keyed_live_nonlinear863 import canonical_first, first_forms

ROOT = Path(__file__).resolve().parent


def wiring_class(last):
    if last[0] == last[1]:
        return 'constant'
    if last == (3, 4):
        return 'two_mad_outputs'
    if any(v >= 5 for v in last):
        return 'polynomial_xor_constant'
    if 0 in last or 2 in last:
        return 'polynomial_xor_a'
    if 1 in last:
        return 'polynomial_xor_prior'
    raise AssertionError(last)


def audit():
    evidence = {}

    def read(name):
        path = ROOT / name
        evidence[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    coarse = read('keyed_live863_outer_first_forms.json')
    fine = read('keyed_live863_outer_second_roles.json')
    one = read('free_keys863_complete_grammar.json')
    rectangles = read('keyed_live863_rectangles.json')
    assert one['excluded_orders'] == 64 and one['unresolved_orders'] == 0
    assert fine['refinement']['sha256'] == evidence['keyed_live863_outer_first_forms.json']
    assert fine['refinement']['single_input_sha256'] == evidence['free_keys863_complete_grammar.json']
    assert {r['bypass'] for r in rectangles['experiments']
            if r['width'] == 11 and r['remaining_output_keys'] == 0} == {None, 'a', 'prior'}
    expected = {('a', None), ('keyed', None)} | {(None, role) for _, role in first_forms()}

    def validate(row):
        assert row['sequence'] == ['mad', 'mad', '^'] and row['width'] == 11
        assert row['last_operands'] == [3, 4, 0] and row['middle_source'] is None
        assert row['fixed_key'] is None and row['arbitrary_keys']
        assert not any(row[k] for k in ('control', 'control_program', 'pin_control', 'pin_original'))
        return row['first_source'], tuple(row['first_operands']) if row['first_operands'] else None

    seen = {}
    for row in coarse['results']:
        case = validate(row)
        assert case in expected and case not in seen and row.get('second_operands') is None
        assert row['result'] in ('necessary_subdomain_unsat', 'unknown')
        seen[case] = row['result']
    assert set(seen) == expected
    assert {case for case, status in seen.items() if status == 'unknown'} == {('keyed', None)}
    all_second = {(a, b, c) for a, b in combinations_with_replacement(range(6), 2) for c in range(6)}
    one_input = {role for role in all_second if not set(role).intersection((0, 1))}
    assert {tuple(role) for role in fine['second_roles_covered_by_single_input_proof']} == one_input
    refined = set()
    for row in fine['results']:
        assert validate(row) == ('keyed', None)
        role = tuple(row['second_operands'])
        assert role in all_second - one_input and role not in refined
        assert row['result'] == 'necessary_subdomain_unsat'
        refined.add(role)
    assert refined == all_second - one_input
    classes = Counter()
    for pair, c, second, last in product(combinations_with_replacement(range(5), 2),
            range(5), sorted(all_second), combinations_with_replacement(range(7), 2)):
        first = (*pair, c)
        category = wiring_class(last)
        classes[category] += 1
        if category == 'two_mad_outputs':
            kind, role = canonical_first(first)
            assert kind == 'affine' or (None, role) in expected
            # At width 11 prior=a-C0, so affine prior is contained in affine a.
            # Constant-only and repeated-variable first MADs are also affine.
            if kind == 'affine':
                assert ('a', None) in expected and ('keyed', None) in expected
                assert second in refined | one_input
    return dict(width=11, executable_best_cycles=864, new_executable=False,
        coarse_counts=dict(Counter(seen.values())), refinement_queries=len(refined),
        one_input_second_roles=len(one_input), unresolved_within_order=0,
        closed_order=['mad', 'mad', '^'], wiring_patterns=sum(classes.values()),
        wiring_classes=dict(classes), evidence_sha256=evidence,
        scope='Live inputs a, prior=(a-C0)/4097, a XOR (a>>19) XOR K; arbitrary K,O; '
              'target F(a XOR (a>>19) XOR C1) XOR O. Exactly MAD/MAD/XOR with arbitrary '
              'constant and runtime operands, including dead code and aliases.',
        reduction='Final XOR with keyed is final XOR with a plus a constant on the legal '
                  'low-a domain. Other final initial words use the recorded rectangle '
                  'screens; two MAD results use the complete first/second partitions.',
        excluded_from_claim='Other orders/opcodes, additional live words, whole-kernel optimality.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('coarse_counts', 'refinement_queries',
        'one_input_second_roles', 'wiring_patterns', 'wiring_classes')}))


if __name__ == '__main__':
    main()
