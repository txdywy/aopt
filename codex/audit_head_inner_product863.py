"""Audit MAD/XOR/MAD/XOR, retaining raw UNKNOWN results and their refinements."""
import argparse
from collections import Counter
import hashlib
from itertools import combinations_with_replacement, product
import json
from pathlib import Path

from codex.audit_head_nonlinear863 import audit as initial_audit
from codex.audit_head_xor_pairs863 import audit as previous_audit
from codex.audit_keyed_live_outer863 import audit as outer_audit
from codex.scan_head_inner_product863 import roles
from codex.search_cross_encoding863 import C1

ROOT = Path(__file__).resolve().parent
ORDER = ['mad', '^', 'mad', '^']
POLYNOMIAL_ROLES = {(1, 1, 3), (1, 3, 3), (1, 3, 4)}
SCALED_ROLES = {(2, 3, 4), (2, 4, 3)}
PRODUCT_ROLE = (1, 2, 4)


def classify(second, third, last):
    """Classify literal connections, including dead code and constant aliases."""
    if last[0] == last[1] or 3 not in last or last[1] >= 4:
        # A final XOR with a constant is absorbed in the arbitrary output key.
        return 'three_operations', None
    if 2 not in third and 2 not in last:
        return 'three_operations', None
    a, b = second
    if a == b or a >= 2:
        return 'three_operations', None
    if a == 0 and b >= 2:
        return 'keyed_reordering', None
    source = 'input' if (a, b) == (0, 1) else 'key'
    assert source == 'input' or a == 1 and b >= 2
    return 'fixed_third_query', (source, third, last[0])


def audit():
    evidence = {}

    def read(name):
        path = ROOT / name
        evidence[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    previous = previous_audit()
    assert read('head_xor_pairs863_audit.json') == previous
    outer = outer_audit()
    assert read('keyed_live863_outer_audit.json') == outer
    assert outer['unresolved_within_order'] == 0
    single = read('free_keys863_complete_grammar.json')
    assert (single['excluded_orders'], single['unresolved_orders']) == (64, 0)
    polynomial = read('head_polynomial863_polynomial_xor.json')
    assert polynomial['family'] == 'polynomial_xor' and polynomial['width'] == 11
    assert polynomial['input_key'] == C1 and polynomial.get('fixed_output_key') is None
    assert not polynomial['control'] and not polynomial.get('pin_control', False)
    assert polynomial['result'] == 'relaxed_family_unsat'
    for name in ('input', 'first', 'middle'):
        row = read('head_merge863_family_key_'+name+'.json')
        assert row['width'] == 11 and row['input_key'] == C1
        assert row['family'] == 'key_'+name and row['free_output_key']
        assert not row['control'] and row['result'] == 'projected_family_unsat'

    def validate(raw, fixed):
        assert raw['sequence'] == ORDER and raw['width'] == 11
        assert raw['fixed_input_key'] == C1 and raw['fixed_output_key'] is None
        assert raw['first_form'] == 'affine'
        assert not raw['pin_original'] and raw['control_program'] is None
        assert not raw['constant_multipliers_only']
        assert raw['two_arbitrary_constants_per_instruction']
        for field in ('middle_source', 'tail_form', 'last_operands', 'second_form'):
            assert raw[field] is None
        assert raw['fixed_operands'] == {str(k): list(v) for k, v in fixed.items()}

    raw = read('head_inner_product863_results.json')
    expected = set(product(('input', 'key'), roles()))
    results = {}
    for row in raw['results']:
        case = row['source'], tuple(row['third_operands'])
        assert case in expected and case not in results
        fixed = {0: (0, 1, 2), 1: (0, 1, 0) if case[0] == 'input' else (1, 2, 0),
                 2: case[1]}
        validate(row['raw'], fixed)
        assert row['result'] == row['raw']['result']
        assert row['result'] in ('projected_grammar_unsat', 'unknown')
        results[case] = row['result']
    assert set(results) == expected
    unresolved = {case for case, status in results.items() if status == 'unknown'}
    assert unresolved == {('key', role) for role in POLYNOMIAL_ROLES | SCALED_ROLES | {PRODUCT_ROLE}}
    for tail in range(3):
        row = read(f'head_inner_product863_tail_{tail}.json')
        validate(row, {0: (0, 1, 2), 1: (1, 2, 0), 2: PRODUCT_ROLE, 3: (tail, 3, 0)})
        assert row['result'] == 'projected_grammar_unsat'
    squares = [row for row in initial_audit()['first_families']
               if row['sequence'] == ORDER and row['first_form'] != 'affine']
    assert len(squares) == 2 and all(row['status'] != 'open' for row in squares)

    classes = Counter()
    refinements = Counter()
    for second, third, last in product(combinations_with_replacement(range(4), 2),
            roles(), combinations_with_replacement(range(6), 2)):
        category, case = classify(second, third, last)
        classes[category] += 1
        if case is None:
            continue
        source, role, tail = case
        if results[source, role] != 'unknown':
            refinements['direct_unsat'] += 1
        elif role in POLYNOMIAL_ROLES:
            # u=t XOR K is not read by the third MAD. If the final XOR
            # reads u, its K folds into O, leaving P2 XOR P1. Otherwise
            # u is dead and the complete three-operation proof applies.
            assert 2 not in role
            assert tail == 2
            refinements['polynomial_xor'] += 1
        elif role in SCALED_ROLES:
            refinements['scaled_family'] += 1
        else:
            assert role == PRODUCT_ROLE and tail in range(3)
            refinements['fixed_final_xor'] += 1
    statuses = previous['order_statuses']
    for row in statuses:
        if row['sequence'] == ORDER:
            row.update(status='closed', reason='150 fixed-third queries; six raw UNKNOWN cases '
                'closed by polynomial/scaled-family proofs and three fixed-final queries; '
                'complete wiring enumeration and earlier square-first exclusions')
    return dict(executable_best_cycles=864, new_executable=False, width=11,
        coarse_query_counts=dict(Counter(results.values())), fixed_final_queries=3,
        affine_wiring_patterns=sum(classes.values()), affine_wiring_classes=dict(classes),
        live_query_classes=dict(refinements), order_statuses=statuses,
        closed_orders=sum(row['status'] == 'closed' for row in statuses),
        open_orders=sum(row['status'] == 'open' for row in statuses),
        remaining_orders=[row['sequence'] for row in statuses if row['status'] == 'open'],
        evidence_sha256=evidence, scope=previous['scope'],
        remaining='Other orders/opcodes/live inputs, whole-hash transformations and layouts '
                  'remain open. No whole-kernel optimality claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('coarse_query_counts',
        'affine_wiring_patterns', 'live_query_classes', 'closed_orders', 'remaining_orders')}))


if __name__ == '__main__':
    main()
