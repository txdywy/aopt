"""Audit all affine-first MAD/XOR/XOR/MAD connections and their reductions."""
import argparse
from collections import Counter
import hashlib
from itertools import combinations_with_replacement, product
import json
from pathlib import Path

from codex.audit_head_nonlinear863 import audit as initial_audit
from codex.audit_head_polynomial863 import audit as polynomial_audit
from codex.audit_keyed_live_nonlinear863 import audit as live_audit
from codex.scan_head_xor_pairs863 import PAIRS, fixed_roles, last_roles
from codex.search_cross_encoding863 import C1

ROOT = Path(__file__).resolve().parent


def classify(second, third, last):
    # XOR expressions are exact masks over independent symbols x,t,K,L,J,N.
    u = (1, 2, 4, 8)[second[0]] ^ (1, 2, 4, 8)[second[1]]
    v = (1, 2, u, 16, 32)[third[0]] ^ (1, 2, u, 16, 32)[third[1]]
    ur, vr = u & 3, v & 3
    if 3 not in last:
        return 'three_operations', None
    if 2 not in last:
        return ('keyed_reordering' if vr == 3 and v & ~3 else 'three_operations'), None
    if not ur or not vr or v in (1, 2, u):
        return 'three_operations', None
    if ur == 1 or vr == 1:
        return 'keyed_reordering', None
    swap = False
    if ur == vr == 3:
        family = 'xor_then_key'
    elif ur == vr == 2:
        family = 'parallel_keys'
    elif ur == 3 and vr == 2:
        family = 'parallel_input_key'
    elif ur == 2 and vr == 3:
        if v & ~3:
            family = 'key_then_xor'
        else:
            family, swap = 'parallel_input_key', True
    else:
        raise AssertionError((second, third, last, u, v))
    roles = tuple(5-i if swap and i in (2, 3) else min(i, 4) for i in last)
    return family, (*sorted(roles[:2]), roles[2])


def audit():
    evidence = {}

    def read(name):
        p = ROOT / name
        evidence[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text())

    raw = read('head_xor_pairs863_results.json')
    previous = polynomial_audit()
    assert read('head_polynomial863_audit.json') == previous
    live = live_audit()
    assert read('keyed_live863_nonlinear_audit.json') == live
    assert live['unresolved_within_three_orders'] == 0
    single = read('free_keys863_complete_grammar.json')
    assert single['unresolved_orders'] == 0 and single['excluded_orders'] == 64
    squares = [r for r in initial_audit()['first_families']
        if r['sequence'] == ['mad', '^', '^', 'mad'] and r['first_form'] != 'affine']
    assert len(squares) == 2 and all(r['status'] != 'open' for r in squares)
    expected = set(product(PAIRS, last_roles()))
    results = {}
    for row in raw['results']:
        case = row['family'], tuple(row['last_operands'])
        assert case in expected and case not in results
        r = row['raw']
        assert r['sequence'] == ['mad', '^', '^', 'mad'] and r['width'] == 11
        assert r['fixed_input_key'] == C1 and r['fixed_output_key'] is None
        assert r['first_form'] == 'affine' and r['control_program'] is None and not r['pin_original']
        assert r['middle_source'] is None and r['tail_form'] is None
        assert r['last_operands'] is None and r['second_form'] is None
        assert r['fixed_operands'] == {str(k): list(v) for k, v in fixed_roles(*case).items()}
        assert row['result'] == r['result'] == 'projected_grammar_unsat'
        results[case] = r['result']
    assert set(results) == expected
    classes = Counter()
    for second, third, pair, c in product(combinations_with_replacement(range(4), 2),
            combinations_with_replacement(range(5), 2),
            combinations_with_replacement(range(6), 2), range(6)):
        category, role = classify(second, third, (*pair, c))
        classes[category] += 1
        if role is not None:
            assert (category, role) in results
    statuses = previous['order_statuses']
    for row in statuses:
        if row['sequence'] == ['mad', '^', '^', 'mad']:
            row.update(status='closed', reason='complete affine XOR-pair classification and '
                '52 fixed-DAG UNSAT queries; earlier square-first exclusions')
    return dict(executable_best_cycles=864, new_executable=False, width=11,
        direct_queries=len(results), affine_wiring_patterns=sum(classes.values()),
        affine_wiring_classes=dict(classes), order_statuses=statuses,
        closed_orders=sum(row['status'] == 'closed' for row in statuses),
        open_orders=sum(row['status'] == 'open' for row in statuses),
        evidence_sha256=evidence,
        reduction='If only one XOR output survives, fold constants/cancel XOR terms into '
            'three operations or move x XOR K before the affine MAD. If either XOR word '
            'is x XOR K, move it first and use the complete keyed-live MAD/XOR/MAD proof. '
            'Otherwise both outputs must be operands of the last MAD; four templates suffice.',
        scope=previous['scope'],
        remaining='Other four-op orders, opcodes/live inputs, whole-hash and layout changes remain open.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('direct_queries', 'affine_wiring_patterns',
        'affine_wiring_classes', 'closed_orders', 'open_orders')}))


if __name__ == '__main__':
    main()
