"""Extend the previous head audit using new polynomial and live-output proofs."""
import argparse
import hashlib
import json
from pathlib import Path

from codex.audit_head_nonlinear863 import audit as previous_audit
from codex.audit_keyed_live_outer863 import audit as outer_audit
from codex.search_cross_encoding863 import C1, PRE_C5

ROOT = Path(__file__).resolve().parent


def audit():
    previous = previous_audit()
    outer = outer_audit()
    assert outer['unresolved_within_order'] == 0
    evidence = {}

    def read(name):
        path = ROOT / name
        evidence[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    assert read('keyed_live863_outer_audit.json') == outer
    results = []
    for suffix, input_key, output_key, families in (
            ('', C1, None, ('polynomial_xor', 'scaled_quadratic_xor')),
            ('c5_', 0, PRE_C5, ('polynomial_xor', 'scaled_quadratic_xor', 'bypass')),
            ('both_', C1, PRE_C5, ('bypass',))):
        for family in families:
            name = 'head_polynomial863_'+suffix+family+'.json'
            row = read(name)
            actual_family = 'quadratic_xor_bypass' if family == 'bypass' else family
            assert row['family'] == actual_family and row['width'] == 11
            # The first two records predate optional fixed-output and pin flags;
            # that original API searched a free output key without pinning.
            assert row['input_key'] == input_key and row.get('fixed_output_key') == output_key
            assert not row['control'] and not row.get('pin_control', False)
            assert row['result'] == 'relaxed_family_unsat'
            results.append(dict(file=name, family=actual_family, input_key=input_key,
                                output_key=output_key, result=row['result']))
    statuses = previous['order_statuses']
    for row in statuses:
        order = tuple(row['sequence'])
        if order == ('mad', 'mad', 'mad', '^'):
            # A first affine MAD implies degrees <=1, <=2, <=4. The final
            # XOR has at most one degree-four arm; its other arm is <=2.
            square = [case for case in previous['first_families']
                if tuple(case['sequence']) == order and case['first_form'] != 'affine']
            assert len(square) == 2 and all(case['status'] != 'open' for case in square)
            row.update(status='closed', reason='new degree-four XOR degree-two relaxation '
                       'excludes the affine-first case; prior square-first exclusions remain valid')
        elif order == ('^', 'mad', 'mad', '^'):
            row.update(status='closed', reason='new complete live MAD/MAD/XOR proof after x XOR K; '
                       'unused prior input only enlarges the candidate set')
    open_orders = [row['sequence'] for row in statuses if row['status'] == 'open']
    return dict(executable_best_cycles=864, new_executable=False, width=11,
        input_key=C1, free_output_key=True, order_statuses=statuses,
        closed_orders=len(statuses)-len(open_orders), open_orders=len(open_orders),
        remaining_orders=open_orders, new_family_exclusions=results,
        evidence_sha256=evidence, previous_evidence_sha256=previous['evidence_sha256'],
        scope=previous['scope'],
        partial_MM_X_M='With affine first MAD, constant scaling/offset of the XOR is excluded '
            'by scaled_quadratic_xor. Runtime multipliers and arbitrary polynomial bypass '
            'remain outside that exclusion.',
        open_relaxation='(affine XOR quadratic)+degree-four polynomial with free output XOR '
            'remains UNKNOWN in the direct, interpolated, and narrower-AND formulations.',
        remaining='Four four-op orders remain open, along with other opcodes/live inputs, '
            'whole-hash transformations and record layouts. No global optimum claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('closed_orders', 'open_orders', 'remaining_orders')}))


if __name__ == '__main__':
    main()
