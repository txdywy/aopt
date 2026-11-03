"""Complete the narrowly specified four-op MAD/XOR head-absorption audit."""
import argparse
from collections import Counter
import hashlib
from itertools import combinations_with_replacement
import json
from pathlib import Path

from codex.audit_head_affine_frontier863 import audit as previous_audit
from codex.audit_head_nonlinear863 import audit as initial_audit
from codex.scan_head_final_quadratic863 import canonical_second, cases, fixed_roles, first_is_live
from codex.search_cross_encoding863 import C1

ROOT = Path(__file__).resolve().parent


def audit():
    evidence = {}

    def read(name):
        path = ROOT/name
        evidence[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    previous = previous_audit()
    assert read('head_affine_frontier863_audit.json') == previous
    assert previous['input_xor_branch_closed'] and previous['keyed_branch_closed']
    raw = read('head_final_quadratic863_results.json')
    assert raw['prior']['sha256'] == evidence['head_affine_frontier863_audit.json']
    expected = set(cases(previous))
    actual = set()
    for row in raw['results']:
        case = row['source'],tuple(row['second_operands']),tuple(row['last_operands'])
        assert case in expected and case not in actual
        r = row['raw']
        assert r['sequence'] == ['mad','mad','^','mad'] and r['width'] == 11
        assert r['first_form'] == 'affine' and r['fixed_input_key'] == C1
        assert r['fixed_output_key'] is None and not r['pin_original']
        assert r['control_program'] is None and not r['constant_multipliers_only']
        assert all(r[k] is None for k in ('middle_source','tail_form','last_operands','second_form'))
        assert r['fixed_operands'] == {str(k):list(v) for k,v in fixed_roles(*case).items()}
        assert row['result'] == r['result'] == 'projected_grammar_unsat'
        actual.add(case)
    assert actual == expected
    connections = Counter()
    for template in previous['remaining_final_templates']:
        source,last = template['source'],tuple(template['last_operands'])
        for pair in combinations_with_replacement(range(4),2):
            for c in range(4):
                kind,canonical = canonical_second((*pair,c))
                if not first_is_live(source,(*pair,c),last):
                    # Generic synthesis forbids unused intermediate results;
                    # these graphs belong to the already excluded three-op set.
                    connections['dead_first_three_operations'] += 1
                elif kind == 'affine' and source != 'first':
                    connections['affine_prefix_module'] += 1
                else:
                    assert (source,canonical,last) in actual
                    connections['fixed_quadratic_or_affine_query'] += 1
    square = [row for row in initial_audit()['first_families']
        if row['sequence'] == ['mad','mad','^','mad'] and row['first_form'] != 'affine']
    assert len(square) == 2 and all(row['status'] != 'open' for row in square)
    statuses = previous['order_statuses']
    for row in statuses:
        if row['sequence'] == ['mad','mad','^','mad']:
            row.update(status='closed',reason='110 complete second-MAD refinements UNSAT; '
                'affine second terms in input/constant XOR branches covered by exact modules')
    assert all(row['status'] == 'closed' for row in statuses)
    return dict(executable_best_cycles=864,new_executable=False,width=11,
        fixed_input_key=C1,free_output_key=True,direct_refinement_queries=len(actual),
        live_refinement_queries=sum(first_is_live(*case) for case in actual),
        dead_first_refinement_queries=sum(not first_is_live(*case) for case in actual),
        remaining_template_second_roles=sum(connections.values()),
        second_role_coverage=dict(connections),order_statuses=statuses,
        closed_orders=len(statuses),open_orders=0,evidence_sha256=evidence,
        scope=previous['scope'],
        result='all_ten_specified_mad_xor_orders_excluded',
        excluded_from_claim='Other opcode counts, AND/OR/shift and other instructions, '
            'extra live inputs, whole-hash changes, layouts, kernel-wide optimum. '
            'No lower-than-864 executable was found.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = audit()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('result','direct_refinement_queries',
        'second_role_coverage','closed_orders','open_orders')}))


if __name__ == '__main__':
    main()
