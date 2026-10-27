"""Re-audit mixed bitwise orders with first-bitwise operand refinements."""
import argparse
from collections import Counter,defaultdict
import hashlib
from itertools import combinations_with_replacement
import json
from pathlib import Path

from codex.audit_head_two_bitwise863 import audit as previous_audit
from codex.search_cross_encoding863 import C1

ROOT=Path(__file__).resolve().parent


def audit():
    evidence={}
    def read(name):
        path=ROOT/name;evidence[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    previous=previous_audit()
    assert read('head_two_bitwise863_audit.json')==previous
    coarse=read('head_two_bitwise863_results.json')
    fine=read('head_two_bitwise863_refinement.json')
    assert fine['prior']['sha256']==evidence['head_two_bitwise863_results.json']
    orders={tuple(row['sequence']) for row in coarse['results'] if row['result']=='unknown'
            and row['first_form']=='affine' and row['sequence'][0]=='mad'
            and row['sequence'][1]!='mad' and row['sequence'][2]!='mad' and row['sequence'][3]=='mad'}
    expected={(order,(*pair,0)) for order in orders for pair in combinations_with_replacement(range(4),2)}
    seen=set(); grouped=defaultdict(list)
    unknown_classes=set();unknown_reductions=Counter()
    for row in fine['results']:
        order=tuple(row['sequence']);second=tuple(row['fixed_operands']['1']);case=order,second
        assert case in expected and case not in seen
        assert row['width']==11 and row['fixed_input_key']==C1 and row['fixed_output_key'] is None
        assert row['first_form']=='affine' and not row['pin_original'] and row['control_program'] is None
        assert set(row['fixed_operands'])=={'1'}
        assert all(row[k] is None for k in ('middle_source','tail_form','last_operands','second_form'))
        assert row['result'] in ('projected_grammar_unsat','unknown','equivalent_on_reduced_domain')
        if row['result']=='unknown':
            a,b,_=second
            if a==b or a>=2:
                # Bitwise equal operands are an identity (AND/OR) or zero
                # (XOR), and constant-only operands yield a constant.
                unknown_reductions['three_operations']+=1
            else:
                unknown_reductions['still_open']+=1
                unknown_classes.add((order,(a,min(b,2),0)))
        seen.add(case);grouped[order].append(row)
    assert seen==expected
    statuses=previous['order_statuses']
    for row in statuses:
        order=tuple(row['sequence'])
        if order in orders:
            rows=grouped[order]
            assert len(rows)==10
            row['refinement_counts']=dict(Counter(r['result'] for r in rows))
            if all(r['result']=='projected_grammar_unsat' for r in rows):
                row.update(status='closed',reason='ten fixed first-bitwise operand pairs UNSAT; '
                    'previous square-first partitions and three-op dead-code proof')
    return dict(width=11,executable_best_cycles=864,new_executable=False,input_key=C1,
        output_key='arbitrary',refinement_queries=len(seen),
        raw_refinement_counts=dict(Counter(row['result'] for row in fine['results'])),
        unknown_reductions=dict(unknown_reductions),
        remaining_distinct_refinement_classes=[dict(sequence=list(order),second_operands=list(role))
            for order,role in sorted(unknown_classes)],
        order_statuses=statuses,total_orders=len(statuses),
        closed_orders=sum(row['status']=='closed' for row in statuses),
        open_orders=sum(row['status']=='open' for row in statuses),evidence_sha256=evidence,
        scope=previous['scope'],
        note='Synthesis requires live intermediate results; dead instructions are '
             'covered by the complete three-operation proof referenced in the earlier audit.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit();args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('raw_refinement_counts','total_orders','closed_orders','open_orders')}))


if __name__=='__main__':main()
