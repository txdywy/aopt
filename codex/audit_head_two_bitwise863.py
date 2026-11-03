"""Classify the mixed two-bitwise coarse scan without hiding UNKNOWN."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

from codex.scan_head_two_bitwise863 import cases
from codex.screen_head_square863 import certificate
from codex.search_cross_encoding863 import C1

ROOT=Path(__file__).resolve().parent


def audit():
    evidence={}
    def read(name):
        path=ROOT/name;evidence[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    raw=read('head_two_bitwise863_results.json')
    one=read('free_keys863_complete_grammar.json')
    assert one['excluded_orders']==64 and one['unresolved_orders']==0
    cert=certificate()
    assert cert['points']==[0,32,64,96] and cert['target_mixed_differences']==[1024]
    expected=set(cases()); seen=set(); grouped=defaultdict(list)
    for row in raw['results']:
        order=tuple(row['sequence']);case=order,row['first_form']
        assert case in expected and case not in seen
        assert row['width']==11 and row['fixed_input_key']==C1 and row['fixed_output_key'] is None
        assert row['control_program'] is None and not row['pin_original'] and not row['fixed_operands']
        assert all(row[k] is None for k in ('middle_source','tail_form','last_operands','second_form'))
        assert row['result'] in ('projected_grammar_unsat','unknown','equivalent_on_reduced_domain')
        grouped[order].append(row);seen.add(case)
    assert seen==expected
    order_statuses=[]
    for order,rows in sorted(grouped.items()):
        if all(row['result']=='projected_grammar_unsat' for row in rows):
            status,reason='closed','all first-MAD partitions directly UNSAT'
        elif order[0]!='mad' and order[1]!='mad':
            # On the certified points x and any bitwise word over x and
            # constants have disjoint horizontal/vertical bit changes;
            # the remaining integer-polynomial tail has zero mixed difference.
            status,reason='closed','bitwise-prefix polynomial rectangle, arbitrary output key'
        else:
            status,reason='open','coarse search has an unresolved or reduced-SAT partition'
        order_statuses.append(dict(sequence=list(order),status=status,reason=reason,
            raw_counts=dict(Counter(row['result'] for row in rows))))
    return dict(width=11,executable_best_cycles=864,new_executable=False,input_key=C1,
        output_key='arbitrary',raw_counts=dict(Counter(row['result'] for row in raw['results'])),
        total_queries=len(seen),order_statuses=order_statuses,total_orders=len(order_statuses),
        closed_orders=sum(row['status']=='closed' for row in order_statuses),
        open_orders=sum(row['status']=='open' for row in order_statuses),evidence_sha256=evidence,
        scope='Exactly two arbitrary MAD and two XOR/AND/OR operations with at least one '
              'AND/OR, one input and arbitrary constants/output XOR. No shifts or live inputs.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit();args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('raw_counts','total_orders','closed_orders','open_orders')}))


if __name__=='__main__':main()
