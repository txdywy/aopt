"""Keep four-op head absorption exclusions and genuinely open cases distinct."""
from collections import Counter
import argparse
import hashlib
from itertools import combinations_with_replacement,permutations
import json
from pathlib import Path

from codex.audit_keyed_live_nonlinear863 import audit as live_audit
from codex.scan_head_nonlinear863 import orders
from codex.screen_head_square863 import certificate
from codex.search_cross_encoding863 import C1

ROOT=Path(__file__).resolve().parent


def audit():
    evidence={}
    def read(name):
        path=ROOT/name;evidence[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    first=read('head_nonlinear863_first_forms.json')
    second=read('head_nonlinear863_second_roles.json')
    square=read('head_nonlinear863_square_certificate.json')
    assert square==certificate()
    assert second['refinement']['sha256']==evidence['head_nonlinear863_first_forms.json']
    expected={(order,form)for order in orders()for form in('affine','square_const','square_x')}
    coarse={};fine={}
    def check(row):
        assert row['width']==11 and row['fixed_input_key']==C1 and row['fixed_output_key']is None
        assert row['control_program']is None and not row['pin_original']
        assert row['middle_source']is None and row['last_operands']is None
        assert row['second_form']is None and row['tail_form']is None
        assert row['result']in('projected_grammar_unsat','unknown')
        return tuple(row['sequence']),row['first_form']
    for row in first['results']:
        case=check(row)
        assert case in expected and case not in coarse and not row['fixed_operands']
        coarse[case]=row
    assert set(coarse)==expected
    for row in second['results']:
        case=check(row);fixed=row['fixed_operands']
        assert coarse[case]['result']=='unknown'and set(fixed)=={'1'}
        role=tuple(fixed['1'])
        assert (case,role)not in fine
        fine[case,role]=row
    cases=[]
    unresolved_roles=0
    for case,row in sorted(coarse.items()):
        order,form=case
        status='closed_directly'
        counts={}
        if row['result']=='unknown':
            roles={(a,b,c)for a,b in combinations_with_replacement(range(4),2)
                   for c in(range(4)if order[1]=='mad'else(0,))}
            actual={role for coarse_case,role in fine if coarse_case==case}
            assert roles==actual
            counts=dict(Counter(fine[case,role]['result']for role in roles))
            if counts.get('unknown',0)==0:status='closed_by_second_role_partition'
            elif form in('square_const','square_x')and order==('mad','^','mad','mad'):
                status='closed_by_square_rectangle'
            else:
                status='open';unresolved_roles+=counts['unknown']
        cases.append(dict(sequence=list(order),first_form=form,status=status,second_role_counts=counts))
    live=live_audit()
    assert live['unresolved_within_three_orders']==0
    read('keyed_live863_nonlinear_audit.json')
    rectangles=read('keyed_live863_rectangles.json')
    assert any(r['bypass']is None and r['remaining_output_keys']==0 and r['width']==11
               for r in rectangles['experiments'])
    all_orders=sorted({p for source in(('mad','mad','mad','^'),('mad','mad','^','^'))
                       for p in permutations(source)})
    statuses=[]
    for order in all_orders:
        if order[0]=='mad':
            relevant=[r for r in cases if tuple(r['sequence'])==order]
            status='closed'if all(r['status']!='open'for r in relevant)else'open'
            reason='first-MAD partitions'
        elif order in(('^','mad','mad','mad'),('^','^','mad','mad')):
            status,reason='closed','bitwise-prefix polynomial rectangle, arbitrary output key'
        elif order==('^','mad','^','mad'):
            status,reason='closed','new live MAD/XOR/MAD proof after x XOR K; extra prior input is unused'
        else:status,reason='open','initial XOR followed by two MADs and final XOR is outside these partitions'
        statuses.append(dict(sequence=list(order),status=status,reason=reason))
    return dict(executable_best_cycles=864,new_executable=False,width=11,
        input_key=C1,free_output_key=True,first_partition_counts=dict(Counter(r['result']for r in first['results'])),
        second_partition_counts=dict(Counter(r['result']for r in second['results'])),
        first_families=cases,remaining_second_role_unknowns=unresolved_roles,
        order_statuses=statuses,closed_orders=sum(r['status']=='closed'for r in statuses),
        open_orders=sum(r['status']=='open'for r in statuses),evidence_sha256=evidence,
        scope='One input, arbitrary constants and output XOR key; four instructions with '
              'three MAD/one XOR or two MAD/two XOR. Only listed exclusions are proved.',
        remaining='Five affine-first families and XOR/MAD/MAD/XOR stay open, alongside '
                  'other opcodes, other live inputs, whole-hash and layout changes.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    report=audit();args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k]for k in('first_partition_counts','second_partition_counts',
        'remaining_second_role_unknowns','closed_orders','open_orders')}))


if __name__=='__main__':main()
