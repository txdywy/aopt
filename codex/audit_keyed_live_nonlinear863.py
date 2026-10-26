"""Audit all three MAD/bitwise/MAD orders with the specified live pool.

These bounded necessary conditions are not a minimum for the full hash or
the whole kernel. Every raw UNKNOWN is retained and independently refined.
"""
from collections import Counter
import argparse
import hashlib
import itertools
import json
from pathlib import Path

from codex.audit_keyed_live_frontier863 import audit as affine_audit
from codex.scan_keyed_live_frontier863 import canonical_role,partitions
from codex.scan_keyed_live_nonlinear863 import canonical_first,first_forms

ROOT=Path(__file__).resolve().parent


def wiring_class(first,middle,last):
    if 3 not in middle or middle[0]==middle[1]or 4 not in last:
        return 'bitwise_prefix_then_polynomial'
    return canonical_first(first)[0]


def audit():
    prior=affine_audit()
    assert prior['unresolved_within_this_family']==0
    evidence={}
    def read(name):
        path=ROOT/name
        evidence[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    coarse=read('keyed_live863_nonlinear.json')
    fine=read('keyed_live863_nonlinear_refinement.json')
    assert coarse['width']==fine['width']==11
    assert fine['refinement']['sha256']==evidence['keyed_live863_nonlinear.json']
    expected={(kind,first,op,source)for kind,first in first_forms()
              for op in ('^','&','|')for source in ('a','prior','keyed','key')}
    unresolved=set();seen=set();coarse_counts=Counter()
    def validate(row,role):
        case=row['kind'],tuple(row['first_operands']),row['bitwise'],row['middle_source']
        raw=row['raw']
        assert raw['sequence']==['mad',case[2],'mad']and raw['width']==11
        assert not raw['control']and raw['control_program']is None and not raw['pin_control']
        assert not raw['pin_original']and raw['fixed_key']is None
        assert tuple(raw['first_operands'])==case[1]and raw['first_source']is None
        assert raw['middle_source']==case[3]
        assert raw['last_operands']==(None if role is None else list(role))
        assert row['result']==raw['result']
        return case
    for row in coarse['results']:
        case=validate(row,None)
        assert case in expected and case not in seen
        seen.add(case);coarse_counts[row['result']]+=1
        if row['result']=='unknown':unresolved.add(case)
        else:assert row['result']=='necessary_subdomain_unsat'
    assert seen==expected
    roles={role for _,role in partitions(('a',))}
    fine_seen=set()
    for row in fine['results']:
        role=tuple(row['last_operands']);case=validate(row,role)
        assert case in unresolved and role in roles and (case,role)not in fine_seen
        assert row['result']=='necessary_subdomain_unsat'
        fine_seen.add((case,role))
    assert fine_seen==set(itertools.product(unresolved,roles))
    # A complete syntactic coverage check. Constant naming and dead code
    # intentionally remain; these are wiring counts, not distinct functions.
    classes=Counter()
    for first_pair in itertools.combinations_with_replacement(range(5),2):
        for c,middle,last_pair,z in itertools.product(range(5),
                itertools.combinations_with_replacement(range(6),2),
                itertools.combinations_with_replacement(range(7),2),range(7)):
            first=(*first_pair,c);last=(*last_pair,z)
            label=wiring_class(first,middle,last)
            classes[label]+=1
            if label in ('mixed','product'):
                kind,shape=canonical_first(first)
                other=middle[1]if middle[0]==3 else middle[0]
                source=('a','prior','keyed')[other]if other<3 else'key'
                assert canonical_role(last)in roles
                assert all((kind,shape,op,source)in seen for op in('^','&','|'))
    return dict(executable_best_cycles=864,new_executable=False,width=11,
        coarse_counts=dict(coarse_counts),refinement_queries=len(fine_seen),
        unresolved_within_three_orders=0,
        closed_orders=[['mad',op,'mad']for op in('^','&','|')],
        wiring_patterns_per_order=sum(classes.values()),wiring_classes=dict(classes),
        evidence_sha256=evidence,affine_evidence_sha256=prior['evidence_sha256'],
        scope='Live inputs a, prior=(a-C0)/4097, keyed=a XOR (a>>19) XOR K; arbitrary K,O; '
              'target F(a XOR (a>>19) XOR C1) XOR O; three operations MAD/bitwise/MAD. '
              'Any first and final MAD operands, including products of live words.',
        excluded_from_claim='Other instruction orders, additional live inputs, shifts or other '
              'opcodes, four or more operations, record layouts, whole-kernel optimality.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    report=audit();args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k]for k in('coarse_counts','refinement_queries',
        'unresolved_within_three_orders','wiring_patterns_per_order','wiring_classes')}))


if __name__=='__main__':main()
