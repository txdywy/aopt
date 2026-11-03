"""Check the finite coverage ledger for the 64 live-input unshifted orders.

This validates links between recorded solver/enumeration results and the
algebraic cases in SUB863_LIVE_AND_SHIFT_20261023.md. It is not a substitute
for the stated proofs or a global machine-checkable kernel lower bound.
"""
import argparse
import itertools
import json
from pathlib import Path

from codex.refine_live_middle_roles863 import canonical
from codex.search_live_middle_roles863 import roles

ROOT=Path(__file__).parent


def audit():
    def read(name):
        return json.loads((ROOT/name).read_text())
    structure=read('middle_live_structure863_results.json')
    single=read('middle_live_single_mad863_results.json')
    outer=read('middle_live_outer863_results.json')
    key=read('middle_live_constant_key863_results.json')
    initial=read('middle_live_roles863_results.json')
    refined=read('middle_live_roles863_refinement.json')
    residual=read('middle_live_residual863_results.json')
    controls=read('middle_live_roles863_controls.json')
    assert structure['prefix']['excluded'] and outer['excluded'] and key['excluded']
    assert residual['all_eight_residual_classes_excluded']
    assert len(single['rows'])==18 and all(r['target']['result']=='unsat' for r in single['rows'])
    assert all(r['positive_control']['result']=='sat' and
               r['positive_control'].get('verification')=='unsat' for r in single['rows'])
    assert len(controls)==3 and all(r['result']=='sat' and r.get('verification')=='unsat' for r in controls)
    raw={(r['sequence'][1],tuple(r['last_mad_role'])):r for r in initial['rows']}
    split={(r['sequence'][1],tuple(r['last_mad_role']),r['middle_other']):r for r in refined['rows']}
    assert len(raw)==174 and len(split)==52
    rule={('&',(0,6,5)):'AND mask forcing and two exact exceptions',
          ('|',(0,6,5)):'OR neutral-high-mask certificate',
          ('^',(0,4,5)):'b*t+u enumeration',
          ('^',(0,5,4)):'b*u+t enumeration',
          ('^',(0,6,5)):'D*b+u enumeration',
          ('^',(4,6,5)):'D*t+u injective first-result partition',
          ('^',(5,6,0)):'D*u+b low-bit and period certificate',
          ('^',(5,6,4)):'D*u+t injective first-result partition'}
    role_evidence=[]
    for op,role in itertools.product(('^','&','|'),roles()):
        row=raw[op,role]
        if row['result']=='unsat':
            evidence=['fixed final-role UNSAT']
        else:
            assert row['result']=='unknown'
            evidence=[]
            for source in (0,1,2,5):
                child=split[op,canonical(role),source]
                if child['result']=='unsat':
                    evidence.append('fixed middle-source UNSAT')
                else:
                    assert child['result']=='unknown' and source==5
                    evidence.append(rule[op,canonical(role)])
        role_evidence.append({'op':op,'role':role,'closed':True,'evidence':evidence})
    orders=[]
    singles={tuple(r['sequence']) for r in single['rows']}
    last={r['bitwise']:r['excluded'] for r in structure['last_bitwise']}
    for seq in itertools.product(('mad','^','&','|'),repeat=3):
        count=seq.count('mad')
        if count in (0,3) or count==1 and seq[2]=='mad' or count==2 and seq[0]!='mad':
            evidence='bitwise-prefix / integer-polynomial rectangle certificate'
        elif count==1:
            assert seq in singles
            evidence='fixed-order UNSAT with an independently verified unpinned positive control'
        elif seq[2]!='mad':
            assert last[seq[2]]
            evidence='complete first-MAD superset enumeration for final bitwise instruction'
        else:
            evidence='prefix/bypass reduction; sole-u lemma; 58 final roles, refinements and residual proofs'
        orders.append({'sequence':seq,'excluded':True,'evidence':evidence})
    assert len(orders)==64
    return {'best_verified_cycles':864,'new_executable_candidate':False,
            'closed_orders':len(orders),'orders':orders,
            'fixed_final_role_results':initial['counts'],
            'middle_source_refinement_results':refined['counts'],
            'final_roles':role_evidence,'positive_controls_verified':21,
            'scope':'Middle F(b), three operations from MAD/XOR/AND/OR, arbitrary constants; '
                    'extra live prior,a,shift=a>>19,mix=a XOR C1,b=shift XOR mix. '
                    'Exclusion uses the valid full32 subdomain a<2**w, w=11 or 12. '
                    'It does not include other live inputs, shifts, division, selection, '
                    'memory, cross-lane operations, or an entire-kernel optimality claim.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=audit()
    with open(args.output,'w') as output:
        json.dump(result,output,indent=2)
        output.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('orders','final_roles')}))


if __name__=='__main__':
    main()
