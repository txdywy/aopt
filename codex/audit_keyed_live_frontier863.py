"""Check the evidence for a bounded affine-first live-input frontier.

This does not close the general live-input grammar. In particular the first
MAD may not multiply two live values or combine two distinct live sources.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import argparse

from codex.scan_keyed_live_frontier863 import canonical_role,partitions


ROOT=Path(__file__).resolve().parent


def audit():
    evidence={}
    def read(name):
        path=ROOT/name
        evidence[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    previous=read('free_keys863_complete_grammar.json')
    assert previous['excluded_orders']==64 and previous['unresolved_orders']==0
    assert previous['minimum_instructions_over_endpoint_keys']==4
    even=read('keyed_live863_bypass_even.json')
    assert even['result']=='relaxed_family_unsat'and even['parity']==0 and even['width']==11
    refinements={source:read(f'keyed_live863_exact_bypass_{source}.json')for source in('a','prior')}
    for source,row in refinements.items():
        assert row['result']=='relaxed_family_unsat'and row['width']==11
        assert row['bypass_source']==source and row['parity']==1
    counts=Counter()
    for label,op,firsts in [('affine','^',('a','keyed')),('and','&',('keyed',)),('or','|',('keyed',))]:
        report=read(f'keyed_live863_{label}_key_frontier.json')
        assert report['width']==11
        expected=set(partitions(firsts))
        seen=set()
        for row in report['results']:
            pair=row['first_source'],tuple(row['last_operands'])
            assert pair in expected and pair not in seen
            seen.add(pair)
            if row['result']=='covered_by_prior_single_input_proof':
                assert pair[0]=='keyed'and not set(pair[1]).intersection((0,1))
                assert report['prior_single_input_evidence']['sha256']==evidence['free_keys863_complete_grammar.json']
            else:
                raw=row['raw']
                assert raw['width']==11 and raw['sequence']==['mad',op,'mad']
                assert not raw['control']and not raw['pin_original']and raw['fixed_key']is None
                assert raw['first_source']==pair[0]and tuple(raw['last_operands'])==pair[1]
                assert raw['middle_source']=='key'
                if raw['result']=='unknown':
                    assert op=='^'and pair[0]=='keyed'and pair[1]in((0,5,4),(1,5,4))
                else:assert raw['result']=='necessary_subdomain_unsat'
            counts[row['result']]+=1
        assert seen==expected
    role_set={role for _,role in partitions()}
    wiring_count=0
    for a in range(7):
        for b in range(a,7):
            for c in range(7):
                if 4 in(a,b,c):
                    assert canonical_role((a,b,c))in role_set
                    wiring_count+=1
    runtime_cases=0
    for first in('a','keyed'):
        for other in('a','prior','keyed'):
            row=read(f'keyed_live863_xor_{first}_{other}.json')
            assert row['result']=='necessary_subdomain_unsat'and row['width']==11
            assert row['sequence']==['mad','^','mad']and row['last_operands']is None
            assert row['first_source']==first and row['middle_source']==other
            assert not row['control']and row['fixed_key']is None
            runtime_cases+=1
    lattice=read('keyed_live863_affine_lattice.json')
    expected={(op,first,other)for op in('&','|')for first in('a','keyed')for other in('a','prior','keyed','key')}
    seen=set()
    for row in lattice:
        case=row['sequence'][1],row['first_source'],row['middle_source']
        assert case in expected and case not in seen
        seen.add(case)
        assert row['width']==11 and row['sequence'][::2]==['mad','mad']
        assert not row['control']and row['fixed_key']is None and row['last_operands']is None
        if row['result']=='unknown':assert case[1:]==('keyed','key')
        else:assert row['result']=='necessary_subdomain_unsat'
    assert seen==expected
    rectangles=read('keyed_live863_rectangles.json')
    for row in rectangles['experiments']:
        assert row['width']==11 and row['initial_output_keys']==2048
        assert row['result']=='all_output_keys_excluded'and row['remaining_output_keys']==0
    for row in rectangles['controls']:assert row['zero_control_retained']
    assert {row['bypass']for row in rectangles['experiments']}=={None,'a','prior'}
    full=read('keyed_live863_full32_control.json')
    assert full['result']=='proved_equivalent_full32'and full['pin_original']and full['width']==32
    return dict(executable_best_cycles=864,new_executable=False,width=11,
        constant_middle_partitions=sum(counts.values()),raw_partition_counts=dict(counts),
        refined_unknowns=2,unresolved_within_this_family=0,
        last_mad_operand_wirings_before_constant_relaxation=wiring_count,
        canonical_last_mad_roles=50,runtime_xor_cases=runtime_cases,
        lattice_coarse_cases=len(lattice),evidence_sha256=evidence,
        first_prior_reduction='prior=a-C0 mod 2048; absorb -A*C0 into the arbitrary first offset B.',
        scope='t=A*s+B with s in {a,prior,keyed}; q=t OP u with OP in {XOR,AND,OR}, '
              'u an initial live source or arbitrary constant; final MAD has arbitrary live operands and two constants. '
              'Input/output XOR keys are free. A bitwise prefix followed by arbitrary integer polynomials is separately excluded.',
        remaining='No general live-input minimum: mixed-source or nonlinear first MADs, different orders, '
                  'shifts and other opcodes, larger full-hash rewrites, record-layout changes remain outside this audit.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    report=audit();args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k]for k in('constant_middle_partitions','raw_partition_counts','refined_unknowns','unresolved_within_this_family')}))


if __name__=='__main__':main()
