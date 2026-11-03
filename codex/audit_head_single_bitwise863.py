"""Audit exactly three MADs plus one AND/OR for fixed C1 absorption."""
import argparse
from collections import Counter
import hashlib
from itertools import combinations_with_replacement,product
import json
from pathlib import Path

from codex.audit_head_affine_frontier863 import replay_tables
from codex.audit_head_four_xor863 import audit as xor_audit
from codex.scan_head_and_final863 import SOURCES,last_roles
from codex.scan_head_final_quadratic863 import canonical_second,quadratic_roles
from codex.scan_head_keyed_module863 import canonical_pair
from codex.scan_head_single_bitwise863 import cases
from codex.screen_affine_head_keys863 import decode
from codex.screen_head_square863 import certificate
from codex.search_cross_encoding863 import C1,necessary_inputs

ROOT=Path(__file__).resolve().parent


def audit():
    evidence={}
    def read(name):
        path=ROOT/name;evidence[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    xor=xor_audit()
    assert read('head_four_xor863_audit.json')==xor
    prior=read('affine_head_cubes863_joint.json')
    pairs=replay_tables(prior)
    samples=list(dict.fromkeys(necessary_inputs(11)+[32,64,96]))
    source_hash={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                 for name in ('screen_head_and_module863.cpp','screen_head_bivariate863.cpp')}

    def module(name,kind):
        row=read(name)
        assert row['width']==11 and row['degree']==4 and row['source']==kind and not row['control']
        assert row['samples']==samples and row['source_sha256']==source_hash
        assert row['prior_evidence']['sha256']==evidence['affine_head_cubes863_joint.json']
        assert row['result']=='sample_module_excluded' and row['remaining_triples']==0 and not row['survivors']
        return row

    input_module=module('head_and_module863_input.json','input')
    assert input_module['all_pairs_tested'] and input_module['tested_pairs']==len(pairs)
    assert input_module['tested_triples']==len(pairs)
    normalized=sorted({canonical_pair(int(a),int(b)) for a,b in pairs})
    shards=[]
    for shard in range(3):
        row=module(f'head_and_module863_key_shard_{shard}.json','key')
        assert row['shard']==shard and row['shards']==3 and row['all_shard_pairs_tested']
        subset=normalized[shard::3]
        assert row['total_pairs']==len(normalized) and row['tested_pairs']==len(subset)
        assert row['tested_triples']==sum(512 if a%2==0 else 1024 for a,b in subset)
        shards.append(row)
    for kind in ('input','key'):
        control=read('head_and_module863_'+kind+'_control.json')
        assert control['source']==kind and control['control'] and control['positive_witness_retained']

    coarse=read('head_single_bitwise863_results.json')
    expected=set(cases());seen={}
    for row in coarse['results']:
        case=tuple(row['sequence']),row['first_form']
        assert case in expected and case not in seen
        assert row['width']==11 and row['fixed_input_key']==C1 and row['fixed_output_key'] is None
        assert not row['pin_original'] and row['control_program'] is None
        assert not row['fixed_operands'] and row['result'] in ('projected_grammar_unsat','unknown')
        seen[case]=row['result']
    assert set(seen)==expected
    for op in ('&','|'):
        for position in (1,2,3):
            order=tuple(op if i==position else 'mad' for i in range(4))
            assert all(seen[order,form]=='projected_grammar_unsat' for form in ('square_const','square_x'))
        row=read('head_polynomial_'+('and' if op=='&' else 'or')+'863_result.json')
        assert row['width']==11 and row['operator']==op and not row['control']
        assert row['fixed_input_key']==C1 and row['free_output_key'] and row['degrees']==[4,2]
        assert row['result']=='relaxed_family_unsat'
    square=certificate()
    assert square['target_mixed_differences']==[1024] and square['polynomial_mixed_difference']==0
    assert square['points']==[0,32,64,96]
    # Bitwise prefixes of x and constants preserve the same axis separation;
    # any polynomial continuation has zero difference for all output keys.

    coarse_final=read('head_and_final863_results.json')
    fine=read('head_and_final863_refinement.json')
    assert fine['refinement']['sha256']==evidence['head_and_final863_results.json']

    def validate(row,second=None):
        op,source,last=row['operator'],row['source'],tuple(row['last_operands'])
        raw=row['raw']
        assert raw['sequence']==['mad','mad',op,'mad'] and raw['width']==11
        assert raw['first_form']=='affine' and raw['fixed_input_key']==C1 and raw['fixed_output_key'] is None
        assert not raw['pin_original'] and raw['control_program'] is None and not raw['constant_multipliers_only']
        assert all(raw[k] is None for k in ('middle_source','tail_form','last_operands','second_form'))
        fixed={0:(0,1,2),2:SOURCES[source],3:last}
        if second is not None:fixed[1]=second
        assert raw['fixed_operands']=={str(k):list(v) for k,v in fixed.items()}
        assert row['result']==raw['result'] and row['result'] in ('projected_grammar_unsat','unknown')
        return op,source,last

    expected=set(product(('&','|'),SOURCES,last_roles()))
    coarse_cases={}
    for row in coarse_final['results']:
        case=validate(row)
        assert case in expected and case not in coarse_cases
        coarse_cases[case]=row['result']
    assert set(coarse_cases)==expected
    templates={(op,source,tuple(min(v,4) for v in last)) for (op,source,last),status
               in coarse_cases.items() if status=='unknown'}
    assert len(templates)==10 and all(source=='key' for op,source,last in templates)
    expected_fine={(op,source,last,second) for op,source,last in templates for second in quadratic_roles()}
    fine_cases=set()
    for row in fine['results']:
        second=tuple(row['second_operands'])
        op,source,last=validate(row,second)
        case=op,source,last,second
        assert case in expected_fine and case not in fine_cases
        assert row['result']=='projected_grammar_unsat'
        fine_cases.add(case)
    assert fine_cases==expected_fine

    # Exhaust literal wiring, including constant aliases and dead code.
    wiring=Counter()
    all_second=[(*pair,c) for pair in combinations_with_replacement(range(4),2) for c in range(4)]
    all_last=[(*pair,c) for pair in combinations_with_replacement(range(6),2) for c in range(6)]
    for op,second,third,last in product(('&','|'),all_second,
            combinations_with_replacement(range(5),2),all_last):
        if (3 not in last or third[0]==third[1] or third[0]>=3 or
                (1 not in second and 1 not in third and 1 not in last)):
            wiring['three_operations']+=1
        elif 2 not in third:
            wiring['reordered_bitwise_prefix']+=1
        else:
            source='input' if third==(0,2) else 'first' if third==(1,2) else 'key'
            case=op,source,last
            assert case in coarse_cases
            if coarse_cases[case]=='projected_grammar_unsat':
                wiring['coarse_query']+=1
            else:
                canonical_last=tuple(min(v,4) for v in last)
                kind,canonical=canonical_second(second)
                if kind=='affine':
                    assert source=='key'
                    wiring['affine_module']+=1
                else:
                    assert (op,source,canonical_last,canonical) in fine_cases
                    wiring['quadratic_query']+=1
    return dict(executable_best_cycles=864,new_executable=False,width=11,input_key=C1,
        output_key='arbitrary',closed_orders=8,orders=[[op if i==position else 'mad'
            for i in range(4)] for op in ('&','|') for position in range(4)],
        input_module_pairs=input_module['tested_pairs'],key_module_pairs=len(normalized),
        key_module_triples=sum(row['tested_triples'] for row in shards),
        coarse_counts=dict(Counter(coarse_cases.values())),refinement_queries=len(fine_cases),
        affine_final_wiring_patterns=sum(wiring.values()),affine_final_wiring_classes=dict(wiring),
        evidence_sha256=evidence,result='all_eight_single_and_or_orders_excluded',
        scope='One input; four instructions consisting of exactly three arbitrary MAD and one '
              'AND/OR; arbitrary constants/runtime operands and output XOR.',
        excluded_from_claim='XOR combinations, other opcodes, extra live values, whole-hash '
                            'and memory-layout transformations, global kernel optimum.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit();args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('result','input_module_pairs','key_module_triples',
        'coarse_counts','refinement_queries','affine_final_wiring_classes')}))


if __name__=='__main__':main()
