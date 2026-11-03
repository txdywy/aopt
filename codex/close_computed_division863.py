"""Join the new computed-prefix certificates and retain the real frontier."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from codex.audit_comparison_core863 import fibre
from codex.audit_division_first_t863 import source_inverse
from codex.search_early_boundary863 import C0, MASK, NAMES, NODE, early_inputs, wanted, legal_point


def audit():
    evidence = {}
    def read(path):
        raw = Path(path).read_bytes(); value = json.loads(raw)
        evidence[path] = hashlib.sha256(raw).hexdigest()
        assert value['node'] == NODE
        return value
    smt = read('codex/middle_division_residual863_screen.json')
    enum = read('codex/middle_residual_enum863.json')
    mad = read('codex/middle_mad_quotient863_audit.json')
    raw_collision = read('codex/odd_prefix_collisions863_audit.json')
    collision = read('codex/odd_prefix_collisions863_refinement.json')
    verified = read('codex/odd_prefix_collisions863_verified.json')
    small = read('codex/small_numerator863_audit.json')
    runtime = read('codex/middle_runtime_factor863_audit.json')
    read('codex/final_division863_audit.json')
    read('codex/bitmask_affine_tail863_audit.json')
    assert len(smt['results']) == 18 and all(r['status'] == 'unknown' for r in smt['results'])
    assert enum['complete'] and enum['all_one_high_excluded'] and enum['all_bounded_excluded']
    assert len(enum['results']) == 9
    assert enum['scanner_source_sha256'] == hashlib.sha256(Path('codex/enumerate_middle_residual863.cpp').read_bytes()).hexdigest()
    assert {r['id'] for r in enum['results']} == set(range(9))
    for row in enum['results']:
        assert row['complete'] and row['width'] == 32 and row['evaluated_odd_parameters'] == 1 << 31
        assert not row['models'] and not row['bounded_models']
    for case in enum['dataset']['cases']:
        assert case['width'] == 32 and len(set(case['residuals'])) == len(case['residuals'])
        for b, r in zip(case['primary_witnesses'], case['residuals']):
            v = early_inputs(b); assert (wanted(v)-v[case['id']]) & MASK == r
    assert raw_collision['excluded'] == 63 and raw_collision['parity_cases'] == 64
    assert collision['excluded'] == collision['parity_cases'] == 64 and collision['canonical_prefixes'] == 702
    assert verified['remaining'] == 0 and verified['full32_parameter_queries'] == 64
    assert verified['source_sha256'] == evidence['codex/odd_prefix_collisions863_refinement.json']
    assert mad['remaining_in_stated_families'] == runtime['remaining_in_stated_family'] == 0
    assert small['all_stated_cases_excluded'] and len(small['results']) == 9
    anchors = smt['finite_anchors']['zero_dividend_addend_cases']
    assert len(anchors) == 9 and all(not r['possible_initial_addends'] for r in anchors)
    assert all(r['zero_witness']['target'] != 0 for r in anchors)
    ff = fibre(); assert len({y & 3 for y in ff['target']}) == 4
    folded = (((-C0)*pow(4097, -1, 1 << 32)) & MASK) ^ NODE
    e = folded ^ (folded >> 16); b = source_inverse(0, e); v = early_inputs(b)
    assert wanted(v) == 0 and all(x != MASK for x in v)
    return dict(node=NODE, input_names=NAMES, evidence_sha256=evidence,
                constant_inverse_enumeration=dict(smt_unknown_preserved=18,
                    exact_odd_parameters_per_source=1 << 31, sources=9, total_odd_parameters=9*(1 << 31),
                    survivors=0, not_divisor_enumeration=True),
                odd_prefixes=dict(canonical=702, admissible_parity_cases=64,
                    discovery_closed=63, algebraic_reflection_closed=1, full32_key_queries=64),
                closed_families=[
                    'T prefix;middle DIV/REM/CDIV;last G*q+W_initial. Prepared constant G is odd. Computed numerator / prepared D uses bounded enumeration (D>=2 for DIV/CDIV), REM D<=2^31+1 uses the same bound; larger REM D gives at most two low16 classes on the eight-state fibre. D=1 reduces to T-only operations. Prepared numerator / computed denominator uses the one-high enumeration; initial numerator / computed denominator uses zero anchors.',
                    'P is one MAD/bitop;out=P*q+W_initial, q any function only of P/root constants. All702 prefixes and key parities covered by the odd-prefix collision certificates. This includes either prepared divisor or prepared numerator. An initial numerator with computed denominator is separately excluded by zero anchors.',
                    'P is one MAD; q=DIV/REM/CDIV(P,D_prepared); out=R(initials,P,K)+q for any integer polynomial R, or out=P+G*q. Exact2^31 rectangle covers every nonzero uint32 divisor.',
                    'P is one MAD/bitop denominator;initial numerator;out=P*q+H_prepared. Small-numerator forcing excludes all nine sources. If q depends only on P/root constants, a required always-odd P is a nonbijective wordcut, also impossible for a permutation target.',
                    'A last q*q+T-only-addend has at most two residues modulo4 on the eight-state fibre; q*q+q is always even.',
                    'A last q*initial+T-only-addend fails a legal even-factor parity pair. The q*initial+q exception v+1 is closed by219 high-output forcing points; other plus-one factors fail parity.',
                    'A last P*q+P forces P odd. For an initial numerator, target0 would require q=MASK, but none of the initial numerators equals MASK there. For a root-only/data-P numerator, the odd-prefix wordcut is nonbijective. A last P*q+q instead forces P even; initial-numerator zero anchors or the wordcut argument exclude it.',
                ],
                target0_initial_numerators_not_mask=legal_point(b),
                no_claim_of_complete_three_op_ISA=True,
                still_open=[
                    'Computed-denominator cases with a prepared numerator and a final q-plus-product tail containing an initial-word bypass.',
                    'Computed-denominator cases with an initial numerator and final q-plus-product or P+G*q; more small-numerator branching is available.',
                    'A first bitop followed by a prepared-divisor quotient and a final q-plus-product or P+G*q, when not reduced by other certificates.',
                    'Multiple divisions, division/shift or division/comparison combinations (including REM), additional live words, larger joint windows, state encodings and other kernel algorithms.',
                ], best_executable_cycles=864, new_executable=False, no_global_optimality_claim=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Computed-prefix certificates joined; exact inverse parameters19327352832; best executable864', flush=True)


if __name__ == '__main__':
    main()
