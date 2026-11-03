"""Close the stated three-op MAD/bitop core with at most one division.

This joins the prepared-divisor middle placement to prior division-first,
division-last and computed-denominator certificates. Other ISA operations,
more than one division, additional free words and larger windows stay open.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from codex.audit_quotient_circular863 import circular_span, unit_span_scan
from codex.audit_floor_xor_prefix863 import inverse_profile
from codex.audit_runtime_shift_core863 import base_fields, monomials
from codex.audit_small_numerator863 import bit_fits
from codex.close_computed_denominator863 import tail_class
from codex.cegis_c1_absorb863 import _instr_options
from codex.audit_shift_division863 import dependent_tail_class, independent_tail_class, point_divisors
from codex.audit_denominator_products863 import intersect, linear_class
from codex.screen_division_xor_affine863 import quotient
from codex.screen_xor_mad_polynomial import consistent
from codex.search_early_boundary863 import MASK, NAMES, NODE, early_inputs, wanted


CERTIFICATE_MAP = {
    'no_quotient_parity':'All initial and T-only LSBs ignore b16, target parity flips.',
    'quotient_only_or_zero':'A permutation output depending only on conservative q forces permutation P/q and q=P; two T ops still ignore b16.',
    'quotient_square_plus_self_even':'q*q+q is even.',
    'quotient_square_mod4':'T-only addend low2 is constant on the fibre; q*q has only0/1 modulo4, fewer than the four target residues.',
    'even_initial_factor':'An initial coefficient has an even-parity class with a b16 partner; the remaining T-only addend cannot flip output parity.',
    'initial_plus_one_factor':'Prior initial-plus-one audit: all but v+1 fail parity; v+1 requires219 distinct high numerators inconsistent with one MAD/bitop.',
    'prefix_factor_initial_addend':'Prior complete always-odd prefix collisions:702 canonical prefixes,64 admissible parameter parities, all initial addends refuted.',
    'prefix_factor_plus_self':'Permutation output is a function only of P, forcing permutation P; parity forces P always odd, contradiction.',
    'prefix_plus_one_factor':'Permutation output is a function only of P, forcing permutation P; parity forces P always even, contradiction.',
    'prefix_factor_prepared_addend':'Permutation output is a function only of P, forcing permutation P; parity forces P always odd, contradiction.',
    'prepared_factor_initial_addend':'Prior complete odd-inverse enumeration excludes even the relaxed q<=2^31 family. Floor/ceil D>=2 obey this; REM has the new302675 bound. D=1 is T-only.',
    'prepared_factor_prefix_addend':'One-MAD prefix: prior exact2^31 rectangle. First-bitop prefix: permutation/zero reduction and conservative P=1 parity certificate.',
    'quotient_affine_prepared':'Permutation output forces permutation q/P. Since0<=q<=P, sums force q=P, leaving only T operations.',
    'quotient_plus_fixed_initial_product':'New zero-factor anchor and circular floor/ceil/REM certificates, valid for every one-T prefix.',
    'quotient_plus_prefix_initial_product':'New zero-factor anchor and circular floor/ceil/REM certificates, valid for every one-T prefix.',
    'quotient_plus_scalar_initial_product':'New zero-factor anchor and circular floor/ceil/REM certificates, arbitrary prepared scalar including nonunits.',
    'quotient_plus_prefix_square':'One-MAD prefix: prior exact2^31 rectangle. First-bitop prefix: zero/permutation reduction; at P=1 the output is at most2.',
    'quotient_plus_scalar_prefix':'One-MAD prefix: prior exact2^31 rectangle. First-bitop prefix: P=1 gives two complete scalar candidates; both violate q<=P.',
    'bitop_initial':'AND/OR fail at target MASK/0 witnesses for arbitrary q. XOR requires high q: floor/ceil reduce to D=1; REM then has at most two low16 states.',
    'bitop_prefix':'AND/OR are pointwise<=P/>=P; permutation forces identity. XOR: ceil/REM noninjective at P=0,1; floor has64 complete small-target profiles inconsistent with every first MAD/bitop.',
    'bitop_prepared':'A permutation bitwise-mask output forces permutation q/P; conservative q=P, giving a T-only expression.',
}


def audit():
    evidence = {}
    def read(name):
        path = Path('codex')/name; raw = path.read_bytes(); result = json.loads(raw)
        assert result['node'] == NODE
        evidence[str(path)] = hashlib.sha256(raw).hexdigest()
        return result
    circle = read('quotient_circular863_audit.json')
    xor = read('floor_xor_prefix863_audit.json')
    computed = read('computed_denominator863_closed_audit.json')
    previous = read('computed_division863_closed_audit.json')
    mad = read('middle_mad_quotient863_audit.json')
    inverse = read('middle_residual_enum863.json')
    collisions = read('odd_prefix_collisions863_verified.json')
    runtime = read('middle_runtime_factor863_audit.json')
    first = read('division_first_t863_audit.json')
    last = read('final_division863_audit.json')
    no_division = read('comparison_core863_audit.json')
    shift = read('shift_division863_audit.json')
    assert no_division['remaining_in_stated_three_op_core'] == 0
    assert computed['complete_in_stated_scope']
    assert previous['best_executable_cycles'] == 864
    assert mad['remaining_in_stated_families'] == runtime['remaining_in_stated_family'] == 0
    assert inverse['complete'] and inverse['all_bounded_excluded'] and collisions['remaining'] == 0
    ff = circle['eight_state_fibre']
    fv = [early_inputs(b) for b in ff['primary_b']]
    assert fv == ff['initial_values'] and [wanted(v) for v in fv] == ff['target']
    assert len({y & 7 for y in ff['target']}) == 8
    assert all(len({v[j] & 65535 for v in fv}) == 1 for j in range(9))
    scan = unit_span_scan(ff['target']); assert scan == circle['rem_affine_quotient_bound']
    assert scan['complete_unit_domain'] and scan['enumerated_odd_units'] == 32768
    assert scan['minimum_span'] == 14190 and scan['necessary_max_divisor'] == 302675
    assert {r['source'] for r in circle['zero_factor_results']} == set(range(9))
    for row in circle['zero_factor_results']:
        v = early_inputs(row['zero_factor_primary_b']); y0 = wanted(v)
        assert v[row['source']] == 0 and y0 == row['target'] > scan['necessary_max_divisor']
        for case in row['cases']:
            assert case['status'] == 'excluded'
            if case['op'] == '%':
                assert case['required_divisor_minimum'] > case['global_divisor_maximum']
            else:
                maximum = MASK//y0 if case['op'] == '//' else (MASK-1)//(y0-1)
                assert len(case['divisor_checks']) == maximum
                for check in case['divisor_checks']:
                    d = check['divisor']; actual = circular_span([d*y for y in ff['target']])
                    assert actual == check['span'] > d-1
                assert not case['remaining_divisors']
    assert {r['source'] for r in circle['bitop_initial_xor']} == set(range(9))
    for row in circle['bitop_initial_xor']:
        w = row['witness']; v = early_inputs(w['primary_b'])
        assert wanted(v) ^ v[row['source']] == w['q'] > 1 << 31
    for op, target in (('or', 0), ('and', MASK)):
        witness = circle['bitop_initial_and_or'][op+'_witness']; v = early_inputs(witness['primary_b'])
        assert wanted(v) == target and all(x != target for x in v)
    vv = [early_inputs(b) for b in xor['primary_b']]
    assert [wanted(v) for v in vv] == list(range(128))
    matrix = [monomials(base_fields(b, v)) for b, v in zip(xor['primary_b'], vv)]
    assert [r['divisor'] for r in xor['floor_xor_profiles']] == [*range(2, 128, 2), 128]
    for row in xor['floor_xor_profiles']:
        p = inverse_profile(row['divisor'])
        assert p == row['required_prefix_values']
        assert row['status'] == 'excluded' and not consistent(matrix, p, 32) and not bit_fits(vv, p)
    assert len(xor['bitop_only_prefix_tails']) == 7
    zero = xor['bitop_only_prefix_reduction']['zero_initial_values']
    assert all(zero) and len(set(zero)) == 9
    for row in xor['bitop_only_prefix_tails']:
        assert row['pinned_xor_key'] == zero[row['source']]
        for p, b, y in zip((1, 2, 3), row['primary_b'], row['targets']):
            v = early_inputs(b)
            assert v[row['source']] ^ row['pinned_xor_key'] == p and wanted(v) == y
        for case in row['scalar_times_P_plus_q']['scalar_cases']:
            g = case['forced_scalar']; w = case['impossible_conservative_quotient']
            assert (w['target']-g*w['prefix']) & MASK == w['required_quotient'] > w['prefix']
        assert row['targets'][0] & 1 and row['targets'][0] > 2
    manifest = computed['last_operation_manifest']
    expected_manifest = {(op, tuple(w)) for op in ('mad', '^', '&', '|') for w in _instr_options(10, op)}
    expected_manifest.update((op, (10, 10)) for op in ('^', '&', '|'))
    actual_manifest = {(r['op'], tuple(r['wiring'])) for r in manifest}
    assert actual_manifest == expected_manifest and len(actual_manifest) == len(manifest) == 1158
    assert {r['certificate'] for r in manifest} == set(CERTIFICATE_MAP)
    assert all(tail_class(r['op'], r['wiring']) == r['certificate'] for r in manifest)
    assert first['complete'] and first['final_family_status'] == 'excluded'
    assert first['only_floor_survivor'] == [[0, '//', [[65536, 65536]]]]
    assert all(not r['remaining_odd_divisors_for_rem'] for r in first['results'] if r['op'] == '//')
    assert last['remaining_in_stated_family'] == 0
    assert shift['complete_in_stated_scope'] and shift['exact_independent_divisor_candidates'] == 43
    assert len(shift['independent_quotient_results']) == 243
    assert {(r['shift_source'], r['numerator'], r['op']) for r in shift['independent_quotient_results']} == {
        (s, n, op) for s in range(9) for n in range(9) for op in ('//', '%', 'cdiv')}
    for row in shift['independent_quotient_results']:
        v, w = [early_inputs(b) for b in row['zero_primary_pair']]
        y, z = wanted(v), wanted(w)
        assert v[row['shift_source']] == 0 and w[row['shift_source']] in (0, 1) and y != z
        domain = point_divisors(v[row['numerator']], y, row['op'])
        assert domain == row['initial_divisor_domain']
        candidates = set(domain['isolated_divisors'])
        for lo, hi in domain['intervals']: candidates.update(range(lo, hi+1))
        assert sorted(candidates) == row['candidate_divisors']
        assert all(quotient(w[row['numerator']], d, row['op']) != z for d in candidates)
    for row in shift['dependent_quotient_zero_results']:
        vv = [early_inputs(b) for b in row['zero_primary_pair']]; yy = [wanted(v) for v in vv]
        for initial in range(9):
            assert intersect(linear_class(vv[0][initial], yy[0]), linear_class(vv[1][initial], yy[1])) is None
        for item in row['fixed_initial_products']:
            a, b = item['initial_factors']; assert [(v[a]*v[b]) & MASK for v in vv] != yy
    for row in shift['quotient_numerator_zero']:
        v = early_inputs(row['primary_b']); y = wanted(v)
        assert v[row['numerator']] == 0 and all(y != v[h] >> s for h in range(9) for s in range(33))
    for row in shift['final_forms_using_both_results']:
        assert row['certificate'] == independent_tail_class(row['op'], row['wiring'])
    for row in shift['dependent_quotient_live_final_forms']:
        assert row['certificate'] == dependent_tail_class(row['op'], row['wiring'])
    assert len(shift['final_forms_using_both_results']) == 37
    assert len(shift['dependent_quotient_live_final_forms']) == 248
    assert {(r['op'], tuple(r['wiring'])) for r in shift['final_forms_using_both_results']} == {
        (op, w) for op, w in expected_manifest if 9 in w and 10 in w}
    assert {(r['op'], tuple(r['wiring'])) for r in shift['dependent_quotient_live_final_forms']} == {
        (op, w) for op, w in expected_manifest if 10 in w}
    return dict(node=NODE, raw_root=0x12345678, input_names=NAMES, evidence_sha256=evidence,
                prepared_divisor_middle=dict(complete=True, final_operation_forms=len(manifest),
                    quotient_using_forms=sum(10 in r['wiring'] for r in manifest),
                    family_certificates=CERTIFICATE_MAP, final_manifest_evidence='codex/computed_denominator863_closed_audit.json'),
                computed_denominator_middle=dict(complete=True, evidence='codex/computed_denominator863_closed_audit.json'),
                division_first=dict(complete=True, evidence='codex/division_first_t863_audit.json'),
                division_last=dict(complete=True, evidence='codex/final_division863_audit.json'),
                division_free=dict(complete=True, evidence='codex/comparison_core863_audit.json'),
                prepared_shr_then_division=dict(complete=True, evidence='codex/shift_division863_audit.json',
                    independent_quotient_cases=243, exact_divisor_candidates=43,
                    scope=shift['scope']),
                independent_middle_numerator='If middle division reads an initial numerator and prepared divisor without P, commute it before the independent first T and apply the division-first/two-T certificate. A prepared numerator and prepared divisor is constant, leaving T-only parity.',
                initial_denominator='Every initial-word denominator has a legal zero witness, irrespective of its numerator. Prepared divisor0 is also invalid. If both operands are the same valid P, floor/ceil give1 and REM gives0, leaving T-only parity.',
                completed_scope='At most three instructions, at most one DIV/REM/CDIV, remaining instructions MAD/XOR/AND/OR, arbitrary operands on the nine live words and root-only prepared constants, fixed allowed raw root0x12345678, legal execution on every reachable b. All placements of that single division are covered. Division-free core is covered by the earlier three-op core. This is a local macro grammar, not a full-kernel or full-ISA bound.',
                complete_in_stated_scope=True, best_executable_cycles=864, new_executable=False,
                still_open=[
                    'Two or more DIV/REM/CDIV operations; variable shift counts; a shift reading a prior quotient; REM mixed with comparisons; other division/shift orders where not covered by existing conservative-chain/Boolean certificates.',
                    'Additional free intermediates, root/leaf-dependent preparation, root-conditional macros, larger joint windows, nonlinear state encodings, cross-lane/FLOW/memory mechanisms and different kernel algorithms.',
                ], no_global_optimality_claim=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Single-division three-op core closed:', result['complete_in_stated_scope'], flush=True)
    print('Prepared-divisor final forms:', result['prepared_divisor_middle']['quotient_using_forms'], flush=True)


if __name__ == '__main__':
    main()
