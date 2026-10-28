"""Join the complete one-T/computed-denominator/one-T local grammar.

No full ISA or whole-kernel lower bound is asserted. Each canonical last
MAD/bitop is mapped to a concrete certificate, including quotient reuse,
initial-word bypasses, nonunit constants, and a prepared numerator.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import z3

from codex.audit_division_first_t863 import source_inverse
from codex.audit_odd_prefix_collisions863 import evaluate, manifest
from codex.audit_prepared_numerator863 import denominator_intervals, low_scalar_filter, numerator_candidates, remainder_progressions
from codex.cegis_c1_absorb863 import _instr_options
from codex.screen_middle_division_residual863 import pool
from codex.search_early_boundary863 import C0, MASK, NAMES, NODE, early_inputs, wanted, legal_point


def tail_class(op, wiring):
    """Pool: initial0..8, computed P9, quotient q10, root keys11/12."""
    p, q = 9, 10
    if q not in wiring: return 'no_quotient_parity'
    if op != 'mad':
        other = wiring[1] if wiring[0] == q else wiring[0]
        if other == q: return 'quotient_only_or_zero'
        return 'bitop_initial' if other < 9 else 'bitop_prefix' if other == p else 'bitop_prepared'
    a, b, addend = wiring
    if a == q and b == q:
        return 'quotient_square_plus_self_even' if addend == q else 'quotient_square_mod4'
    if a == q or b == q:
        factor = b if a == q else a
        if factor < 9:
            return 'initial_plus_one_factor' if addend == q else 'even_initial_factor'
        if factor == p:
            return ('prefix_factor_plus_self' if addend == p else
                    'prefix_plus_one_factor' if addend == q else
                    'prefix_factor_initial_addend' if addend < 9 else 'prefix_factor_prepared_addend')
        return ('prepared_factor_prefix_addend' if addend == p else
                'prepared_factor_initial_addend' if addend < 9 else 'quotient_affine_prepared')
    assert addend == q
    if a < 9 and b < 9: return 'quotient_plus_fixed_initial_product'
    if a < 9 and b == p or b < 9 and a == p: return 'quotient_plus_prefix_initial_product'
    if a < 9 or b < 9: return 'quotient_plus_scalar_initial_product'
    if a == p and b == p: return 'quotient_plus_prefix_square'
    if a == p or b == p: return 'quotient_plus_scalar_prefix'
    return 'quotient_affine_prepared'


def edge_target(value):
    folded = (((value-C0)*pow(4097, -1, 1 << 32)) & MASK) ^ NODE
    e = folded ^ (folded >> 16)
    b = source_inverse(0, e)
    assert wanted(early_inputs(b)) == value
    return b


def prepared_bitop_certificates():
    bs = pool(); vv = [early_inputs(b) for b in bs]; yy = [wanted(v) for v in vv]
    xor = []
    for source in range(9):
        high = {}
        for b, v, y in zip(bs, vv, yy):
            q = y ^ v[source]
            if q > 1 << 31: high.setdefault(q, b)
            if len(high) == 2: break
        assert len(high) == 2
        xor.append(dict(source=source, distinct_high_quotients=[dict(q=q, primary_b=b) for q, b in high.items()]))
    zero, maximum = edge_target(0), edge_target(MASK)
    assert all(x != 0 for x in early_inputs(zero))
    assert all(x != MASK for x in early_inputs(maximum))
    return dict(xor_initial=xor, or_initial_zero_witness=legal_point(zero),
                and_initial_mask_witness=legal_point(maximum),
                prefix_or_prepared_operand='With a prepared numerator, q depends only on the nonzero denominator P. Any output using only P/q/root constants has at most2^32-1 distinct values and cannot equal the permutation target.')


def audit():
    evidence = {}
    def read(name):
        path = Path('codex')/name; raw = path.read_bytes(); result = json.loads(raw)
        assert result['node'] == NODE
        evidence[str(path)] = hashlib.sha256(raw).hexdigest()
        return result
    previous = read('computed_division863_closed_audit.json')
    branches = read('computed_denominator_branches863.json')
    products = read('denominator_products863_audit.json')
    bitops = read('denominator_bitops863_audit.json')
    prepared = read('prepared_numerator863_audit.json')
    scalar = read('prepared_scalar863_refinement.json')
    odd = read('odd_denominator_bypass863_audit.json')
    assert previous['best_executable_cycles'] == 864
    assert branches['complete'] and len(branches['results']) == 54
    assert {(r['source'], r['op'], r['family']) for r in branches['results']} == {
        (source, op, family) for source in range(9) for op in ('//', '%', 'cdiv') for family in ('plus', 'times')}
    assert all(r['mad_excluded'] and r['bit_excluded'] for r in branches['results'])
    for group in products['counts'].values(): assert group['cases'] == group['excluded']
    assert bitops['all_stated_cases_excluded'] and len(bitops['results']) == 9
    assert {r['source'] for r in products['results']} == set(range(9))
    assert {r['source'] for r in prepared['prefix_initial_products']} == set(range(9))
    assert {r['source'] for r in prepared['scalar_initial_products']} == set(range(9))
    assert len(prepared['fixed_initial_products']) == 45
    for row in prepared['fixed_initial_products']:
        assert row['status'] == 'excluded' and len(row['high_outputs']) == 2
        a, b = row['factors']; high = []
        for w in row['high_outputs']:
            v = early_inputs(w['primary_b']); q = (wanted(v)-v[a]*v[b]) & MASK
            assert q == w['q'] and q > 1 << 31; high.append(q)
        assert len(set(high)) == 2
    for row in prepared['scalar_initial_products']:
        if row.get('certificate') == 'two_distinct_high_targets_at_zero_factor':
            assert row['status'] == 'excluded'
            assert len(row['high_outputs']) == len({w['q'] for w in row['high_outputs']}) == 2
            for w in row['high_outputs']:
                v = early_inputs(w['primary_b'])
                assert v[row['source']] == 0 and wanted(v) == w['q'] > 1 << 31
        else:
            lows, history = low_scalar_filter(row['initial_factors'], row['targets'])
            assert lows == row['low16_survivors'] and history == row['history']
            assert not lows or row['source'] == 0
            for b, x, y in zip(row['primary_b'], row['initial_factors'], row['targets']):
                v = early_inputs(b); assert v[row['source']] == x and wanted(v) == y
    assert scalar['previous_sha256'] == evidence['codex/prepared_numerator863_audit.json']
    lows, history = low_scalar_filter(scalar['initial_factors'], scalar['targets'])
    assert scalar['status'] == 'excluded' and not lows and history == scalar['history']
    for b, x, y in zip(scalar['primary_b'], scalar['initial_factors'], scalar['targets']):
        v = early_inputs(b); assert v[0] == x and wanted(v) == y
    numerator_count = 0; residual_rem = []
    for row in prepared['prefix_initial_products']:
        source = row['source']
        if row.get('all_ops'):
            assert row['status'] == 'excluded'; continue
        for b, x, y in zip(row['primary_b'], row['initial_factors'], row['targets']):
            v = early_inputs(b); assert v[source] == x and wanted(v) == y
        for case in row['cases']:
            if case['op'] == '%':
                if case['status'] == 'open': residual_rem.append(source); continue
                index = case['impossible_index']
                assert case['pinned_numerator'] == row['targets'][0] > 1 << 31
                assert not remainder_progressions(row['initial_factors'][index], case['pinned_numerator'], row['targets'][index])
                continue
            candidates = numerator_candidates(row['targets'][0], case['op'])
            assert candidates == case['initial_numerator_candidates'] and case['status'] == 'excluded'
            assert not case['surviving_numerators']
            deaths = case['empty_denominator_certificates']; numerator_count += len(candidates)
            assert sorted(w['numerator'] for w in deaths) == candidates
            for w in deaths:
                index = w['witness_index']; n = row['initial_factors'][index]
                assert n == w['impossible_factor']
                assert not denominator_intervals(n, w['numerator'], row['targets'][index], case['op'])
    assert residual_rem == [0, 6]
    expected = {(r['op'], tuple(r['wiring']), tuple(parity), factor)
                for r in manifest() for parity in r['odd_key_parities'] for factor in residual_rem}
    actual = {(r['op'], tuple(r['wiring']), tuple(r['key_parity']), r['factor']) for r in odd['results']}
    assert expected == actual and len(actual) == 128 and odd['excluded'] == 128 and not odd['unknown']
    replayed_equations = 0; replayed_collisions = 0; legal_bs = set()
    for row in odd['results']:
        assert row['status'] == 'unsat' and row['full32_bypass_equations_refuted']
        kk = [z3.BitVec(f'replay_odd_key_{j}', 32) for j in range(2)]
        conditions = [kk[j-9] & 1 == row['key_parity'][j-9] for j in set(row['wiring']) & {9, 10}]
        solver = z3.SolverFor('QF_BV'); solver.set(timeout=10000); solver.add(*conditions)
        for w in row['witnesses']:
            v, u = [early_inputs(b) for b in w['primary_pair']]; legal_bs.update(w['primary_pair'])
            a = evaluate(row['op'], row['wiring'], v, kk); b = evaluate(row['op'], row['wiring'], u, kk)
            same = z3.SolverFor('QF_BV'); same.set(timeout=10000); same.add(*conditions, a != b)
            assert same.check() == z3.unsat; replayed_collisions += 1
            if not z3.is_bv(a): a = z3.BitVecVal(a, 32)
            dy = (wanted(v)-wanted(u)) & MASK; di = (v[row['factor']]-u[row['factor']]) & MASK
            assert w['target_difference'] == dy and w['factor_difference'] == di
            solver.add(a*di == dy)
        assert solver.check() == z3.unsat; replayed_equations += 1
    rows = []
    for op in ('mad', '^', '&', '|'):
        for wiring in _instr_options(10, op):
            rows.append(dict(op=op, wiring=list(wiring), certificate=tail_class(op, wiring)))
    for op in ('^', '&', '|'):
        rows.append(dict(op=op, wiring=[10, 10], certificate=tail_class(op, [10, 10])))
    classes = dict(Counter(r['certificate'] for r in rows))
    return dict(node=NODE, input_names=NAMES, evidence_sha256=evidence,
                last_operation_manifest=rows, last_operation_counts=classes,
                exact_small_denominator_scalar_branches=sum(r.get('finite_scalars', 0) for r in branches['results']),
                exact_prepared_numerator_candidates=numerator_count,
                independent_full32_collision_queries=replayed_collisions,
                independent_full32_bypass_queries=replayed_equations,
                odd_collision_legal_points=[legal_point(b) for b in sorted(legal_bs)],
                prepared_bitop_certificates=prepared_bitop_certificates(),
                completed_scope='Nine stated initial live words, arbitrary root-only constants; P is one MAD/XOR/AND/OR; q=DIV/REM/CDIV(initial,P) or DIV/REM/CDIV(C_prepared,P), valid nonzero P on every legal input; final one MAD/XOR/AND/OR. Every canonical last operation is classified, including q reuse. P/P gives only1/0 and is separately T-only. Pure prepared and identity first-prefix cases are included by the input-range/parity/wordcut arguments or invalid zero-divisor witness.',
                numerator_P_denominator_P='With nonzero P, floor/ceil(P/P)=1 and P%P=0; final T cannot change the necessary b16 parity.',
                complete_in_stated_scope=True, new_executable=False, best_executable_cycles=864,
                still_open=[
                    'P first bitop, prepared divisor D, q=DIV/REM/CDIV(P,D), final q-plus-product or P+G*q when not reduced by earlier certificates.',
                    'Multiple divisions; division with shifts/comparisons; additional live words and root/leaf-dependent preparation; larger joint windows; nonlinear state encodings and other kernel algorithms.',
                ], no_global_optimality_claim=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Computed-denominator grammar closed:', result['complete_in_stated_scope'], flush=True)
    print('Final-operation forms:', len(result['last_operation_manifest']), flush=True)
    print('Full32 collision/bypass replays:', result['independent_full32_collision_queries'],
          result['independent_full32_bypass_queries'], flush=True)


if __name__ == '__main__':
    main()
