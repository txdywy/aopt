"""Prepared-count SHR, then one division, then one MAD/bitop.

Two legal inputs with initial h=0,1 make h>>s=0 for every s>=1.
They force exact divisor candidates for independent quotients and
initial-product/scalar constraints for a zero-preserving dependent q.
This does not close D,S,T with the shift reading a prior quotient.
"""
from __future__ import annotations

import argparse
from itertools import combinations_with_replacement
import json
from math import isqrt
from pathlib import Path

from codex.audit_computed_shift_amount863 import zero_points
from codex.audit_denominator_products863 import intersect, linear_class
from codex.audit_division_first_t863 import source_inverse
from codex.cegis_c1_absorb863 import _instr_options
from codex.screen_division_xor_affine863 import quotient
from codex.search_early_boundary863 import MASK, NAMES, NODE, early_inputs, wanted, legal_point


def point_divisors(n, y, op, width=32):
    """Exact D domain with op(n,D)=y>1; retain equality intervals for REM."""
    assert y > 1 and 0 <= n < 1 << width
    mask = (1 << width)-1; intervals = []; points = set()
    if op == '//':
        lo, hi = max(1, n//(y+1)+1), min(mask, n//y)
        if lo <= hi: intervals.append([lo, hi])
    elif op == 'cdiv':
        lo, hi = max(1, (n+y-1)//y), min(mask, (n-1)//(y-1))
        if lo <= hi: intervals.append([lo, hi])
    elif op == '%':
        if n == y:
            if n < mask: intervals.append([n+1, mask])
        elif n > y:
            delta = n-y; bound = min(delta//(y+1), isqrt(delta))
            for factor in range(1, bound+1):
                if delta % factor == 0:
                    other = delta//factor
                    if factor > y: points.add(factor)
                    if other > y: points.add(other)
    else:
        raise ValueError(op)
    return dict(intervals=intervals, isolated_divisors=sorted(points))


def shift_zero_pair(source):
    b0 = zero_points()[NAMES[source]]
    b1 = b0 ^ (1 << 23) if source == 3 else b0 ^ 65536 if source == 7 else source_inverse(source, 1)
    v, w = early_inputs(b0), early_inputs(b1)
    assert v[source] == 0 and w[source] in (0, 1)
    assert wanted(v) != wanted(w)
    return [b0, b1]


def independent_tail_class(op, wiring):
    assert 9 in wiring and 10 in wiring
    if op != 'mad': return 'zero_output' if op == '&' else 'shift_zero_pins_quotient'
    a, b, c = wiring
    if c == 10 and 9 in (a, b): return 'shift_zero_pins_quotient'
    if c == 9 and 10 in (a, b): return 'quotient_zero_returns_shift'
    assert {a, b} == {9, 10}
    return 'shift_zero_returns_initial' if c < 9 else 'two_shift_zeros_constant'


def dependent_tail_class(op, wiring):
    assert 10 in wiring
    if not any(j < 9 for j in wiring): return 'noninjective_shift_wordcut'
    if op != 'mad': return 'zero_or_initial_bitop'
    a, b, c = wiring
    if 10 in (a, b):
        return 'returns_initial' if c < 9 else 'prepared_addend_at_two_zeros' if c >= 11 else 'zero_output'
    assert c == 10
    if 9 in (a, b): return 'zero_output'
    return 'fixed_initial_product_at_two_zeros' if a < 9 and b < 9 else 'scalar_initial_product_at_two_zeros'


def audit():
    independent = []; dependent = []; legal = set(); total_candidates = 0
    zeros = zero_points(); zero_q = []
    for numerator in range(9):
        b = zeros[NAMES[numerator]]; v = early_inputs(b); y = wanted(v); legal.add(b)
        assert v[numerator] == 0
        # s=32 represents every s>=32; a Python wide shift also gives0.
        assert all(y != v[h] >> s for h in range(9) for s in range(33))
        zero_q.append(dict(numerator=numerator, primary_b=b, target=y,
                           initial_values=v, all_prepared_shr_values_disagree=True))
    for source in range(9):
        bs = shift_zero_pair(source); legal.update(bs)
        vv = [early_inputs(b) for b in bs]; yy = [wanted(v) for v in vv]
        assert yy[0] > 1 and all(yy[0] != x for x in vv[0])
        for numerator in range(9):
            for op in ('//', '%', 'cdiv'):
                domain = point_divisors(vv[0][numerator], yy[0], op)
                candidates = domain['isolated_divisors'][:]
                for lo, hi in domain['intervals']:
                    assert hi-lo < 128  # A consequence of this actual target, not a search cutoff.
                    candidates.extend(range(lo, hi+1))
                candidates = sorted(set(candidates)); total_candidates += len(candidates)
                assert all(quotient(vv[0][numerator], d, op) == yy[0] for d in candidates)
                remaining = [d for d in candidates if quotient(vv[1][numerator], d, op) == yy[1]]
                assert not remaining
                independent.append(dict(shift_source=source, numerator=numerator, op=op,
                                        zero_primary_pair=bs, targets=yy,
                                        initial_divisor_domain=domain, candidate_divisors=candidates,
                                        second_quotients=[quotient(vv[1][numerator], d, op) for d in candidates],
                                        remaining=remaining, status='excluded'))
        scalar_rows = []
        for initial in range(9):
            scalar = intersect(linear_class(vv[0][initial], yy[0]), linear_class(vv[1][initial], yy[1]))
            assert scalar is None
            scalar_rows.append(dict(initial=initial, scalar_congruence_intersection=None))
        fixed_rows = []
        for a, b in combinations_with_replacement(range(9), 2):
            products = [(v[a]*v[b]) & MASK for v in vv]
            assert products != yy
            fixed_rows.append(dict(initial_factors=[a, b], products=products, targets=yy))
        dependent.append(dict(shift_source=source, zero_primary_pair=bs, targets=yy,
                              scalar_initial_products=scalar_rows, fixed_initial_products=fixed_rows,
                              status='excluded'))
    both = [(op, list(w)) for op in ('mad', '^', '&', '|') for w in _instr_options(10, op)
            if 9 in w and 10 in w]
    live_q = [(op, list(w)) for op in ('mad', '^', '&', '|') for w in _instr_options(10, op) if 10 in w]
    live_q += [(op, [10, 10]) for op in ('^', '&', '|')]
    return dict(node=NODE, input_names=NAMES, independent_quotient_results=independent,
                dependent_quotient_zero_results=dependent, quotient_numerator_zero=zero_q,
                exact_independent_divisor_candidates=total_candidates,
                final_forms_using_both_results=[dict(op=op, wiring=w, certificate=independent_tail_class(op, w)) for op, w in both],
                dependent_quotient_live_final_forms=[dict(op=op, wiring=w, certificate=dependent_tail_class(op, w)) for op, w in live_q],
                independent_last_op_partition='When both S and q are read, every last MAD either returns q at S=0, returns S at q=0, gives an initial addend at S=0, gives a prepared addend at S=0, or gives0. The first case is the complete divisor screen; second uses the numerator-zero witnesses; initial/zero addends fail at the first S-zero witness; a prepared addend fails on the two distinct targets. XOR/OR return q at S=0; AND gives0.',
                dependent_last_op_partition='q=DIV/REM/CDIV(S,D) is zero whenever S=0. A tail depending only on S/q/root constants cannot be injective because S=h>>s has fewer than2^32 values for s>=1. Every remaining final MAD has an initial bypass: a quotient/shift factor plus an initial addend returns that initial at the first zero witness; a prepared addend is the same at both zero witnesses; an initial times S/q with S/q as addend gives0. An initial product plus q or prepared G*initial+q must fit the recorded two zero inputs, and every such fixed-product/scalar congruence fails. Binary q/initial bitops return0 or that initial at the first zero witness. This also covers dependent tails that do not directly read S.',
                constant_quotient_or_dead_op='A prepared numerator/prepared denominator gives a constant q and reduces to the closed shift core. For independent quotients only, an unused last-op result is truly dead and reduces to the earlier division-free or division-first core; no such deletion is assumed for dependent quotients. s=0 makes S an already available initial word, covered by the division-first core. All s>=32 give0 and obey the same two-zero argument.',
                invalid_denominators='Every initial-word divisor and S itself reaches0 on a legal input, so neither can be a valid denominator in this three-op grammar.',
                legal_points=[legal_point(b) for b in sorted(legal)],
                complete_in_stated_scope=True, best_executable_cycles=864, new_executable=False,
                scope='First S=SHR(initial_h,prepared_count), then one DIV/REM/CDIV with operands from initials/S/root constants, then MAD/XOR/AND/OR with arbitrary initial/S/q/root operands. Includes independent D,S,T when it commutes to S,D,T. Excludes variable shift counts and a second shift reading an earlier quotient.',
                no_global_optimality_claim=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Independent quotient cases:', len(result['independent_quotient_results']), flush=True)
    print('Exact divisor candidates:', result['exact_independent_divisor_candidates'], flush=True)
    print('Both-result final forms:', len(result['final_forms_using_both_results']), flush=True)


if __name__ == '__main__':
    main()
