"""Exclude a final division after at most two non-division operations.

On outputs >2^31 a floor/ceil quotient of uint32 operands must have
denominator1 and equal its numerator. Existing finite high-output prefix
certificates therefore apply regardless of the computed denominator.
Any final remainder misses uint32 MASK, attained by a real legal input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import z3

from codex.audit_comparison_core863 import fibre
from codex.audit_division_first_t863 import source_inverse
from codex.search_early_boundary863 import C0, NODE, MASK, early_inputs, wanted, legal_point


def audit():
    evidence = {}
    def read(path):
        raw = Path(path).read_bytes(); evidence[path] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    right = read('codex/runtime_shift_core863_audit.json')
    both = read('codex/bidirectional_shift_core863_audit.json')
    assert right['node'] == both['node'] == NODE
    middle = right['high_middle']
    assert len(middle['cases']) == 297 and middle['base_status'] == 'inconsistent_full32_high_outputs'
    assert all(row['quadratic_status'] == 'inconsistent_full32_high_outputs' and row['positive_control']
               for row in middle['cases'])
    assert all(wanted(early_inputs(b)) > 1 << 31 for b in middle['primary_samples'])
    ff = fibre(); high = [i for i, y in enumerate(ff['target']) if y > 1 << 31]
    assert len(high) == 4 and len({ff['target'][i] & 65535 for i in high}) == 4
    pair = both['odd_high_ancestor']['primary_pair']
    assert all(wanted(early_inputs(b)) > 1 << 31 and wanted(early_inputs(b)) & 1 for b in pair)
    # Exact integer inequalities implied by floor and ceil definitions.
    # Ceil uses n>D*(q-1), preserving the unbounded numerator carry.
    n, d, q = z3.Ints('final_dividend final_divisor final_quotient')
    bounds = []
    for op in ('//', 'cdiv'):
        s = z3.Solver(); s.set(timeout=30000)
        s.add(n >= 0, n <= MASK, d >= 2, d <= MASK, q > 1 << 31)
        s.add(n >= d*q if op == '//' else n >= d*(q-1)+1)
        assert s.check() == z3.unsat, op
        bounds.append(dict(op=op, high_quotient_requires_denominator1='unsat'))
    r = z3.Int('high_remainder')
    s = z3.Solver(); s.add(n >= 0, n <= MASK, d > r, r > 1 << 31, n >= d+r)
    assert s.check() == z3.unsat
    folded = (((MASK-C0)*pow(4097, -1, 1 << 32)) & MASK) ^ NODE
    e = folded ^ (folded >> 16)
    b = source_inverse(0, e)
    assert wanted(early_inputs(b)) == MASK
    partitions = {
        'two_T': 'Every T-only prefix datum has one low16 class on the eight-state fibre; four high targets have four different low16 values.',
        'one_comparison_one_T': 'A Boolean comparison splits the prefix low16 values at most twice. Four distinct high target low16 values remain impossible. A final Boolean numerator cannot be high.',
        'right_shift_then_T': 'Literal initial-word right shifts are the297 existing high-middle MAD/bitwise certificates. An initial-word shift count is>=32 on every high sample, giving0 and reducing to the base polynomial/bitwise certificates. Constant data/root counts reduce to a constant or an already covered case.',
        'left_shift_then_T': 'A prepared-count SHL normalizes to MAD. An initial-word count is>=32 on every high sample, giving0 and the same base necessary systems.',
        'T_then_shift': 'On the odd/high pair the numerator shift must have count0, for either direction. Its T/initial/root value ancestor has equal low2 on the pair, whereas targets differ at bit1.',
        'two_shifts_or_comparison_shift': 'Every value-operand shift on the odd/high pair has count0. An initial/root ancestor has equal low2; a Boolean ancestor cannot be high.',
    }
    return dict(node=NODE, prefix_inputs=list(range(9)), high_division_bounds=bounds,
                reused_high_middle_cases=297, prefix_order_partition=partitions,
                four_high_targets_fibre=ff, odd_high_ancestor_pair=[legal_point(x) for x in pair],
                final_rem_exclusion=dict(target_mask_witness=legal_point(b),
                    proof='A valid remainder satisfies0<=r<D<=MASK and therefore cannot equal MASK, regardless of its numerator or preceding operations.'),
                high_conservative_chains=dict(high_rem_must_equal_numerator='unsat',
                    proof='For output>2^31, DIV/CDIV require divisor1; REM requires quotient0. Their result equals the numerator. An odd/high SHL/SHR result also equals its data operand. Along a value-operand chain of these instructions every value remains the same odd/high word. A comparison ancestor cannot be high. The recorded pair has equal initial low2 and different target bit1, so any such chain, even with arbitrary computed amounts/denominators, is impossible. The same holds with one first MAD/bitop before the chain, since its low2 is equal on the pair.',
                    includes='T,D,S; D,S,S; S,D,S; division/shift/comparison-only graphs of any length; T is first and no later mixing T is allowed.'),
                evidence_sha256=evidence, remaining_in_stated_family=0,
                best_executable_cycles=864, new_executable=False,
                scope='Final DIV/CDIV after at most two MAD/XOR/AND/OR/SHL/SHR/comparison operations on nine live words and root-only constants; denominator may be computed. No other divisions in the prefix. Final REM is excluded for any prefix. Does not close division-first plus shifts, computed division in the middle, or multiple DIV/CDIV.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Final single DIV/CDIV core and arbitrary-prefix final REM excluded', flush=True)


if __name__ == '__main__':
    main()
