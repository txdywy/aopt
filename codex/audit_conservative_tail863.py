"""One ordinary tail cannot repair an arbitrary word0<=q<=initial N.

The computation of q is otherwise completely unconstrained and may
depend on every input. Legal N=0,1 (or two N=0 witnesses) give finite
necessary systems. A q-only permutation tail additionally forces q=N.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations_with_replacement
import json
from pathlib import Path

from codex.audit_denominator_products863 import intersect, linear_class, square_possible
from codex.audit_shift_division863 import shift_zero_pair
from codex.cegis_c1_absorb863 import _instr_options
from codex.search_early_boundary863 import MASK, NAMES, NODE, early_inputs, wanted, legal_point


def scalar_initial_fits(values, targets, second_bound):
    fits = []
    for initial in range(9):
        first = linear_class(values[0][initial], targets[0])
        for q in range(second_bound+1):
            cls = intersect(first, linear_class(values[1][initial], targets[1]-q))
            if cls is not None: fits.append(dict(initial=initial, q_at_second=q, scalar_class=list(cls)))
    return fits


def factor_offset_fits(values, targets, second_bound):
    difference = (targets[1]-targets[0]) & MASK
    return [dict(initial=j, q_at_second=q) for j in range(9) for q in range(second_bound+1)
            if (q*values[1][j]) & MASK == difference]


def fixed_product_fits(values, target):
    return [[i, j] for i, j in combinations_with_replacement(range(9), 2)
            if (values[i]*values[j]) & MASK == target]


def tail_class(op, wiring):
    q = 9
    if q not in wiring: return 'T_only_parity'
    if not any(j < 9 for j in wiring): return 'q_only_permutation_and_order'
    if op != 'mad': return 'zero_bitop_initial'
    a, b, c = wiring
    if q in (a, b):
        return 'zero_returns_initial' if c < 9 else 'zero_output' if c == q else 'initial_factor_prepared_offset'
    assert c == q
    return 'zero_fixed_product' if a < 9 and b < 9 else 'scalar_initial_product'


def audit():
    parity_path = Path('codex/division_first_t863_audit.json')
    parity_raw = parity_path.read_bytes(); parity = json.loads(parity_raw)
    assert parity['node'] == NODE
    facts = parity['full32_class_facts']
    assert len(facts['initial_lsb_depends_only_on_b0_b19']) == 9
    assert all(row['status'] == 'unsat' for row in facts['initial_lsb_depends_only_on_b0_b19'])
    assert facts['target_b16_flip_full32'] == 'unsat'
    rows = []; legal = set()
    for source, name in enumerate(NAMES):
        bs = shift_zero_pair(source); legal.update(bs)
        vv = [early_inputs(b) for b in bs]; yy = [wanted(v) for v in vv]
        bound = vv[1][source]
        assert vv[0][source] == 0 and bound in (0, 1)
        assert yy[0] and all(x != yy[0] for x in vv[0])
        scalar = scalar_initial_fits(vv, yy, bound)
        offset = factor_offset_fits(vv, yy, bound)
        fixed = fixed_product_fits(vv[0], yy[0])
        assert not scalar and not offset and not fixed
        difference = (yy[1]-yy[0]) & MASK
        assert difference not in range(bound*bound+1)
        assert not square_possible(yy[0])
        rows.append(dict(source=source, name=name, primary_pair=bs, initial_values=vv, targets=yy,
                         q_second_bound=bound, scalar_initial_fits=scalar,
                         initial_factor_prepared_offset_fits=offset, fixed_initial_product_fits=fixed,
                         target_zero_not_initial_or_zero=True, target_at_zero_is_not_word_square=True,
                         target_difference=difference, product_offset_increment_maximum=bound*bound,
                         chain_product_offset_excluded=True))
    manifest = [dict(op=op, wiring=list(w), certificate=tail_class(op, w))
                for op in ('mad', '^', '&', '|') for w in _instr_options(9, op)]
    manifest += [dict(op=op, wiring=[9, 9], certificate=tail_class(op, [9, 9])) for op in ('^', '&', '|')]
    return dict(node=NODE, input_names=NAMES, results=rows, tail_manifest=manifest,
                initial_parity_evidence=dict(path=str(parity_path), sha256=hashlib.sha256(parity_raw).hexdigest()),
                q_only_root_proof='If output is a function only of q/root constants and is a full32 permutation, q must be a full32 permutation. For the seven permutation initials N,0<=q<=N and equality of finite sums force q=N pointwise; the remaining one T then cannot change b16 parity. The other initials v/s19 have maximum below MASK, so q cannot be a full32 permutation.',
                initial_factor_offset_proof='At N=0,q=0 pins H=Y0 in initial*q+H. At the second witness q is between0 and the recorded bound, and none of these outputs equals the target.',
                scalar_initial_product_proof='At N=0,q=0 requires G*initial=Y0. At the second witness enumerate every0<=q<=N and intersect exact fullword congruence classes for G, including nonunits. No class survives.',
                two_value_chain_proof='For0<=q2<=q1<=N, any final T using both q1/q2 is0 at N=0, returns an initial addend unequal to Y0, or is q1*q2+H. In the last case H=Y0 and the next witness permits an increment at most N1^2; the target difference exceeds this bound. A last T reading only one of q1/q2 is covered by the arbitrary bounded-q tail certificate.',
                legal_points=[legal_point(b) for b in sorted(legal)],
                complete_in_stated_scope=True, best_executable_cycles=864, new_executable=False,
                scope='Any arbitrary computed word q(b) satisfying0<=q(b)<=one of the nine initial N(b), followed by a single MAD/XOR/AND/OR on initials/q/root constants. Also two computed value-chain words0<=q2<=q1<=N followed by one such T. The construction of q/q1/q2 is otherwise unrestricted; this is a local target-specific certificate.',
                no_global_optimality_claim=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Arbitrary bounded-tail sources:', len(result['results']), 'all excluded', flush=True)


if __name__ == '__main__':
    main()
