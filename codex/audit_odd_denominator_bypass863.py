"""Universal odd-prefix collisions with a multiplied initial-word bypass.

If P is equal on two inputs, Y=I*P+f(P) requires dY=P*dI.
For REM(C,P), P must be odd on every input: an even P and the b16
partner have equal remainder/output parity while the target flips.
The finite odd-prefix manifest therefore covers every possible P.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import z3

from codex.audit_division_first_t863 import source_inverse
from codex.audit_odd_prefix_collisions863 import evaluate, key_grid, manifest
from codex.search_early_boundary863 import C1, MASK, NAMES, NODE, early_inputs, wanted


def additional_pairs(row):
    pairs = []
    for source in (0, 1, 2, 4, 5, 6, 8):
        for x in [0, *(1 << j for j in range(1, 32)), 0x12345678, 0xABCDEFFE]:
            pairs.append([source_inverse(source, x), source_inverse(source, x ^ 1)])
    # Equal products are found and checked as full words. The bit lifting
    # is only a witness finder; a failed inversion never proves exclusion.
    if row['op'] == 'mad' and row['wiring'][2] >= 9:
        for b in [*range(8), 0x12345678, 0x20000, 0x1234FFFF]:
            target = evaluate('mad', row['wiring'], early_inputs(b), [0, 0])
            partner = b ^ 1
            for bit in range(1, 32):
                got = evaluate('mad', row['wiring'], early_inputs(partner), [0, 0])
                if ((got ^ target) >> bit) & 1: partner ^= 1 << bit
            if evaluate('mad', row['wiring'], early_inputs(partner), [0, 0]) == target:
                pairs.append([b, partner])
    # The v*s19 product is zero on this entire s19=0 subspace.
    for a in [*range(32), 65536, 131072, 262144]:
        pairs.append([C1, C1 ^ a])
    return pairs


def admissible_pairs(row, parity, pairs):
    grid = key_grid(row['op'], parity); out = []
    for pair in pairs:
        if pair[0] == pair[1]: continue
        vv = [early_inputs(b) for b in pair]
        if all(evaluate(row['op'], row['wiring'], vv[0], keys) ==
               evaluate(row['op'], row['wiring'], vv[1], keys) for keys in grid):
            if tuple(pair) not in {tuple(p) for p in out}: out.append(pair)
    return out


def solve_case(row, parity, pairs, factor, seconds=10, target=wanted):
    kk = [z3.BitVec(f'odd_den_key_{j}', 32) for j in range(2)]
    conditions = [kk[j-9] & 1 == parity[j-9] for j in sorted(set(row['wiring']) & {9, 10})]
    solver = z3.SolverFor('QF_BV'); solver.set(timeout=int(seconds*1000)); solver.add(*conditions)
    for index, pair in enumerate(pairs):
        v, w = [early_inputs(b) for b in pair]
        p = evaluate(row['op'], row['wiring'], v, kk)
        if not z3.is_bv(p): p = z3.BitVecVal(p, 32)
        equation = p*((v[factor]-w[factor]) & MASK) == (target(v)-target(w)) & MASK
        solver.assert_and_track(equation, z3.Bool(f'witness_{index}'))
    status = solver.check()
    result = dict(op=row['op'], wiring=row['wiring'], key_parity=parity,
                  factor=factor, factor_name=NAMES[factor], status=str(status),
                  universal_collision_candidates=len(pairs))
    if status == z3.unsat:
        indices = sorted(int(str(label).split('_')[-1]) for label in solver.unsat_core())
        witnesses = []
        for index in indices:
            v, w = [early_inputs(b) for b in pairs[index]]
            a = evaluate(row['op'], row['wiring'], v, kk)
            b = evaluate(row['op'], row['wiring'], w, kk)
            independent = z3.SolverFor('QF_BV'); independent.set(timeout=int(seconds*1000))
            independent.add(*conditions, a != b)
            check = independent.check()
            assert check == z3.unsat, (row, parity, pairs[index], check)
            witnesses.append(dict(primary_pair=pairs[index], full32_collision_refutation=str(check),
                                  target_difference=(target(v)-target(w)) & MASK,
                                  factor_difference=(v[factor]-w[factor]) & MASK))
        result['witnesses'] = witnesses
        result['full32_bypass_equations_refuted'] = True
    elif status == z3.sat:
        result['necessary_keys_only'] = [solver.model().eval(k, model_completion=True).as_long() for k in kk]
    else:
        result['reason'] = solver.reason_unknown()
    return result


def audit():
    path = Path('codex/odd_prefix_collisions863_refinement.json')
    old = json.loads(path.read_text()); rows = []; original = manifest()
    assert sum(len(row['odd_key_parities']) for row in original) == 64
    assert len(original) == 702
    expected = {(row['op'], tuple(row['wiring']), tuple(parity))
                for row in original for parity in row['odd_key_parities']}
    actual = {(c['op'], tuple(c['wiring']), tuple(c['key_parity'])) for c in old['cases']}
    assert expected == actual and len(old['cases']) == 64
    for case in old['cases']:
        row = dict(op=case['op'], wiring=case['wiring']); parity = case['key_parity']
        pairs = admissible_pairs(row, parity,
                                  [w['primary_pair'] for w in case['witnesses']] + additional_pairs(row))
        for factor in (0, 6):
            result = solve_case(row, parity, pairs, factor); rows.append(result)
            if result['status'] != 'unsat': print('Remaining', result, flush=True)
    return dict(node=NODE, input_names=NAMES, previous_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                canonical_prefixes=len(original), always_odd_parameter_cases=64,
                factor_sources=[0, 6], results=rows,
                excluded=sum(r['status'] == 'unsat' for r in rows),
                unknown=sum(r['status'] == 'unknown' for r in rows),
                scope='P one MAD/XOR/AND/OR, arbitrary root-only keys, P always odd; Y=I*P+f(P) for I=e or a and arbitrary f(P). This closes the remaining prepared-numerator REM(C,P) tails because even P is excluded by b16 parity.',
                constant_prefix_excluded='A prepared-only P and f(P) leave an affine initial-word output whose parity cannot flip on b16.',
                best_executable_cycles=864, new_executable=False, no_global_optimality_claim=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Odd denominator bypass cases:', result['excluded'], '/', len(result['results']), flush=True)


if __name__ == '__main__':
    main()
