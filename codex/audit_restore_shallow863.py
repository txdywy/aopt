"""Combine shallow CDCL checks with normalized polynomial certificates.

The symbolic classifier keeps x, the one bitwise result q, and every private
constant as separate indeterminates. A circuit is assigned to G*q+R2(x) only
when its exact expanded polynomial has no x*q, q**2 or higher x terms. This
does not assume that a runtime multiplier is a constant.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from codex.search_cross_encoding863 import PRE_C5


N = 10  # x, q, eight private constants
ONE = (0,)*N
MASK = (1 << 32)-1


def variable(i):
    powers = [0]*N
    powers[i] = 1
    return {tuple(powers): 1}


def add(a, b):
    out = dict(a)
    for term, c in b.items():
        out[term] = (out.get(term, 0)+c) & MASK
        if not out[term]:
            del out[term]
    return out


def multiply(a, b):
    out = {}
    for u, c in a.items():
        for v, d in b.items():
            term = tuple(x+y for x, y in zip(u, v))
            out[term] = (out.get(term, 0)+c*d) & MASK
    return {t: c for t, c in out.items() if c}


def mixture(row):
    order, wiring = row['order'], row['wiring']
    if sum(op != 'mad' for op in order) != 1:
        return None
    values = [variable(0)]
    bitop = None
    for i, (op, w) in enumerate(zip(order, wiring)):
        pool = values + [variable(2+2*i), variable(3+2*i)]
        args = [pool[j] for j in w]
        if op == 'mad':
            value = add(multiply(args[0], args[1]), args[2])
        else:
            if op not in ('^', '&', '|'):
                return None
            degrees = sorted(max((t[0] for t in arg), default=0) for arg in args)
            if degrees[0] > 1 or degrees[1] > 2:
                return None
            bitop = op
            value = variable(1)
        values.append(value)
    out = values[-1]
    if any(t[1] > 1 or (t[1] and t[0]) or (not t[1] and t[0] > 2) for t in out):
        return None
    factor = {t[2:]: c for t, c in out.items() if t[1] == 1}
    residual_degree = max((t[0] for t in out if not t[1]), default=0)
    return dict(bitop=bitop, residual_degree=residual_degree,
                unit_scale=factor == {(0,)*8: 1}, terms=len(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path('codex/restore_wiring863_complete_audit.json'))
    ap.add_argument('--kissat', type=Path, default=Path('codex/restore_shallow863_kissat.json'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--last-families', action='store_true')
    ap.add_argument('--final-pair', action='store_true')
    args = ap.parse_args()
    evidence = {}

    def read(path):
        data = path.read_bytes()
        evidence[str(path)] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    source = read(args.source)
    key = lambda r: (tuple(r['order']), tuple(tuple(w) for w in r['wiring']))
    original = {key(r): r for r in source['unresolved']}
    assert len(original) == source['remaining'] == 176
    controls = read(Path('codex/restore_shallow863_normalization_controls.json'))
    assert controls['normalization_programs'] == 240 and controls['inputs_per_program'] == 64
    scaled = read(Path('codex/restore_shallow863_scaled.json'))
    assert scaled['query']['status'] == 'unsat' and not scaled['query']['control']
    assert scaled['query']['width'] == 11 and scaled['query']['output_key'] == PRE_C5
    assert scaled['normalization_control']['programs'] == 80
    proofs = {}
    for op, suffix in (('^', ''), ('&', '_and'), ('|', '_or')):
        path = Path('codex/restore_shallow863_constscale'+suffix+'.json')
        if not path.exists():
            continue
        row = read(path)
        if row['status'] == 'unsat':
            assert row['width'] == 11 and row['degrees'] == [1, 2]
            assert row['family'] == 'product' and row['constant_scale']
            assert row['normalized_input_key'] and not row['control']
            assert row['bitop'] == op and row['output_key'] == PRE_C5
            proofs[op] = str(path)
    counts = Counter()
    closed, remaining = [], []
    for k, row in original.items():
        shape = mixture(row)
        cert = None
        if shape:
            if shape['bitop'] == '^' and shape['residual_degree'] == 0:
                cert = 'normalized_constant_offset_scaled_xor'
            elif shape['bitop'] in proofs:
                cert = 'normalized_quadratic_offset_scaled_'+shape['bitop']
        if cert:
            counts[cert] += 1
            closed.append(dict(order=row['order'], wiring=row['wiring'], shape=shape, certificate=cert))
        else:
            remaining.append(row)
    sat = read(args.kissat)
    assert sat['completed'] == sat['jobs'] == len(sat['rows']) == 65
    inputs = read(Path(sat['source']))
    assert {key(r) for r in inputs['unresolved']} == {key(r) for r in sat['rows']}
    assert all(key(r) in original for r in sat['rows'])
    sat_unsat = {key(r) for r in sat['rows'] if r['status'] == 'unsat'}
    candidates = [r for r in sat['rows'] if r['status'] == 'reduced_sat']
    assert not candidates, 'A reduced-domain candidate needs lifting, not exclusion'
    kept = []
    for row in remaining:
        if key(row) in sat_unsat:
            counts['kissat_60s_shallow_wiring_unsat'] += 1
            closed.append(dict(order=row['order'], wiring=row['wiring'],
                               certificate='kissat_60s_shallow_wiring_unsat'))
        else:
            kept.append(row)
    if args.last_families:
        last_control = read(Path('codex/restore_shallow863_last_controls.json'))
        assert all(r['status'] == 'reduced_sat' and r['verified_inputs'] == 2048
                   for r in last_control['queries'])
        for family in ('mixed', 'quad'):
            proof = read(Path('codex/restore_shallow863_'+family+'.json'))
            assert proof['query']['family'] == family and proof['query']['width'] == 11
            assert proof['query']['status'] == 'unsat' and not proof['query']['control']
            assert proof['normalization_controls']['programs_per_family'] == 80
        equivalence = read(Path('codex/restore_shallow863_equivalence.json'))
        assert equivalence['and_counterpart']['status'] == 'unsat'
        assert equivalence['or_to_and_full32_random_controls'] == 1000
        assert equivalence['three_op_free_key_grammar_excluded_orders'] == 64
        mixed_keys = {
            (('mad','^','mad','^'), ((0,1,2),(0,2),(2,3,4),(1,3))),
            (('^','mad','mad','^'), ((0,1),(0,2,3),(1,3,4),(2,3))),
            (('^','mad','mad','^'), ((0,1),(1,2,3),(0,3,4),(2,3))),
        }
        quad_key = (('mad','^','mad','mad'), ((0,1,2),(1,2),(2,3,4),(2,3,4)))
        or_key = (('mad','|','mad','^'), ((0,1,2),(1,2),(2,3,4),(1,3)))
        merge_key = (('mad','^','^','mad'), ((0,1,2),(1,2),(2,3),(4,5,3)))
        rest = []
        for row in kept:
            k = key(row)
            cert = ('normalized_two_affine_xor_key_family' if k in mixed_keys else
                    'normalized_quadratic_of_affine_xor' if k == quad_key else
                    'or_to_and_complement_equivalence' if k == or_key else
                    'merge_consecutive_constant_xors_to_three_ops' if k == merge_key else None)
            if cert:
                counts[cert] += 1
                closed.append(dict(order=row['order'], wiring=row['wiring'], certificate=cert))
            else:
                rest.append(row)
        kept = rest
    if args.final_pair:
        assert args.last_families
        pair_control = read(Path('codex/restore_shallow863_final_pair_controls.json'))
        assert all(r['status'] == 'reduced_sat' and r['verified_inputs'] == 2048
                   for r in pair_control['queries'])
        for family in ('xor_affine', 'sum'):
            proof = read(Path('codex/restore_shallow863_'+family+'.json'))
            q = proof['query']
            assert q['family'] == family and q['width'] == 11
            assert q['status'] == 'unsat' and not q['control']
            assert q['target_output_key'] == PRE_C5 and q['normalized_input_key']
            assert proof['normalization_controls']['programs_per_family'] == 80
            assert proof['normalization_controls']['sum_offset_orbit_controls'] > 0
            if family == 'sum':
                assert q['sum_offset'] == 'exact four-case orbit'
        pair = {
            (('mad','^','mad','^'), ((0,1,2),(1,2),(2,3,4),(2,3))): 'normalized_xor_of_affine_xor_word',
            (('^','mad','^','mad'), ((0,1),(1,2,3),(2,3),(0,4,3))): 'normalized_sum_with_exact_offset_orbit',
        }
        rest = []
        for row in kept:
            cert = pair.get(key(row))
            if cert:
                counts[cert] += 1
                closed.append(dict(order=row['order'], wiring=row['wiring'], certificate=cert))
            else:
                rest.append(row)
        kept = rest
    assert len(closed)+len(kept) == len(original)
    result = dict(initial_remaining=176, newly_excluded=len(closed), remaining=len(kept),
                  counts=dict(counts), closed=closed, unresolved=kept,
                  evidence_sha256=evidence, best_executable_cycles=864, new_executable=False)
    result['enumerated_wirings'] = sum(r['jobs'] for r in source['reports'])
    result['remaining_four_instruction_wirings'] = len(kept)
    result['scope'] = ('One live uint32 x; MAD/XOR/AND/OR and arbitrary constants; '
                       'target F(x XOR I) XOR PRE_C5 with arbitrary I. '
                       'No shifts, division, other live inputs, or whole-kernel optimality claim.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('newly_excluded', 'remaining', 'counts')}), flush=True)


if __name__ == '__main__':
    main()
