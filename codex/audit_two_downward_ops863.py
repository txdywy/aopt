"""Two divisions/downward value operations followed by one ordinary T.

An oriented tail G*F+Q is screened with arbitrary0<=F<=initial N and a
concrete initial-word quotient Q. Zero/small N exhaust all possible F.
Prepared/data-dependent SHR of an initial word instead uses its complete
33-value spectrum. Root-data variable SHR is explicitly outside scope.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

from codex.audit_conservative_tail863 import audit as bounded_audit
from codex.audit_division_first_t863 import refine, source_inverse
from codex.audit_high_division_middle863 import suffix_checks
from codex.audit_shift_division863 import point_divisors
from codex.cegis_c1_absorb863 import _instr_options
from codex.screen_division_xor_affine863 import quotient
from codex.search_early_boundary863 import MASK, NAMES, NODE, early_inputs, wanted, legal_point


def oriented_checks(values, targets, concrete_q):
    """F1<=1,F2<=2. Return necessary survivors; arbitrary fullword G."""
    r = (targets[0]-concrete_q[0]) & MASK
    if not r:
        return dict(status='open', reason='F1=0 remains possible; no scalar pinned.')
    outputs = [(concrete_q[1]+r*f) & MASK for f in range(3)]
    initial = [j for j in range(9) if (values[0][j]+concrete_q[0]) & MASK == targets[0]]
    bit = [op for op in ('^', '|') if (concrete_q[0]^1 if op == '^' else concrete_q[0]|1) == targets[0]]
    square = ((concrete_q[0]+1) & MASK) == targets[0]
    scalar_survives = targets[1] in outputs
    return dict(status='excluded' if not scalar_survives and not initial and not bit and not square else 'open',
                forced_scalar=r, scalar_outputs_at_second=outputs,
                scalar_survives=scalar_survives, initial_coefficient_survivors=initial,
                bitop_survivors=bit, prefix_square_survives=square,
                F_at_one_must_equal1=True)


def two_result_class(op, wiring):
    assert 9 in wiring and 10 in wiring
    if op != 'mad': return 'AND_zero' if op == '&' else 'oriented_bitwise'
    a, b, c = wiring
    if c in (9, 10):
        if a == b: return 'square_plus_word_nonsquare'
        if {a, b} == {9, 10}: return 'product_plus_factor_zero'
        other = 9 if c == 10 else 10
        assert other in (a, b)
        coefficient = b if a == other else a
        return 'oriented_initial_coefficient' if coefficient < 9 else 'oriented_prepared_coefficient'
    assert {a, b} == {9, 10}
    return 'product_initial_addend' if c < 9 else 'product_prepared_offset'


def audit():
    evidence = {}
    def read(name):
        path = Path('codex')/name; raw = path.read_bytes(); result = json.loads(raw)
        assert result['node'] == NODE
        evidence[str(path)] = hashlib.sha256(raw).hexdigest()
        return result
    bounded = read('conservative_tail863_audit.json')
    assert bounded == json.loads(json.dumps(bounded_audit()))
    anchors = read('shift_division863_audit.json')
    prior = read('single_division_core863_closed_audit.json')
    last = read('final_division863_audit.json')
    high = read('high_division_middle863_audit.json')
    assert bounded['complete_in_stated_scope'] and prior['complete_in_stated_scope']
    assert last['remaining_in_stated_family'] == 0
    assert high['complete_in_stated_scope']
    assert {(r['source'], r['op']) for r in high['results']} == {
        (src, op) for src in range(9) for op in ('//', 'cdiv')}
    high_classes = {}
    for b in high['high_primary_pool']:
        y = wanted(early_inputs(b))
        if y > 1 << 31: high_classes.setdefault(b & ((1 << 19) | 1), set()).add(y & 1)
    assert len(high_classes) == 4 and all(p == {0, 1} for p in high_classes.values())
    for row in high['results']:
        intervals = [[1, MASK]]; src = row['source']; ceil = row['op'] == 'cdiv'
        for b in row['tested_high_primary']:
            v = early_inputs(b); y = wanted(v); cls = b & ((1 << 19) | 1)
            base_b = row['anchors'][str(cls)]; base = early_inputs(base_b); base_y = wanted(base)
            assert y > 1 << 31 and base_y > 1 << 31 and base_b & ((1 << 19) | 1) == cls
            n, a = v[src], base[src]; assert (n ^ a) & 1 == 0
            desired = (y ^ base_y ^ int(ceil and n > 0) ^ int(ceil and a > 0)) & 1
            nf, af = (max(n-1, 0), max(a-1, 0)) if ceil else (n, a)
            intervals = ([] if desired else intervals) if nf == af else refine(intervals, nf, af, desired)
        assert intervals == row['survivor_intervals']
    assert high['remaining_first_quotient_families'] == [[0, '//', [[65536, 65536]]]]
    suffix = high['remaining_floor_suffix']; vv_high = [early_inputs(b) for b in suffix['primary_b']]
    assert all(wanted(v) > 1 << 31 for v in vv_high)
    assert all(len({v[j] & 65535 for v in vv_high}) == 1 for j in range(9))
    assert [v[0] >> 16 for v in vv_high] == suffix['quotients']
    assert [wanted(v) & 65535 for v in vv_high] == suffix['targets_low16']
    replay_suffix = suffix_checks(suffix['quotients'], suffix['targets_low16'])
    assert replay_suffix['status'] == suffix['status'] == 'excluded'
    assert replay_suffix['quadratic_consistent'] == suffix['quadratic_consistent']
    assert replay_suffix['bitop_fits'] == suffix['bitop_fits']
    certificate = suffix['quadratic_null_certificate']; weights = certificate['weights']
    assert all(sum(w*q**power for w, q in zip(weights, suffix['quotients'])) & 65535 == 0 for power in range(3))
    assert sum(w*y for w, y in zip(weights, suffix['targets_low16'])) & 65535 == certificate['target_residue'] != 0
    collision = suffix['unary_bit_collision']; i, j = collision['pair']; bit = collision['bit']
    assert ((suffix['quotients'][i] ^ suffix['quotients'][j]) >> bit) & 1 == 0
    assert ((suffix['targets_low16'][i] ^ suffix['targets_low16'][j]) >> bit) & 1 == 1
    ancestor_pair = last['odd_high_ancestor_pair']
    ancestors = [early_inputs(p['primary_b']) for p in ancestor_pair]
    ancestor_targets = [wanted(v) for v in ancestors]
    assert all(y > 1 << 31 and y & 1 for y in ancestor_targets)
    assert [x & 3 for x in ancestors[0]] == [x & 3 for x in ancestors[1]]
    assert ancestor_targets[0] & 3 != ancestor_targets[1] & 3
    rows = []; legal = set(); total = 0
    for row in anchors['independent_quotient_results']:
        h, n, op = row['shift_source'], row['numerator'], row['op']
        bs = row['zero_primary_pair']; vv = [early_inputs(b) for b in bs]; yy = [wanted(v) for v in vv]
        legal.update(bs)
        assert vv[0][h] == 0
        domain = point_divisors(vv[0][n], yy[0], op)
        candidates = set(domain['isolated_divisors'])
        for lo, hi in domain['intervals']: candidates.update(range(lo, hi+1))
        assert sorted(candidates) == row['candidate_divisors']
        cases = []
        for d in sorted(candidates):
            total += 1
            q = [quotient(v[n], d, op) for v in vv]
            assert q[0] == yy[0] and q[1] != yy[1]
            if h in (3, 7):
                assert vv[1][h] == 0
                check = dict(status='excluded', reason='Both ancestor values0 force F=0; the second quotient misses the second target.')
            else:
                b2 = source_inverse(h, 2); legal.add(b2); v2 = early_inputs(b2)
                q2 = quotient(v2[n], d, op)
                check = oriented_checks([vv[1], v2], [yy[1], wanted(v2)], [q[1], q2])
                check.update(primary_at_ancestor2=b2, target_at_ancestor2=wanted(v2), concrete_q_at_ancestor2=q2)
            assert check['status'] == 'excluded'
            cases.append(dict(divisor=d, **check))
        rows.append(dict(bounded_ancestor=h, concrete_numerator=n, concrete_op=op,
                         zero_primary_pair=bs, candidate_divisors=sorted(candidates), cases=cases,
                         status='excluded'))
    assert len(rows) == 243 and total == 43
    products = []
    for h in range(9):
        for n in range(9):
            if h in (3, 7) or n in (3, 7):
                src = h if h in (3, 7) else n; r = bounded['results'][src]
                bs = r['primary_pair']; targets = r['targets']; assert r['q_second_bound'] == 0
                method = 'A nonpermutation ancestor has two zero witnesses; one factor is0 at both.'
            elif h != n:
                bs = [bounded['results'][h]['primary_pair'][0], bounded['results'][n]['primary_pair'][0]]
                targets = [wanted(early_inputs(b)) for b in bs]
                assert targets[0] != targets[1]
                method = 'One factor vanishes at each distinct ancestor zero, pinning H to two different targets.'
            else:
                r = bounded['results'][h]; bs, targets = r['primary_pair'], r['targets']
                assert r['target_difference'] not in (0, 1)
                method = 'At ancestor0 the product is0; at ancestor1 both factors are0/1. The target increment is neither0 nor1.'
            products.append(dict(ancestors=[h, n], primary_pair=bs, targets=targets,
                                 method=method, status='excluded'))
    spectral = []
    for row in bounded['results']:
        b = row['primary_pair'][0]; v = early_inputs(b); y = wanted(v)
        assert all(y != x >> s for x in v for s in range(33))
        spectral.append(dict(bounded_ancestor=row['source'], primary_b=b, target=y,
                             all_initial_SHR_spectra_miss=True))
    manifest = [dict(op=op, wiring=list(w), certificate=two_result_class(op, w))
                for op in ('mad', '^', '&', '|') for w in _instr_options(10, op) if 9 in w and 10 in w]
    assert len(manifest) == 37
    ordinary = ('mad', '^', '&', '|'); divisions = ('//', '%', 'cdiv')
    orders = []
    for order in product(ordinary+divisions, repeat=3):
        count = sum(op in divisions for op in order)
        certificate = ('prior_single_division_core' if count <= 1 else 'prior_high_conservative_chain'
                       if count == 3 or order[0] in ordinary else 'new_high_middle_T'
                       if order[1] in ordinary else 'new_two_downward_T')
        orders.append(dict(order=list(order), certificate=certificate))
    assert len(orders) == 343
    return dict(node=NODE, raw_root=0x12345678, input_names=NAMES, evidence_sha256=evidence,
                oriented_concrete_division=rows, exact_concrete_divisor_candidates=total,
                product_offset_ancestor_pairs=products, concrete_initial_SHR_spectrum=spectral,
                two_result_final_manifest=manifest,
                one_result_tail_evidence='codex/conservative_tail863_audit.json',
                chain_case='If the second data/numerator operand is the prior result,0<=q2<=q1<=initial N. The arbitrary bounded-tail and chain certificates cover final operations reading either/both computed results, without assuming a dead dependency or a static shift count.',
                separate_initial_data_case='If both data/numerator operands are initials, each output is bounded by its own initial. A last scalar/initial coefficient times one result plus the other is oriented with the other result as its concrete addend. An initial-data DIV addend has a prepared divisor (initial/prior-result divisors hit0); its entire domain is exhausted by the43 candidate checks. An initial-data SHR addend, even with a data-dependent count, has the recorded complete33-value spectrum at the zero witness. Product/offset, square/addend and AND families use zero/nonsquare witnesses independently of orientation.',
                valid_divisor_partition='A nonconstant first DIV has an initial numerator and nonzero prepared divisor; every initial divisor reaches0. A later divisor equal to an initial or a prior value bounded by an initial also reaches0. Prepared-only divisions fold to root constants and reduce to the earlier single-division core. A second DIV numerator may be initial, prior result or prepared; these are the separate-data, chain or constant cases respectively.',
                excluded_orderings=dict(D_D_T='complete for DIV/REM/CDIV,DIV/REM/CDIV,MAD/XOR/AND/OR, all legal operands/root constants.',
                    initial_data_downward_pair_T='complete for two DIV/REM/CDIV/SHR with data/numerator operands from initials or the previous result; SHR counts may be arbitrary initials/root/prior values. A prepared-only operation can be folded. Variable SHR of a prepared DATA constant is outside this added scope.',
                    T_D_D='Prior high-conservative-chain certificate: a first ordinary T followed by two divisions, no later mixing T.',
                    D_T_D='New high-only first-divisor interval certificate, with its unique e//65536 survivor excluded by the middle-T low16 polynomial/bitop systems.',
                    D_T_SHR='The same high-only certificate: a final right shift above2^31 must have count0 and equal its data, even with a data-dependent count.',
                    D_D_D='Prior high-conservative-chain certificate: division-only value paths cannot produce the target.'),
                ordinary_division_opcode_orders=orders, ordinary_division_three_op_core_complete=True,
                high_prefix_replays=18,
                prepared_shift_normalization='For a prepared count s<32, SHR(x,s) is floor(x/2^s), with a valid prepared uint32 divisor, and SHL(x,s) is MAD(x,2^s,0). Counts>=32 give0 in word semantics. These preserve or reduce the operation count, so the343-order core also covers prepared-count shifts in any placement. This does not normalize data-dependent shift counts.',
                legal_points=[legal_point(b) for b in sorted(legal)],
                complete_in_stated_scope=True, best_executable_cycles=864, new_executable=False,
                still_open=['Prepared-data/data-count SHR mixed with DIV; variable SHL and remaining data-dependent shift/comparison combinations, including REM with comparisons.',
                            'Additional free words, larger joint windows, root-conditional preparation/algorithms, nonlinear encodings, cross-lane and memory mechanisms.'],
                no_global_optimality_claim=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Two-downward oriented cases:', len(result['oriented_concrete_division']), flush=True)
    print('Fullword concrete divisor candidates:', result['exact_concrete_divisor_candidates'], flush=True)
    print('D,D,T and initial-data downward pairs: closed', flush=True)


if __name__ == '__main__':
    main()
