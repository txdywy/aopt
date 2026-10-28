"""Rule out a sole data-dependent SHR whose amount is one bitwise result.

Nine original live words include e. Eight legal states have identical
initial low16 and eight distinct target low3 values. Two shift amounts
>=32 therefore suffice to contradict a macro whose other ops are T ops.
Unknown XOR/AND/OR constants are handled without bounding their values.
This leaves MAD-computed amounts and multiple downward shifts open.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations_with_replacement
import hashlib
import json
from pathlib import Path
import random

import z3

from codex.search_early_boundary863 import early_inputs, wanted, NAMES, NODE, legal_point
from codex.search_full_live_shift863 import zero_witnesses, inverse_target, live_words


def fibres():
    rng = random.Random(8642841)
    bases = [0, 1, 2, 3, 31, 32, 63, 65535, 0xfff80000, 0xfff8ffff,
             *(rng.getrandbits(32) & ~(7 << 16) for _ in range(128))]
    result = []
    for base in bases:
        assert base & (7 << 16) == 0
        bs = [base | (j << 16) for j in range(8)]
        values = [early_inputs(b) for b in bs]
        low = [v & 65535 for v in values[0]]
        assert all([v & 65535 for v in vv] == low for vv in values)
        target = [wanted(v) for v in values]
        assert len({y & 7 for y in target}) == 8
        result.append(dict(base=base, primary_b=bs, values=values, target=target,
                           common_low16=low))
    return result


def zero_points():
    result = {}
    for row in zero_witnesses('tail'):
        b = live_words(row['prior'], 'tail')[4]
        result[row['name']] = b
    p = inverse_target(0, 'middle'); result['e'] = live_words(p, 'tail')[4]
    assert set(result) == set(NAMES)
    for i, name in enumerate(NAMES):
        assert early_inputs(result[name])[i] == 0
    return result


def audit():
    probes = fibres(); used = set(); pair_cases = []; zero = zero_points()
    for op in ('^', '&', '|'):
        for left, right in combinations_with_replacement(range(9), 2):
            row = dict(op=op, operands=[left, right])
            if left == right:
                row['certificate'] = 'fixed_zero_amount' if op == '^' else 'initial_word_amount'
                used.add(0)
            else:
                found = None
                for index, fibre in enumerate(probes):
                    vv = fibre['values']
                    amounts = [v[left] ^ v[right] if op == '^' else
                               v[left] & v[right] if op == '&' else v[left] | v[right] for v in vv]
                    if sum(a >= 32 for a in amounts) >= 2:
                        found = index; break
                assert found is not None, row
                row.update(certificate='at_least_two_large_amounts', fibre=found, amounts=amounts)
                used.add(found)
            pair_cases.append(row)
    xor_cases, or_cases, and_cases = [], [], []
    for source, name in enumerate(NAMES):
        allowed = None; checks = []
        for index, fibre in enumerate(probes):
            counts = Counter(v[source] >> 5 for v in fibre['values'])
            these = {v for v, count in counts.items() if count >= 7}
            allowed = these if allowed is None else allowed & these
            checks.append(dict(fibre=index, allowed_constant_high27=sorted(these)))
            used.add(index)
            if not allowed:
                break
        assert not allowed
        xor_cases.append(dict(source=source, name=name, checks=checks,
                              certificate='No K_high27 makes at least7 of8 XOR amounts smaller than32 on every fibre'))
        amounts = [v[source] for v in probes[0]['values']]
        assert sum(a >= 32 for a in amounts) >= 2
        or_cases.append(dict(source=source, name=name, fibre=0,
                             large_amounts_without_K=sum(a >= 32 for a in amounts),
                             certificate='V OR K >= V for unsigned words'))
        used.add(0)
        # v has low9=0; a>>19 has high19=0. Every other word is a full
        # permutation. Prove the irrelevant high mask positions exactly.
        irrelevant = sum(1 << j for j in (range(5, 9) if name == 'v' else
                                          range(13, 32) if name == 'a_shr19' else ()))
        if irrelevant:
            b = z3.BitVec('amount_irrelevant_primary', 32)
            s = z3.SolverFor('QF_BV'); s.add(early_inputs(b)[source] & irrelevant != 0)
            assert s.check() == z3.unsat
        bits = []
        for bit in range(5, 32):
            if irrelevant >> bit & 1:
                continue
            found = None
            for index, fibre in enumerate(probes):
                count = sum((v[source] >> bit) & 1 for v in fibre['values'])
                if count >= 2:
                    found = index; break
            assert found is not None, (source, bit)
            bits.append(dict(bit=bit, fibre=found, one_count=count)); used.add(found)
        b0 = zero[name]; b1 = b0 ^ (1 << 16)
        aa, bb = early_inputs(b0), early_inputs(b1)
        assert aa[source] == 0 and bb[source] & 31 == 0
        assert aa[source] & irrelevant == bb[source] & irrelevant == 0
        assert [v & 1 for v in aa] == [v & 1 for v in bb]
        assert (wanted(aa) ^ wanted(bb)) & 1
        and_cases.append(dict(source=source, name=name, irrelevant_mask=irrelevant,
                              forced_zero_mask_bits=bits, effective_remaining_mask=31,
                              zero_amount_pair=[b0, b1],
                              legal_points=[legal_point(b0), legal_point(b1)],
                              certificate='Active mask bits>=5 force two zero shift results; after removing them, the zero pair gives identical output bit0 but different target bit0'))
    t = z3.BitVec('self_shift_word', 32)
    s = z3.SolverFor('QF_BV'); s.set(timeout=10000); s.add(z3.LShR(t, t) != 0)
    assert s.check() == z3.unsat
    checks = 0
    for width, low_width in ((4, 2), (8, 3)):
        for c in range(1 << width):
            for low in range(1 << low_width):
                out = {c >> amount for amount in range(low, 1 << width, 1 << low_width)}
                assert len(out) <= 2
                checks += 1
    selected = [dict(index=i, **probes[i], legal_points=[legal_point(b) for b in probes[i]['primary_b']])
                for i in sorted(used)]
    fixed_scope = 'codex/early_boundary863_closed_audit.json'
    prior = json.loads(Path(fixed_scope).read_bytes())
    assert prior['remaining_in_stated_core_grammar'] == 0 and prior['node'] == NODE
    return dict(node=NODE, input_names=NAMES, fibres=selected,
                pair_cases=pair_cases, pair_counts=dict(Counter(r['certificate'] for r in pair_cases)),
                xor_constant_cases=xor_cases, or_constant_cases=or_cases, and_constant_cases=and_cases,
                first_bitwise_amount_remaining=0, self_source_and_amount_full32='unsat',
                constant_shift_source=dict(exhaustive_small_cases=checks,
                    proof='On each eight-state fibre every T-computed amount shares low16. For a constant source C, the shift result is either C>>r (one common r<32) or zero, so output low16 has at most2 values rather than8.',
                    includes_MAD_computed_amount=True),
                fixed_scope_evidence=dict(path=fixed_scope, sha256=hashlib.sha256(Path(fixed_scope).read_bytes()).hexdigest()),
                best_executable_cycles=864, new_executable=False,
                scope='Exactly one data-dependent SHR; amount produced by one first XOR/AND/OR over nine live words and arbitrary constants; remaining ops are T ops. Also constant shift sources with T-computed amounts and t>>t. Does not close MAD-computed amounts with an initial data source or multiple downward shifts.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Bitwise data pairs', result['pair_counts'], flush=True)
    print('Unknown XOR/OR/AND constants:9 sources each closed; self-shift full32 UNSAT; constant-source cases closed', flush=True)


if __name__ == '__main__':
    main()
