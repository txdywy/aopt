"""Extend the local three-op boundary exclusion to Boolean comparisons.

Comparisons are relaxed to an independently chosen Boolean at each legal
input. The audit covers MAD/XOR/AND/OR, both variable shifts and < / ==,
with nine initial words and arbitrary root-prepared uint32 constants.
Division, memory, FLOW, extra inputs and larger windows are not covered.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from math import gcd
from pathlib import Path

from codex.close_early_boundary863 import rectangle
from codex.search_early_boundary863 import C1, MASK, NAMES, NODE, early_inputs, wanted, legal_point


@lru_cache(maxsize=1)
def fibre():
    a0 = rectangle()['coordinates'][0][6] & ~(7 << 16)
    aa = [a0 | (i << 16) for i in range(8)]
    bs = [a ^ (a >> 19) ^ C1 for a in aa]
    vv = [early_inputs(b) for b in bs]
    low16 = [v & 65535 for v in vv[0]]
    assert all([v & 65535 for v in row] == low16 for row in vv)
    assert min(min(v) for v in vv) >= 32
    targets = [wanted(v) for v in vv]
    assert len({t & 65535 for t in targets}) == 8
    assert len({t & 7 for t in targets}) == 8
    assert sum(t >= 1 << 31 for t in targets) == 4
    assert sum(t & 1 for t in targets) == 4
    low17_counts = [len({v[j] & 131071 for v in vv}) for j in range(9)]
    assert max(low17_counts) <= 2
    return dict(primary_b=bs, initial_values=vv, common_low16=low16,
                target=targets, low17_class_counts=low17_counts,
                legal_points=[legal_point(b) for b in bs])


def fitting_forms(q, y, width=16):
    """All single-MAD low-word forms using both q and a Boolean c.

    Constants/initial words are freely relaxed to arbitrary low-word
    constants. The Boolean is independently selectable at every point.
    """
    modulus = 1 << width; mask = modulus-1
    q, y = [int(x) & mask for x in q], [int(x) & mask for x in y]
    assert q and len(q) == len(y)
    found = {}
    if all(x == 0 for x in q):
        if all(t in (0, 1) for t in y):
            found['A*q+c'] = 0
    else:
        i = min((i for i, x in enumerate(q) if x), key=lambda i: gcd(q[i], modulus))
        common = gcd(q[i], modulus); reduced = modulus//common
        for c in (0, 1):
            rhs = (y[i]-c) & mask
            if rhs % common:
                continue
            base = ((rhs//common)*pow(q[i]//common, -1, reduced)) % reduced
            for k in range(common):
                a = base+k*reduced
                if all(((t-a*x) & mask) in (0, 1) for x, t in zip(q, y)):
                    found['A*q+c'] = a
                    break
            if 'A*q+c' in found:
                break
    residuals = {((t-x) & mask) for x, t in zip(q, y)}-{0}
    if len(residuals) <= 1:
        found['A*c+q'] = next(iter(residuals), 0)
    offsets = {y[0], (y[0]-q[0]) & mask}
    for x, t in zip(q[1:], y[1:]):
        offsets &= {t, (t-x) & mask}
    if offsets:
        found['q*c+B'] = min(offsets)
    if all(t in (x, (2*x) & mask) for x, t in zip(q, y)):
        found['q*c+q'] = True
    if all(t in (0, (x+1) & mask) for x, t in zip(q, y)):
        found['q*c+c'] = True
    if all(((t-x*x) & mask) in (0, 1) for x, t in zip(q, y)):
        found['q*q+c'] = True
    return found


def audit():
    path = Path('codex/bidirectional_shift_core863_audit.json')
    prior_bytes = path.read_bytes(); prior = json.loads(prior_bytes)
    assert prior['node'] == NODE and prior['remaining_in_stated_three_op_core'] == 0
    ff = fibre(); vv, yy = ff['initial_values'], ff['target']
    lows = [y & 65535 for y in yy]; rows = []
    for source, name in enumerate(NAMES):
        for shift in range(33):
            q = [(v[source] >> shift) & 65535 for v in vv]
            forms = fitting_forms(q, lows)
            assert not forms, (source, shift, forms)
            # XOR/OR with a Boolean leave every bit >=1 unchanged.
            assert any((a >> 1) != (b >> 1) for a, b in zip(q, lows))
            rows.append(dict(source=source, source_name=name, right_shift=shift,
                             shifted_low16=q, fitting_mad_forms=forms,
                             xor_or_high15_mismatch=True, and_range_at_most_one=True,
                             status='excluded'))
    # Comparisons are represented only by their actual output range.
    # A final comparison cannot match even one output >1 on this fibre.
    assert min(yy) > 1
    forms = ['A*q+c', 'A*c+q', 'q*c+B', 'q*c+q', 'q*c+c', 'q*q+c']
    partition = {
        'no_shifts': 'At most two non-final comparisons split the common low16 class at most four ways. MAD and bit operations do not split a class; eight distinct target low16 values are required.',
        'CST/SCT_literal_right': 'A final MAD using both a Boolean c and a literal initial-word right shift q is one of six recorded forms. All nine times33 source/count cases are excluded. XOR/OR preserve q bits>=1; AND is <=1. Dead comparisons or dead shifts reduce to the prior core.',
        'CST_boolean_amount': 'A comparison count is0/1. For an initial-word right-shift datum, count0 gives one low16 class; count1 gives at most two by the recorded low17 counts, at most three (c,q) classes total. A left shift gives at most two classes. Constant/Boolean data give at most two. Eight distinct outputs are impossible.',
        'CST/SCT_initial_amount': 'Every initial-word amount is>=32 at all eight recorded inputs. Either shift produces0, so the remaining Boolean and initial words yield at most two low16 outputs.',
        'CST/SCT_literal_left': 'A prepared-constant left shift normalizes to a T operation. With one comparison, low16 can split at most two ways.',
        'CTS/TCS': 'Before the final shift, the value datum has at most two low16 classes. A high target requires final right-shift count0 at four distinct low16 targets; an odd target requires final left-shift count0 at four distinct low16 targets. Both contradict the two-class bound.',
        'CSS/SCS': 'On the prior odd/high ancestor pair, every shift on the value-operand chain has count0. Initial/root-constant ancestors have equal low2 but targets differ at bit1. A Boolean ancestor cannot be high. This also covers two comparisons plus one final shift.',
        'final_C': 'A final comparison returns0/1, whereas all recorded targets are>1.',
        'two_C_one_T': 'Two Boolean results give at most four low16 states to the final T, fewer than eight target states.',
    }
    return dict(node=NODE, input_names=NAMES,
                prior_core=dict(path=str(path), sha256=hashlib.sha256(prior_bytes).hexdigest()),
                eight_state_fibre=ff, six_mad_forms=forms, literal_right_cases=rows,
                odd_high_ancestor=prior['odd_high_ancestor'], order_partition=partition,
                remaining_in_stated_three_op_core=0, best_executable_cycles=864, new_executable=False,
                scope='At most3 straight-line uint32 MAD/XOR/AND/OR/SHL/SHR/< /== operations, nine stated live words, arbitrary root-only constants and arbitrary shift/comparison operands. Comparisons relaxed to arbitrary Boolean per input. ADD/SUB/MUL normalize to MAD. No DIV/REM/CDIV, vbroadcast, FLOW, memory, cross-lane, extra inputs or larger windows; no global kernel lower bound.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('297 literal right-shift cases times six Boolean/MAD forms excluded; comparison core remaining0', flush=True)


if __name__ == '__main__':
    main()
