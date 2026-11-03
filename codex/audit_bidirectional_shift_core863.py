"""Extend the three-op word-macro audit to data-dependent left shifts.

This covers MAD/XOR/AND/OR and both shifts, not division/comparison/FLOW
or other inputs. It combines the right-shift audit with low16 partition
counts, an odd/high-output ancestor pair, and odd-output count probes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import permutations
from pathlib import Path

import z3

from codex.audit_runtime_shift_core863 import high_samples
from codex.search_early_boundary863 import early_inputs, wanted, NAMES, NODE, MASK, legal_point
from codex.close_early_boundary863 import runtime_amount_probe


def audit():
    source = 'codex/runtime_shift_core863_audit.json'
    encoded = Path(source).read_bytes(); right = json.loads(encoded)
    assert right['node'] == NODE and right['remaining_in_stated_three_op_core'] == 0
    samples = high_samples(); pair = None
    for b, v, y in samples:
        other = b ^ (1 << 17); w = early_inputs(other); yy = wanted(w)
        if y & 1 and yy >= 1 << 31 and yy & 1 and min(w) >= 32:
            assert [x & 3 for x in v] == [x & 3 for x in w] and (y ^ yy) & 2
            pair = [b, other]; break
    assert pair is not None
    a, c = z3.BitVecs('left_source left_count', 32)
    solver = z3.SolverFor('QF_BV'); solver.add((a << c) & 1 == 1, c != 0)
    assert solver.check() == z3.unsat
    solver = z3.SolverFor('QF_BV'); solver.add((a << a) & 1 == 1)
    assert solver.check() == z3.unsat
    rows = []
    for src, name in enumerate(NAMES):
        maximum = 8191 if src == 7 else MASK
        for shift in range(33):
            if maximum >> shift == 0:
                rows.append(dict(source=src, name=name, shift=shift, status='identically_zero',
                                 certificate='Final left shift count0; the middle T cannot transfer b16 to bit0.'))
                continue
            take = next(((b, v, y) for b, v, y in samples if y & 1 and v[src] >> shift), None)
            assert take is not None, (src, shift)
            b, v, y = take
            assert y >= 1 << 31 and y & 1 and v[src] >> shift > 0
            rows.append(dict(source=src, name=name, shift=shift, status='odd_target_with_nonzero_count',
                             primary_b=b, first_right_shift=v[src] >> shift,
                             target=y, legal_point=legal_point(b)))
    no_right = runtime_amount_probe()
    assert len({y & 7 for y in no_right['target']}) == 8
    # Exhaustive finite controls for the low-state branching lemma. For a
    # fixed source low k and count low k, all possible full counts produce
    # either that common small shift or zero modulo 2^k.
    checked = 0
    for width, low_width in ((4, 2), (8, 3)):
        mask = (1 << low_width)-1
        for low_value in range(1 << low_width):
            for low_count in range(1 << low_width):
                values = {((low_value << amount) & mask) if amount < low_width else 0
                          for amount in range(low_count, 1 << width, 1 << low_width)}
                assert len(values) <= 2
                checked += 1
    mixed = {
        'TRL/TLR':'T first: on odd/high target outputs every left/right shift on the value chain has count0. The T ancestor has equal low2 on the recorded pair, contradicting different target bit1.',
        'LRT':'If first left amount is prepared-root constant, normalize L to MAD and use right-core. Otherwise its initial-word amount makes it0 on the rectangle; the second R is literal or has count0, covered by rectangle tests.',
        'RLT':'Prepared-root second left amounts normalize to MAD. Initial-word amounts make L=0 on the rectangle. If second count=first_q, the existing conditional systems use only shift>=32 gives0 and shift0 gives identity, valid for L too.',
        'LTR':'Prepared-root first L normalizes to MAD. An initial-word first amount makes L=0 on all high samples; the middle MAD/bitop high-output tests apply to q=0. Other final value ancestors use the odd/high pair.',
        'RTL':'For a final left shift, an odd target requires count0. Initial-word counts are positive at the recorded odd/high point. Root-constant counts reduce to0 or fail. Count=middle T with identical data is always even; other T-count cases have an initial/shift ancestor and use the odd/high pair. Count=first_q has an odd-target nonzero-count witness for all269 nonzero fixed first-shift cases; zero cases carry no high-to-low information. Initial-word first amounts make first_q=0 on the pair.',
    }
    assert {p for group in mixed for p in group.split('/')} == {''.join(p) for p in permutations('TRL')}
    assert sum(r['status'] == 'identically_zero' for r in rows) == 28
    return dict(node=NODE, input_names=NAMES,
                right_core_evidence=dict(path=source, sha256=hashlib.sha256(encoded).hexdigest()),
                odd_high_ancestor=dict(primary_pair=pair, legal_points=[legal_point(b) for b in pair],
                    left_odd_requires_count0_full32='unsat', left_self_shift_always_even_full32='unsat',
                    proof='An odd/high output forces count0 on every left/right shift on its value-operand ancestor chain. Initial/T-only ancestors have equal low2 on the pair, while target bit1 differs.'),
                first_right_shift_final_left_count_cases=rows,
                no_right_shift=dict(eight_state_fibre=no_right, low_partition_control_cases=checked,
                    maximum_low16_states_with_two_left_shifts=4,
                    required_target_low16_states=8,
                    proof='A T op does not split a low16 state class. A left shift splits it at most twice: one common count<16, or zero for larger counts. At most2 left shifts give at most4 classes. Three shifts with no T use the odd/high ancestor pair.'),
                mixed_direction_partition=mixed,
                all_three_shift_directions='odd/high ancestor pair, all counts on value chain forced0',
                remaining_in_stated_three_op_core=0, best_executable_cycles=864, new_executable=False,
                scope='At most3 straight-line uint32 MAD/XOR/AND/OR/SHL/SHR operations on nine stated live words, arbitrary root-prepared constants and arbitrary data-dependent shift operands. ADD/SUB/MUL normalize to MAD. No DIV/REM/CDIV/comparison/vbroadcast/FLOW/memory/cross-lane operations or larger windows; no global kernel bound.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('297 odd-output count cases; all mixed shift directions covered; stated three-op core remaining0', flush=True)


if __name__ == '__main__':
    main()
