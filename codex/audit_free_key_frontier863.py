"""Combine certificates for all <=3 MAD/XOR/AND/OR programs and XOR keys.

This is a one-input, unshifted middle-hash result. It is not a lower bound
on programs with other live values, other opcodes, or the full kernel.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

from codex.audit_free_key_orders863 import audit as xor_audit
from codex.audit_free_keys_lattice863 import audit as lattice_audit


ROOT = Path(__file__).resolve().parent
LABELS = {'mad':'mad','^':'xor','&':'and','|':'or'}


def audit():
    xor = xor_audit()
    lattice = lattice_audit()
    assert all(v['result'] == 'closed_by_recorded_necessary_conditions' for v in xor['orders'].values())
    assert lattice['excluded_orders'] == 6 and lattice['unresolved_orders'] == 0
    evidence = {}

    def read(name):
        path = ROOT/name
        evidence[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    # Both required rectangle screens must enumerate every endpoint key pair.
    screens = read('free_keys863_bypass_polynomial.json')['experiments']
    for bypass in (False,True):
        matching = [v for v in screens if v['carry_free_rectangles_only'] and v['xor_input'] == bypass]
        assert len(matching) == 1 and matching[0]['remaining_pairs'] == 0
        assert matching[0]['initial_key_pairs'] == 4194304
    polynomial = read('middle_free_keys863_polynomial.json')['experiment']
    assert polynomial['remaining_pairs'] == 0 and polynomial['initial_key_pairs'] == 4194304
    control = read('middle_free_keys863_control_w32.json')
    assert control['width'] == 32 and control['result'] == 'proved_equivalent_full32'
    assert control['pin_original'] and control['input_key'] == control['output_key'] == 0

    def middle(x):
        return (9*(((33*x+0xe9f8cc1d)&2047)^((16896*x+0xaccf6200)&2047))+0xfd7046c5)&2047

    nonlocal_difference = middle(0)^middle(1)
    assert nonlocal_difference & ~1
    orders = []
    for order in itertools.product(('mad','^','&','|'),repeat=3):
        count = order.count('mad')
        row = dict(sequence=list(order),mad_count=count)
        if count == 3:
            row['certificate'] = 'pure integer polynomial, all endpoint keys'
        elif count == 2:
            row['certificate'] = 'XOR wiring audit' if '^' in order else 'AND/OR lattice and wiring audit'
        elif count == 0:
            row['certificate'] = 'bit-local programs cannot match the nonlocal input-pair witness'
        else:
            name = 'free_keys863_one_mad_'+'_'.join(LABELS[v]for v in order)+'.json'
            raw = read(name)
            assert raw['width'] == 11 and tuple(raw['sequence']) == order
            assert not raw['pin_original'] and raw.get('control_program') is None
            assert raw['constant_multipliers_only'] is False
            row['raw_result'] = raw['result']
            if raw['result'] == 'projected_grammar_unsat':
                row['certificate'] = 'fixed-order UNSAT'
            else:
                assert raw['result'] == 'unknown'
                if order[2] == 'mad':
                    row['certificate'] = 'polynomial in coordinate-wise bitwise prefixes, carry-free rectangles'
                elif order == ('^','mad','^'):
                    row['certificate'] = 'carry-free rectangles, allowing a final XOR with x and a constant'
                else:
                    assert order in (('^','mad','&'),('^','mad','|'))
                    row['certificate'] = 'final operand is bijective XOR prefix, constant, or repeated MAD output'
        row['result'] = 'excluded'
        orders.append(row)
    return dict(orders=orders,excluded_orders=len(orders),unresolved_orders=0,
        one_mad_raw_counts=dict(Counter(v['raw_result']for v in orders if 'raw_result'in v)),
        evidence_sha256=evidence,
        xor_wiring_audit=xor,lattice_wiring_audit=lattice,
        bit_local_witness=dict(inputs=[0,1],outputs=[middle(0),middle(1)],
                               output_difference_outside_input_bit=nonlocal_difference&~1),
        minimum_instructions_over_endpoint_keys=4,
        upper_bound='Recorded original four-op program, independently verified over full32.',
        shorter_programs='Zero-MAD programs are bit-local. A one-MAD program of length <=2 '
                         'can be padded with identity MAD into the closed two-MAD orders; '
                         'two-MAD programs of length <=2 are polynomials. Dead code cannot evade these cases.',
        scope='One uint32 input; MAD with arbitrary operands, XOR, AND, OR; arbitrary preloaded constants '
              'and arbitrary input/output XOR keys. No shifts, other opcodes, or earlier live values.',
        new_executable=False,executable_best_cycles=864)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    result = audit()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k]for k in ('excluded_orders','unresolved_orders',
                                         'one_mad_raw_counts','minimum_instructions_over_endpoint_keys')}))


if __name__ == '__main__':
    main()
