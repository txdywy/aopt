"""Necessary certificates for an initial or final AND/OR with two MADs.

Arbitrary endpoint XOR keys are included. Middle AND/OR is covered by the
recorded operand partitions, retaining each raw timeout.
The final-operation argument is given in SUB863_FLOW_ENCODING_20261025.md;
this audit checks its finite target properties and referenced evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

from codex.audit_free_key_orders863 import classify_middle, first_type, mad_rows


ROOT = Path(__file__).resolve().parent


def audit():
    width = 11
    size, mask, high = 1 << width, (1 << width)-1, 1 << (width-1)
    values = [(9*(((33*x+0xe9f8cc1d)&mask)^((16896*x+0xaccf6200)&mask))
               +0xfd7046c5)&mask for x in range(size)]
    assert len(set(values)) == size
    assert all(values[x ^ high] == values[x] ^ high for x in range(size))
    triangular = [(x*x+x)&mask for x in range(size)]
    counts = [sum((t >> j)&1 for t in triangular) for j in range(width)]
    assert counts == [0]+[size//2]*(width-1)
    assert all(triangular[x] == triangular[x ^ mask] for x in range(size))
    difference = values[0] ^ values[mask]
    assert difference & (mask-1)
    # For any B, (32**2+B) differs from B precisely in bit 10.
    square_step = 1 << ((width-1)//2)
    assert square_step*square_step == high

    evidence = {}

    def read(name):
        path = ROOT/name
        evidence[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    rectangles = read('free_keys863_bypass_polynomial.json')['experiments']
    carry = [r for r in rectangles if r['carry_free_rectangles_only'] and not r['xor_input']]
    assert len(carry) == 1 and carry[0]['remaining_pairs'] == 0
    assert carry[0]['initial_key_pairs'] == size*size
    polynomial = read('middle_free_keys863_polynomial.json')['experiment']
    assert polynomial['initial_key_pairs'] == size*size and polynomial['remaining_pairs'] == 0
    orders = {}
    for op in ('and','or'):
        for sequence in ((op,'mad','mad'),('mad',op,'mad'),('mad','mad',op)):
            raw = read('free_keys863_'+'_'.join(sequence)+'.json')
            assert raw['width'] == width and raw['result'] == 'unknown'
            orders[','.join(sequence)] = dict(raw_result=raw['result'],
                refinement=('excluded_by_operand_partitions' if sequence[1] == op else
                            'excluded_by_carry_free_rectangles' if sequence[0] == op else
                            'excluded_by_operand_domination'))
        closed = {'carry_free_prefix_or_polynomial','pure_polynomial'}

        def excluded(name):
            record = read(name)
            assert record['result'] == 'projected_grammar_unsat',name
            assert record['width'] == width and not record['pin_original']
            assert record.get('control_program') is None
            assert record['sequence'] == ['mad','&' if op == 'and' else '|','mad']

        for first,source in itertools.product(('affine','square_const','square_x'),('input','key')):
            if (first,source) == ('affine','key'):
                continue
            excluded(f'free_keys863_{op}_partition_{first}_{source}.json')
            closed.add(f'first_{first}_{source}')
        for tail in ('x_product','first_product','square'):
            excluded(f'free_keys863_{op}_tail_key_{tail}.json')
            closed.add(f'tail_key_{tail}')
        for key,operands in (('scaled0','230'),('scaled1','231'),('scaled4','234'),
                ('addend00','002'),('addend01','012'),('addend11','112'),
                ('addend03','032'),('addend13','132')):
            normalized = '_normalized' if op == 'or' and operands == '231' else ''
            excluded(f'free_keys863_{op}_fixed_key_{operands}{normalized}.json')
            closed.add('fixed_key_'+key)
        classified = Counter()
        for first,bitwise,last in itertools.product(mad_rows(3),
                itertools.combinations_with_replacement(range(4),2),mad_rows(5)):
            label = ('first_affine_input' if first_type(first) == 'affine' and bitwise == (0,1)
                     else classify_middle(first,bitwise,last))
            classified[label] += 1
        assert set(classified) <= closed,set(classified)-closed
        orders[f'mad,{op},mad'].update(wiring_patterns=sum(classified.values()),
                                     classes=dict(classified))
    return dict(width=width,orders=orders,evidence_sha256=evidence,
        target_permutation=True,target_high_bit_equivariance=True,
        square_constant_high_bit_witness=dict(inputs=[0,square_step],difference=high),
        square_x_bit_one_counts=counts,
        square_x_complement_witness=dict(inputs=[0,mask],target_outputs=[values[0],values[mask]],
                                          xor_difference=difference),
        scope='One input, arbitrary endpoint XOR keys/constants; two MADs and one AND or OR.',
        excluded_orders=6,unresolved_orders=0,new_executable=False,
        executable_best_cycles=864)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    result = audit()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('excluded_orders','unresolved_orders','new_executable')}))


if __name__ == '__main__':
    main()
