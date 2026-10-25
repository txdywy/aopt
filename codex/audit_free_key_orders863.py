"""Audit the evidence and wiring coverage for two MADs and one XOR.

This closes only a single-input three-instruction grammar with arbitrary
endpoint XOR keys. It does not prove the full kernel or every ISA optimal.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def mad_rows(pool_size):
    return [(a,b,c) for a,b in itertools.combinations_with_replacement(range(pool_size),2)
            for c in range(pool_size)]


def first_type(row):
    a,b,c = row
    return ('square_x' if c == 0 else 'square_const') if a == b == 0 else 'affine'


def classify_middle(first, bitwise, last):
    a,b = bitwise
    if 1 not in bitwise or a == b:
        return 'carry_free_prefix_or_polynomial'
    source = 'input' if (a,b) == (0,1) else 'key'
    initial = first_type(first)
    if initial != 'affine':
        return f'first_{initial}_{source}'
    a,b,c = last
    if 2 not in last:
        return 'pure_polynomial'
    if a == 2 or b == 2:
        other = b if a == 2 else a
        if other < 3:
            return f'tail_{source}_'+{0:'x_product',1:'first_product',2:'square'}[other]
        # q*D+q is q*(D+1)+0; either named constant can supply the offset.
        suffix = str(c) if c < 2 else '4'
        return f'fixed_{source}_scaled{suffix}'
    assert c == 2
    if source == 'input':
        return 'tail_input_addend'
    if a >= 3:
        return 'fixed_key_scaled4'  # q plus an arbitrary constant
    if b >= 3:
        b = 3
    return f'fixed_key_addend{a}{b}'


def classify_parallel(first, second, last):
    a,b = last
    if a == b:
        return 'pure_polynomial'
    if 2 not in last:
        return 'polynomial_or_input_bypass'
    other = b if a == 2 else a
    if other == 0:
        return 'polynomial_xor_input'
    if other >= 3:
        return 'pure_polynomial'
    assert other == 1
    a,b,c = second
    if a >= 2:
        if c >= 2:
            return 'pure_polynomial'
        # Two constant multiplicands produce an arbitrary constant addend.
        form = 'scale_input' if c == 0 else 'scale_first'
    elif b >= 2:
        form = 'scale_input' if a == 0 else 'scale_first'
    else:
        form = {(0,0):'x_square',(0,1):'x_first',(1,1):'first_square'}[a,b]
    first = first_type(first)
    if first == 'affine' and form.startswith('scale_'):
        return 'parallel_affine_certificate'
    return f'parallel_{first}_{form}'


def audit():
    evidence = {}

    def check(name, expected):
        path = ROOT/name
        data = json.loads(path.read_text())
        assert data['result'] == expected,(name,data['result'])
        assert data['width'] == 11,(name,data['width'])
        if 'sequence' in data:
            assert data['sequence'].count('mad') == 2 and data['sequence'].count('^') == 1
            assert data['constant_multipliers_only'] is False
            assert data['two_arbitrary_constants_per_instruction'] is True
            assert data['pin_original'] is False
            assert data.get('control_program') is None
        evidence[name] = dict(result=data['result'],sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return data

    closed = {'carry_free_prefix_or_polynomial','pure_polynomial',
              'polynomial_or_input_bypass','polynomial_xor_input','parallel_affine_certificate'}
    for first,source in itertools.product(('square_const','square_x'),('input','key')):
        check(f'free_keys863_partition_{first}_{source}.json','projected_grammar_unsat')
        closed.add(f'first_{first}_{source}')
    for source,tail in itertools.product(('input','key'),('x_product','first_product','square')):
        check(f'free_keys863_tail_partition_{source}_{tail}.json','projected_grammar_unsat')
        closed.add(f'tail_{source}_{tail}')
    check('free_keys863_tail_partition_input_addend.json','projected_grammar_unsat')
    closed.add('tail_input_addend')
    for source,c in itertools.product(('input','key'),(0,1,4)):
        name = f'free_keys863_fixed_tail_{source}_scaled{c}.json'
        if source == 'key' and c == 4:
            check(name,'unknown')
            check('middle_free_keys863_nested_normalized.json','projected_family_unsat')
        else:
            check(name,'projected_grammar_unsat')
        closed.add(f'fixed_{source}_scaled{c}')
    for a,b in ((0,0),(0,1),(1,1),(0,3),(1,3)):
        check(f'free_keys863_fixed_tail_key_addend{a}{b}.json','projected_grammar_unsat')
        closed.add(f'fixed_key_addend{a}{b}')
    for first,second in itertools.product(('affine','square_const','square_x'),
            ('x_square','x_first','first_square','scale_input','scale_first')):
        if first == 'affine' and second.startswith('scale_'):
            continue
        check(f'free_keys863_parallel_partition_{first}_{second}.json','projected_grammar_unsat')
        closed.add(f'parallel_{first}_{second}')
    check('middle_free_keys863_parallel.json','projected_family_unsat')
    rectangle_path = ROOT/'free_keys863_bypass_polynomial.json'
    rectangles = json.loads(rectangle_path.read_text())
    assert len(rectangles['experiments']) == 3
    assert {(r['xor_input'],r['carry_free_rectangles_only']) for r in rectangles['experiments']} == {
        (True,False),(False,True),(True,True)}
    assert all(r['remaining_pairs'] == 0 and r['initial_key_pairs'] == 4194304
               and r['width'] == 11 and not r['affine_control']
               for r in rectangles['experiments'])
    evidence[rectangle_path.name] = dict(result='all three screens exclude every key pair',
        sha256=hashlib.sha256(rectangle_path.read_bytes()).hexdigest())
    counts = {}
    # Each instruction has two named arbitrary constants, sufficient for any
    # nonconstant binary op/MAD. All-constant instructions can be folded.
    for order in ('XOR,MAD,MAD','MAD,XOR,MAD','MAD,MAD,XOR'):
        classified = Counter()
        if order == 'XOR,MAD,MAD':
            for _ in itertools.product(itertools.combinations_with_replacement(range(3),2),
                                       mad_rows(4),mad_rows(5)):
                classified['carry_free_prefix_or_polynomial'] += 1
        elif order == 'MAD,XOR,MAD':
            for first,bitwise,last in itertools.product(mad_rows(3),
                    itertools.combinations_with_replacement(range(4),2),mad_rows(5)):
                classified[classify_middle(first,bitwise,last)] += 1
        else:
            for first,second,last in itertools.product(mad_rows(3),mad_rows(4),
                    itertools.combinations_with_replacement(range(5),2)):
                classified[classify_parallel(first,second,last)] += 1
        assert set(classified) <= closed,set(classified)-closed
        counts[order] = dict(wiring_patterns=sum(classified.values()),classes=dict(classified),
                            result='closed_by_recorded_necessary_conditions')
    return dict(executable_best_cycles=864,new_executable=False,orders=counts,evidence=evidence,
        total_wiring_patterns=sum(v['wiring_patterns'] for v in counts.values()),
        scope='One input x, arbitrary uint32 constants and endpoint XOR keys; '
              'two arbitrary-operand MADs and one XOR. Wiring counts include '
              'constant-name symmetries and dead-code forms, not distinct functions.',
        exclusions='Not other opcodes, extra live inputs, four-op graphs, or the whole kernel.',
        unknown_handling='Raw UNKNOWN files are preserved. The remaining fixed scaled-key '
                         'case is the independently normalized nested affine family.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    result = audit()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(wiring_patterns=result['total_wiring_patterns'],
                         orders={k:v['result'] for k,v in result['orders'].items()})))


if __name__ == '__main__':
    main()
