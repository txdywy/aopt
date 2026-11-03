"""Complete the 12 remaining orders in the middle-hash SHR grammar.

Finite bit-rectangle certificates, full32 polynomial relaxations, and the
associated algebraic case splits are documented in
SUB863_SHIFT_COMPLETE_20261024.md. This is not a whole-kernel lower bound.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import random
import time

import z3

from codex.screen_shift_middle863 import MASK, target
from codex.screen_xor_mad_polynomial import consistent


def shift(x, count, code='>>'):
    if not 0 <= x <= MASK or not 0 <= count <= MASK or code not in ('<<','>>'):
        raise ValueError('uint32 operands and a shift opcode required')
    return 0 if count >= 32 else ((x << count)&MASK) if code == '<<' else x >> count


def rectangle(a, b, base, width, fn=target):
    if not 0 <= a < b < 32 or not 1 <= width <= 32 or not 0 <= base <= MASK:
        raise ValueError('invalid bit rectangle')
    if base & ((1 << a)|(1 << b)):
        raise ValueError('base must clear both varying bits')
    xs = [base+d for d in (0,1 << a,1 << b,(1 << a)+(1 << b))]
    ys = [fn(x)&MASK for x in xs]
    delta = (ys[3]-ys[2]-ys[1]+ys[0]) % (1 << width)
    return dict(bits=[a,b],base=base,width=width,inputs=xs,values=ys,mixed_difference=delta)


def literal_rectangle(count, high=False, xor_input=False):
    if not 0 <= count <= 32:
        raise ValueError('representative counts are0..32')
    if 1 <= count <= 6:
        a,b,w,base = 2*count+2,2*count+11,2*count+13,0
    elif count == 7:
        a,b,w,base = 5,6,11,12
    elif count == 8:
        a,b,w,base = 4,7,11,290
    else:
        a,b,w,base = 3,8,11,0
    if xor_input and count == 3:
        base = 1
    if high:
        base |= 1 << 31
    row = rectangle(a,b,base,w,fn=(lambda x:target(x)^x) if xor_input else target)
    hm,vm = ((1 << bit)|shift(1 << bit,count) for bit in (a,b))
    hv,vv = ((mask & -mask).bit_length()-1 for mask in (hm,vm))
    assert hm & vm == 0 and hv+vv >= w and row['mixed_difference'] != 0
    row.update(shift=count,horizontal_mask=hm,vertical_mask=vm,
               valuations=[hv,vv],high_domain=high,target_xor_input=xor_input)
    return row


def bitwise_certificates():
    literal = [literal_rectangle(s) for s in range(33)]
    zero_domain = rectangle(3,8,32,11)
    xor_blocks = [rectangle(3,8,x,11) for x in (0,32)]
    assert not {x >> 5 for x in xor_blocks[0]['inputs']} & {
        x >> 5 for x in xor_blocks[1]['inputs']}
    and_high = []
    for bit in range(5,32):
        row = rectangle(4,7,290,11) if bit == 8 else rectangle(3,8,1 << bit,11)
        assert row['mixed_difference'] and all(x&(1 << bit) for x in row['inputs'])
        row['forbidden_mask_bit'] = bit
        and_high.append(row)
    and_low = rectangle(5,14,0,19)
    assert and_low['mixed_difference'] and all(x%32 == 0 for x in and_low['inputs'])
    return dict(literal_shifts=literal, input_count_zero_domain=zero_domain,
        xor_count_disjoint_blocks=xor_blocks, or_count_zero_domain=zero_domain,
        and_count_high_bits=and_high, and_count_low_mask=and_low,
        orders=[list(seq) for op in ('&','|','^') for seq in
                ((op,'>>','mad'),('>>',op,'mad'))], excluded=True)


def xor_key_query(row, fn=target, timeout_ms=5000, fixed_key=None):
    if timeout_ms <= 0:
        raise ValueError('positive timeout required')
    width = row['width']
    key = z3.BitVec('rectangle_xor_key',width)
    values = [z3.BitVecVal(fn(x),width)^key for x in row['inputs']]
    solver = z3.SolverFor('QF_BV')
    solver.set(timeout=timeout_ms)
    if fixed_key is not None:
        solver.add(key == fixed_key)
    solver.add(values[3]-values[2]-values[1]+values[0] == 0)
    status = solver.check()
    result = dict(result=str(status),width=width,inputs=row['inputs'],timeout_ms=timeout_ms)
    if status == z3.sat:
        result['key'] = solver.model().eval(key,model_completion=True).as_long()
    elif status == z3.unknown:
        result['reason'] = solver.reason_unknown()
    return result


def final_xor_certificates():
    rows = []
    for s in range(33):
        row = literal_rectangle(s)
        ys = [target(x)^shift(x,s) for x in row['inputs']]
        delta = (ys[3]-ys[2]-ys[1]+ys[0]) % (1 << row['width'])
        assert delta
        rows.append(dict(shift=s,polynomial_xor_q_rectangle=row,
            f_xor_q_difference=delta,polynomial_xor_x_rectangle=literal_rectangle(s,xor_input=True),
            constant_xor_key=xor_key_query(row)))
    variable = rectangle(3,8,32,11)
    xy = [target(x)^x for x in variable['inputs']]
    assert (xy[3]-xy[2]-xy[1]+xy[0]) & 2047 == 1024
    variable_key = xor_key_query(variable)
    return dict(order=['>>','mad','^'],literals=rows,
        variable_count_rectangle=variable,variable_count_xor_key=variable_key,
        excluded=all(r['constant_xor_key']['result']=='unsat' for r in rows)
                 and variable_key['result']=='unsat')


def quadratic_terms(x,q,t):
    return [1,x,q,t,x*x,x*q,x*t,q*q,q*t,t*t]


def pair_samples():
    rng = random.Random(863)
    return list(dict.fromkeys([*range(16),32,40,256,264,288,296,512,513,1024,1032]
        +[1 << i for i in range(32)]+[rng.getrandbits(32) for _ in range(16)]))


def literal_pair_scan():
    xs = pair_samples()
    rhs = [target(x) for x in xs]
    results = []
    start = time.monotonic()
    for ops in (('>>','>>'),('<<','>>'),('>>','<<')):
        survivors,controls,tested = [],0,0
        for s in range(33):
            qs = [shift(x,s,ops[0]) for x in xs]
            for k,source in itertools.product(range(33),('x','q')):
                ts = [shift(x if source=='x' else q,k,ops[1]) for x,q in zip(xs,qs)]
                matrix = [quadratic_terms(x,q,t) for x,q,t in zip(xs,qs,ts)]
                if consistent(matrix,rhs,32):
                    survivors.append(dict(first_count=s,second_count=k,source=source))
                # An actual final MAD, rather than an arbitrary fitting RHS.
                control = [(x*q+t)&MASK for x,q,t in zip(xs,qs,ts)]
                controls += consistent(matrix,control,32)
                tested += 1
        results.append(dict(ops=ops,tested=tested,survivors=survivors,
                            positive_controls=controls))
    return dict(arithmetic_width=32,monomials=10,samples=xs,
        rows=results,seconds=round(time.monotonic()-start,3),
        excluded=all(not r['survivors'] for r in results),
        positive_controls=sum(r['positive_controls'] for r in results),
        scope='Literal-count prefixes. Constant sources are absorbed by the independent '
              'polynomial coefficients; this includes bypassed first/second results.')


def variable_pair_certificates():
    # On x>=32 a shift with count x is zero. The only remaining nonzero
    # variable word is a literal shift of x, or x itself when count q=0.
    high = [literal_rectangle(s,high=True) for s in range(33)]
    polynomial = rectangle(3,8,32,11)
    right_counts,left_counts = [],[]
    for s in range(33):
        if s == 0:
            row = polynomial.copy()
        elif s < 9:
            row = literal_rectangle(s,high=True)
        else:
            row = rectangle(3,8,0,11)
        qs = [shift(x,s) for x in row['inputs']]
        assert all(q >= 32 for q in qs) or all(q == 0 for q in qs)
        right_counts.append(dict(first_count=s,rectangle=row,counts=qs))
        if s == 0:
            row = polynomial.copy()
        elif s == 32:
            row = rectangle(3,8,0,11)
        else:
            bit = max(0,5-s)
            row = rectangle(3,8,4,11) if bit == 2 else rectangle(2,11,1 << bit,13)
        qs = [shift(x,s,'<<') for x in row['inputs']]
        assert all(q >= 32 for q in qs) or all(q == 0 for q in qs)
        assert row['mixed_difference']
        left_counts.append(dict(first_count=s,rectangle=row,counts=qs))
    return dict(literal_shr_on_high_domain=high,polynomial_high_domain=polynomial,
        second_count_first_shr=right_counts,second_count_first_shl=left_counts,excluded=True)


def final_shift_certificates():
    high,odd = [1 << 31,(1 << 31)+1],[1,3]
    assert target(high[0]) > high[0]
    assert all(target(x)&(1 << 31) for x in high)
    assert target(high[0]) != target(high[1])
    assert target(1) > 1 and target(1) != target(3)
    assert all(target(x)&1 for x in odd)
    count_witnesses = []
    for s in range(32):
        x = 1 if s == 0 else (1 << s)|1
        assert shift(x,s) > 0 and target(x)&1
        count_witnesses.append(dict(first_count=s,odd_input=x,first_result=shift(x,s)))
    t = z3.BitVec('self_shift_word',32)
    right,left = z3.SolverFor('QF_BV'),z3.SolverFor('QF_BV')
    right.add(z3.LShR(t,t) != 0)
    left.add(((t << t)&1) != 0)
    self_results = dict(right_zero=str(right.check()),left_even=str(left.check()))
    return dict(orders=[['>>','mad','>>'],['>>','mad','<<']],
        high_inputs=high,high_outputs=list(map(target,high)),
        odd_inputs=odd,odd_outputs=list(map(target,odd)),
        positive_first_count_odd_witnesses=count_witnesses,self_shift=self_results,
        high_domain_polynomial_rectangle=rectangle(3,8,32,11),
        excluded=all(v=='unsat' for v in self_results.values()))


def closure(report):
    old = json.loads(Path(__file__).with_name('middle_shift_closure863_results.json').read_text())
    assert old['coverage']['closed'] == 36
    new = {}
    if report['bitwise']['excluded']:
        new.update({tuple(s):'orthogonal bit rectangles and variable-count partition'
                    for s in report['bitwise']['orders']})
    if report['final_xor']['excluded']:
        new[('>>','mad','^')] = 'polynomial rectangle after undoing the final XOR'
    if report['literal_pairs']['excluded'] and report['variable_pairs']['excluded']:
        new.update({(*r['ops'],'mad'):'full32 polynomial relaxation and variable-count partition'
                    for r in report['literal_pairs']['rows']})
    if report['final_shifts']['excluded']:
        new.update({tuple(s):'output highest/lowest-bit conditions and zero-domain certificate'
                    for s in report['final_shifts']['orders']})
    rows = []
    for old_row in old['coverage']['orders']:
        row = dict(old_row)
        seq = tuple(row['sequence'])
        row['previously_excluded'] = row['excluded']
        if seq in new:
            row['excluded'] = True
            row['evidence'] = [*row['evidence'],new[seq]]
        rows.append(row)
    return dict(orders=rows,closed=sum(r['excluded'] for r in rows),
        unresolved=sum(not r['excluded'] for r in rows),
        scope=old['coverage']['scope'],new_executable=False,best_verified_cycles=864)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    report = dict(z3=z3.get_version_string(),bitwise=bitwise_certificates(),
        final_xor=final_xor_certificates(),literal_pairs=literal_pair_scan(),
        variable_pairs=variable_pair_certificates(),final_shifts=final_shift_certificates())
    report['coverage'] = closure(report)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(closed=report['coverage']['closed'],
        unresolved=report['coverage']['unresolved'],
        pair_positive_controls=report['literal_pairs']['positive_controls'],new_executable=False)))


if __name__ == '__main__':
    main()
