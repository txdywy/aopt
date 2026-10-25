"""Search F(x XOR I) XOR O with *both* endpoint keys unknown.

Only MAD/XOR/AND/OR are admitted, so low-width UNSAT is a sound necessary
screen for a full32 program with the same order. Low-width SAT is never a
kernel candidate. MAD has two arbitrary multiplicands, not a fixed factor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import z3

from codex.search_cross_encoding863 import middle
from codex.synthesize_hash_cegis import choose


OPS = ('mad', '^', '&', '|')


def execute(x, rows, width):
    mask = (1 << width)-1
    values = [x & mask]
    for row in rows:
        pool = values+row['constants']
        a, b, c = [pool[j] for j in row['operands']]
        values.append((a*b+c if row['op'] == 'mad' else
                       a ^ b if row['op'] == '^' else
                       a & b if row['op'] == '&' else a | b) & mask)
    return values[-1]


def expression(x, rows, width):
    values = [x]
    for row in rows:
        pool = values+[z3.BitVecVal(k, width) for k in row['constants']]
        a, b, c = [pool[j] for j in row['operands']]
        values.append(a*b+c if row['op'] == 'mad' else a ^ b if row['op'] == '^'
                      else a & b if row['op'] == '&' else a | b)
    return values[-1]


def search(order, width=11, seconds=15, check_ms=12000, *, pin_original=False,
           first_form=None, middle_source=None, tail_form=None, last_operands=None,
           second_form=None, control_program=None):
    if not 1 <= len(order) <= 4 or any(op not in OPS for op in order):
        raise ValueError('one to four MAD/XOR/AND/OR instructions required')
    if width not in (*range(1, 13), 32) or seconds <= 0 or check_ms <= 0:
        raise ValueError('width 1..12 or 32 and positive time limits required')
    if pin_original and tuple(order) != ('mad','mad','^','mad'):
        raise ValueError('original witness has order MAD,MAD,XOR,MAD')
    if first_form not in (None,'affine','square_const','square_x') or middle_source not in (None,'input','key'):
        raise ValueError('invalid fixed first/middle structure')
    if tail_form not in (None,'scaled','x_product','first_product','square','addend'):
        raise ValueError('invalid final MAD structure')
    if last_operands is not None and (len(last_operands) != 3 or any(v not in range(5) for v in last_operands)):
        raise ValueError('last_operands requires three indices in 0..4')
    middle_orders = tuple(('mad',op,'mad') for op in ('^','&','|'))
    final_orders = tuple(('mad','mad',op) for op in ('^','&','|'))
    if first_form and tuple(order) not in middle_orders+final_orders:
        raise ValueError('first MAD partitions require two MADs and one bitwise operation')
    if (middle_source or tail_form or last_operands) and tuple(order) not in middle_orders:
        raise ValueError('middle partitions require MAD,bitwise,MAD')
    if second_form not in (None,'x_square','x_first','first_square','scale_input','scale_first'):
        raise ValueError('invalid second MAD structure')
    if second_form and tuple(order) not in final_orders:
        raise ValueError('second MAD partitions require MAD,MAD,bitwise')
    mask = (1 << width)-1
    ik, ok = z3.BitVecs('equivalent_input_key equivalent_output_key', width)
    solver = z3.SolverFor('QF_BV')
    # F(x XOR H)=F(x) XOR H; absorb the highest input-key bit into O.
    solver.add(z3.Extract(width-1, width-1, ik) == 0)
    if control_program is not None:
        # Positive controls validate structure selectors against an independent
        # concrete program, without assuming its XOR-key symmetries.
        solver.add(ik == 0,ok == 0)

    def reference(x):
        if control_program is None:
            return middle(x)
        return execute(x,control_program,width) if isinstance(x,int) else expression(x,control_program,width)
    spec = []
    for i, op in enumerate(order):
        a,b,c = z3.BitVecs(f'eq_a{i} eq_b{i} eq_c{i}', 4)
        k0,k1 = z3.BitVecs(f'eq_k{i}_0 eq_k{i}_1', width)
        solver.add(*(z3.ULT(v, i+3) for v in (a,b,c)), z3.ULE(a,b))
        if op != 'mad':
            solver.add(c == 0)
        spec.append((op,a,b,c,k0,k1))
    if first_form:
        indices = {'affine':(0,1,2),'square_const':(0,0,1),'square_x':(0,0,0)}[first_form]
        solver.add(*(v == n for v,n in zip(spec[0][1:4],indices)))
    if middle_source:
        indices = (0,1) if middle_source == 'input' else (1,2)
        solver.add(*(v == n for v,n in zip(spec[1][1:3],indices)))
    if tail_form:
        a,b,c = spec[2][1:4]
        if tail_form == 'addend':
            solver.add(c == 2,a != 2,b != 2)
        else:
            left,right = {'scaled':(2,3),'x_product':(0,2),
                          'first_product':(1,2),'square':(2,2)}[tail_form]
            solver.add(a == left,b == right)
    if last_operands is not None:
        solver.add(*(v == n for v,n in zip(spec[2][1:4],last_operands)))
    sole_affine_input = (control_program is None and first_form == 'affine'
        and middle_source == 'key' and last_operands is not None and 0 not in last_operands)
    if sole_affine_input:
        # The whole program is a function of t=A*x+B. Target bijectivity
        # requires A odd. Complementing x can be absorbed into A,B, so the
        # arbitrary endpoint input key can also have its lowest bit cleared.
        solver.add(spec[0][4] & 1 == 1,ik & 1 == 0)
    if second_form:
        left,right = {'x_square':(0,0),'x_first':(0,1),'first_square':(1,1),
                      'scale_input':(0,2),'scale_first':(1,2)}[second_form]
        solver.add(spec[1][1] == left,spec[1][2] == right,
                   spec[2][1] == 1,spec[2][2] == 2)
    for i in range(len(order)-1):
        uses = [v == i+1 for row in spec[i+1:] for v in row[1:4 if row[0] == 'mad' else 3]]
        solver.add(z3.Or(*uses))
    if pin_original:
        solver.add(ik == 0, ok == 0)
        original = [((0,1,2),(33,0xe9f8cc1d)),
                    ((0,2,3),(16896,0xaccf6200)),
                    ((1,2,0),(0,0)), ((3,4,5),(9,0xfd7046c5))]
        for row, (indices, constants) in zip(spec, original):
            solver.add(*(v == n for v,n in zip(row[1:4], indices)),
                       *(v == (n & mask) for v,n in zip(row[4:], constants)))

    def candidate(x):
        values = [x]
        for op,a,b,c,k0,k1 in spec:
            pool = values+[k0,k1]
            av,bv,cv = choose(a,pool),choose(b,pool),choose(c,pool)
            values.append(av*bv+cv if op == 'mad' else av ^ bv if op == '^'
                          else av & bv if op == '&' else av | bv)
        return z3.simplify(values[-1])

    def constrain(value):
        x = z3.BitVecVal(value, width)
        solver.add(candidate(x) == (reference(x ^ ik) ^ ok))

    rng = random.Random(882)
    points = list(dict.fromkeys(x & mask for x in
        [*range(16),256,264,288,296,511,512,1023,1024,mask]
        + [rng.getrandbits(width) for _ in range(16)]))
    for p in points:
        constrain(p)
    checks = (list(range(1 << width)) if width < 32 else
              list(dict.fromkeys(points+[1 << i for i in range(32)]
                                 +[rng.getrandbits(32) for _ in range(256)])))
    start = time.monotonic()
    deadline = start+seconds
    report = dict(sequence=list(order), width=width, seconds_limit=seconds,
        check_ms=check_ms, endpoint_keys='arbitrary; input highest bit normalized to zero',
        constant_multipliers_only=False, two_arbitrary_constants_per_instruction=True,
        initial_samples=points[:], pin_original=pin_original, events=[],
        first_form=first_form, middle_source=middle_source, tail_form=tail_form,
        last_operands=last_operands,
        sole_affine_input_normalization=sole_affine_input,
        second_form=second_form,
        control_program=control_program,
        result='deadline', new_executable=False)
    while time.monotonic() < deadline:
        solver.set(timeout=min(check_ms,max(1,int(1000*(deadline-time.monotonic())))))
        status = solver.check()
        event = dict(samples=len(points), synthesis=str(status))
        report['events'].append(event)
        if status != z3.sat:
            report['result'] = 'projected_grammar_unsat' if status == z3.unsat else 'unknown'
            if status == z3.unknown:
                report['reason'] = solver.reason_unknown()
            break
        model = solver.model()
        iv,ov = [model.eval(k,model_completion=True).as_long() for k in (ik,ok)]
        rows = [dict(op=row[0], operands=[model.eval(v,model_completion=True).as_long() for v in row[1:4]],
                     constants=[model.eval(v,model_completion=True).as_long() for v in row[4:]]) for row in spec]
        report.update(program=rows,input_key=iv,output_key=ov)
        bad = [x for x in checks if execute(x,rows,width) != ((reference(x ^ iv)^ov)&mask)][:8]
        if bad:
            event['concrete_counterexamples'] = bad
            for x in bad:
                assert x not in points
                points.append(x)
                constrain(x)
            continue
        if width < 32:
            report.update(result='equivalent_on_reduced_domain',verified_inputs=len(checks))
            break
        x = z3.BitVec('equivalent_full32_verify',32)
        verify = z3.SolverFor('QF_BV')
        verify.set(timeout=min(check_ms,max(1,int(1000*(deadline-time.monotonic())))))
        verify.add(expression(x,rows,width) != (reference(x ^ iv)^ov))
        checked = verify.check()
        event['full32_verification'] = str(checked)
        if checked == z3.unsat:
            report['result'] = 'proved_equivalent_full32'
            break
        if checked == z3.unknown:
            report.update(result='verification_unknown',reason=verify.reason_unknown())
            break
        p = verify.model().eval(x,model_completion=True).as_long()
        event['symbolic_counterexample'] = p
        assert p not in points
        points.append(p)
        constrain(p)
    report['seconds'] = round(time.monotonic()-start,3)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sequence', required=True)
    ap.add_argument('--width', type=int, default=11)
    ap.add_argument('--seconds', type=float, default=15)
    ap.add_argument('--check-ms', type=int, default=12000)
    ap.add_argument('--pin-original', action='store_true')
    ap.add_argument('--first-form',choices=('affine','square_const','square_x'))
    ap.add_argument('--middle-source',choices=('input','key'))
    ap.add_argument('--tail-form',choices=('scaled','x_product','first_product','square','addend'))
    ap.add_argument('--last-operands',type=lambda x:tuple(map(int,x.split(','))))
    ap.add_argument('--second-form',choices=('x_square','x_first','first_square','scale_input','scale_first'))
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    report = search(args.sequence.split(','),args.width,args.seconds,args.check_ms,
                    pin_original=args.pin_original,first_form=args.first_form,
                    middle_source=args.middle_source,tail_form=args.tail_form,
                    last_operands=args.last_operands,second_form=args.second_form)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report.get(k) for k in ('sequence','width','result','seconds','input_key','output_key')}))


if __name__ == '__main__':
    main()
