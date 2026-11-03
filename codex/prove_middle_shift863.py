"""Certificates for two remaining middle-hash SHR families.

The accompanying SUB863_SHIFT_CLOSURE_20261024.md contains the algebraic
proofs. This module records their finite witnesses, full32 SMT queries and
operand coverage. It does not generate an executable kernel or a score.
"""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import z3

from codex.screen_shift_middle863 import MASK, full_shift, target
from codex.search_cross_encoding863 import middle
from codex.synthesize_hash_cegis import choose

ROOT = Path(__file__).parent
NONSHIFT = ('+', '-', '^', '&', '|', '*', 'mad')


def shapes():
    """Final pool is x,t,q,k0,k1; multiplication is commutative."""
    return [(*pair, c) for pair in itertools.combinations_with_replacement(range(5), 2)
            for c in range(5)]


def classify_odd(role):
    a, b, c = role
    if 2 not in role:
        return 'integer_polynomial'
    if not any(i in (0, 1) for i in role):
        return 'q_only_cardinality'
    if 2 in (a, b):
        other = b if a == 2 else a
        if other in (0, 1):
            return 'q_times_odd_diagonal'
        if other == 2:
            return 'q_squared_plus_odd_diagonal'
        return 'scaled_q_plus_odd_diagonal'
    return 'polynomial_plus_q'


def classify_even(role):
    a, b, c = role
    if 2 not in role:
        return 'integer_polynomial'
    if 0 not in role:
        return 'periodic_without_x'
    if 2 in (a, b):
        other = b if a == 2 else a
        return {0: 'q_times_x', 1: 'q_times_t_plus_x',
                2: 'q_squared_plus_x', 3: 'scaled_q_plus_x',
                4: 'scaled_q_plus_x'}[other]
    return {(0, 0): 'x_squared_plus_q', (0, 1): 'x_times_t_plus_q',
            (0, 3): 'scaled_x_plus_q', (0, 4): 'scaled_x_plus_q'}[a, b]


def difference(x, amount, fn=target):
    return (fn((x+amount) & MASK)-fn(x)) & MASK


def finite_witnesses():
    rng = random.Random(864)
    inputs = list(range(1024)) + [rng.getrandbits(32) for _ in range(1024)]
    periods = []
    for exponent in range(1, 31):
        step = 1 << exponent
        base = difference(0, step)
        other = next(x for x in inputs if difference(x, step) != base)
        periods.append(dict(exponent=exponent, step=step, inputs=[0, other],
            differences=[base, difference(other, step)]))
    ranges = []
    # Only these cases remain after r>s and polynomial projection for s<=21.
    for s in range(22, 31):
        for r in range(s+1, 32):
            period, bound = 1 << (32-r), 1 << (32-s)
            x = next(x for x in inputs if difference(x, period)//period > bound)
            delta = difference(x, period)
            assert delta % period == 0
            ranges.append(dict(shift=s, valuation=r, period=period, input=x,
                difference=delta, quotient=delta//period,
                largest_possible_q_plus_one=bound))
    rectangle = [0, 8, 256, 264]
    values = [target(x) & 2047 for x in rectangle]
    assert (values[3]-values[2]-values[1]+values[0]) & 2047 == 1024
    assert target(0) % 4 == 2
    assert all((target(x)-x) % 4 == 2 for x in range(4))
    return dict(polynomial_rectangle=rectangle, values_mod2048=values,
        mixed_difference=1024, nonconstant_power_of_two_differences=periods,
        large_shift_quotient_witnesses=ranges, f_zero=target(0),
        f_minus_x_mod4=2,
        odd_shapes=dict(Counter(map(classify_odd, shapes()))),
        even_shapes=dict(Counter(map(classify_even, shapes()))),
        roles=[dict(operands=r, odd_reason=classify_odd(r),
                    even_reason=classify_even(r)) for r in shapes()])


def violating_rectangles(fn=target):
    """Every retained rectangle defeats *any* integer polynomial mod 2**w."""
    rng = random.Random(864)
    rows = []
    for w in range(11, 21):
        for i in (1, 3, 5):
            h, k = 1 << i, 1 << (w-i)
            for base in [*range(16)] + [rng.getrandbits(32) for _ in range(8)]:
                xs = [(base+d) & MASK for d in (0, h, k, h+k)]
                ys = [fn(x) for x in xs]
                delta = (ys[3]-ys[2]-ys[1]+ys[0]) % (1 << w)
                if delta:
                    rows.append(dict(width=w, steps=[h,k], inputs=xs,
                        values=ys, mixed_difference=delta))
    return rows


def variable_count_query(timeout_ms=10000, fn=target, nonzero=True,
                         fixed_affine=None):
    """Necessary exception-hitting condition for q=source>>(A*x+B).

On a whole rectangle with A*x+B >=32 the shift vanishes, making the
continuation polynomial. Thus each nonpolynomial rectangle must contain
at least one count below32. This relaxation deliberately ignores the
shift source and all final-MAD coefficients.
    """
    if timeout_ms <= 0:
        raise ValueError('positive timeout required')
    rows = violating_rectangles(fn)
    a,b = z3.BitVecs('exception_a exception_b',32)
    solver = z3.SolverFor('QF_BV')
    solver.set(timeout=timeout_ms)
    if nonzero:
        solver.add(a != 0)
    if fixed_affine is not None:
        solver.add(a == fixed_affine[0], b == fixed_affine[1])
    for row in rows:
        solver.add(z3.Or(*(z3.ULT(a*x+b,32) for x in row['inputs'])))
    start = time.monotonic()
    result = solver.check()
    report = dict(result=str(result), seconds=round(time.monotonic()-start,3),
        timeout_ms=timeout_ms, nonzero_multiplier=nonzero, rectangles=rows)
    if result == z3.sat:
        model = solver.model()
        report['relaxation_model'] = [model.eval(v,model_completion=True).as_long()
                                      for v in (a,b)]
    elif result == z3.unknown:
        report['reason'] = solver.reason_unknown()
    return report


def top_pair_query():
    x = z3.BitVec('middle_top_pair_x',32)
    solver = z3.SolverFor('QF_BV')
    solver.set(timeout=10000)
    solver.add(middle(x ^ (1 << 31)) != (middle(x) ^ (1 << 31)))
    result = solver.check()
    return dict(result=str(result), claim='F(x XOR 2**31)=F(x) XOR 2**31')


def final_mask_certificate():
    """Close SHR,MAD,AND/OR by the necessary output bit containment."""
    rng = random.Random(864)
    inputs = [*range(1024)]+[rng.getrandbits(32) for _ in range(2048)]
    rows = []
    for s in range(1,32):
        x = next(x for x in inputs if (x >> s) & ~target(x))
        rows.append(dict(shift=s,input=x,shifted=x >> s,target=target(x),
                         forbidden_or_bits=(x >> s) & ~target(x)))
    assert target(0) != 0 and target(MASK) != MASK
    assert target(1 << 31) & (1 << 31)
    roles = []
    # Final operands: x,q,t,k0,k1.
    for pair in itertools.combinations_with_replacement(range(5),2):
        if max(pair) >= 3:
            reason = 'permutation forces constant mask to identity; previous SHR/MAD/MAD exclusion'
        elif pair[0] == pair[1]:
            reason = 'idempotence; previous SHR/MAD/MAD exclusion'
        elif 0 in pair:
            reason = 'F(0)!=0 for AND; F(MASK)!=MASK for OR'
        else:
            assert pair == (1,2)
            reason = 'literal-q range/OR-bit witnesses; variable-count zero-domain polynomial certificate'
        roles.append(dict(operands=pair,reason=reason))
    return dict(orders=[['>>','mad','&'],['>>','mad','|']],roles=roles,
        or_literal_witnesses=rows,f_zero=target(0),f_mask=target(MASK),
        and_high_output_input=1 << 31, and_high_output=target(1 << 31),
        variable_count_rectangle=[32,40,288,296],excluded=True)


def binary(op, a, b, c=0):
    if op == '+': return a+b
    if op == '-': return a-b
    if op == '*': return a*b
    if op == '^': return a^b
    if op == '&': return a&b
    if op == '|': return a|b
    if op == 'mad': return a*b+c
    raise ValueError('only projection-compatible operations are supported')


def prefix_orders():
    return sorted(set(pair for op in NONSHIFT for pair in (('mad',op),(op,'mad'))))


def prefix_query(order, width=11, timeout_ms=10000, fn=target):
    """Search any selected output of a two-op nonshift prefix at low width.

The final SHR is NOT narrowed or emulated here. The high-output lemma in
the accompanying report forces its count to zero on a full low-domain
cover, so its numerator alone must implement this projected function.
    """
    if tuple(order) not in prefix_orders() or not 2 <= width <= 12 or timeout_ms <= 0:
        raise ValueError('invalid prefix order, width or timeout')
    solver = z3.SolverFor('QF_BV')
    spec = []
    for i,op in enumerate(order):
        a,b,c = z3.BitVecs(f'prefix_a{i} prefix_b{i} prefix_c{i}',3)
        k0,k1 = z3.BitVecs(f'prefix_k{i}_0 prefix_k{i}_1',width)
        for v in (a,b,c):
            solver.add(z3.ULT(v,i+3))
        if op != '-':
            solver.add(z3.ULE(a,b))
        if op != 'mad':
            solver.add(c == 0)
        spec.append((a,b,c,k0,k1))
    output = z3.BitVec('prefix_output',3)
    key = z3.BitVec('prefix_output_key',width)
    solver.add(z3.ULT(output,4))

    def program(x, model=None):
        values = [x]
        for op,row in zip(order,spec):
            a,b,c,k0,k1 = [model.eval(v,model_completion=True) if model else v for v in row]
            pool = values+[k0,k1]
            values.append(binary(op,choose(a,pool),choose(b,pool),choose(c,pool)))
        idx = model.eval(output,model_completion=True) if model else output
        k = model.eval(key,model_completion=True) if model else key
        return z3.simplify(choose(idx,values+[k]))

    mask = (1 << width)-1
    samples = sorted({x & mask for x in [*range(32),256,264,512,513,1024,1032,1288]})
    for x in samples:
        solver.add(program(z3.BitVecVal(x,width)) == (fn(x) & mask))
    started = time.monotonic()
    events = []
    while True:
        remaining = timeout_ms-int(1000*(time.monotonic()-started))
        if remaining <= 0:
            result = 'unknown'
            break
        solver.set(timeout=remaining)
        status = solver.check()
        events.append(dict(samples=len(samples), result=str(status)))
        if status != z3.sat:
            result = str(status)
            break
        model = solver.model()
        # Independent scalar evaluation of the decoded program, all inputs.
        decoded = [[model.eval(v,model_completion=True).as_long() for v in row]
                   for row in spec]
        oi = model.eval(output,model_completion=True).as_long()
        ok = model.eval(key,model_completion=True).as_long()
        failure = None
        for x in range(1 << width):
            values = [x]
            for op,(a,b,c,k0,k1) in zip(order,decoded):
                pool = values+[k0,k1]
                values.append(binary(op,pool[a],pool[b],pool[c]) & mask)
            if (values+[ok])[oi] != fn(x) & mask:
                failure = x
                break
        if failure is None:
            result = 'sat'
            break
        assert failure not in samples
        samples.append(failure)
        solver.add(program(z3.BitVecVal(failure,width)) == (fn(failure) & mask))
    report = dict(order=list(order), width=width, result=result, events=events,
        timeout_ms=timeout_ms, seconds=round(time.monotonic()-started,3), samples=samples)
    if result == 'sat':
        report.update(program=decoded, selected_output=oi, output_key=ok,
                      verification='exhaustive_low_domain', verified_inputs=1 << width)
    return report


def square_query(form, timeout_ms=15000):
    if form not in ('square-constant','square-input') or timeout_ms <= 0:
        raise ValueError('invalid square family or timeout')
    args = [sys.executable,'-m','codex.synthesize_hash_cegis','--slice','middle',
        '--width','32','--instructions','3','--op-sequence=mad,>>,mad',
        '--first-mad-form',form,'--canonical-operands','--consecutive-samples','32',
        '--high-bit-pairs','--extra-samples','16','--check-ms',str(timeout_ms),
        '--seconds',str((timeout_ms+999)//1000+5)]
    start = time.monotonic()
    p = subprocess.run(args,capture_output=True,text=True)
    return dict(form=form, command=args, returncode=p.returncode,
        results=[json.loads(line) for line in p.stdout.splitlines() if line.startswith('{')],
        stderr=p.stderr, seconds=round(time.monotonic()-start,3))


def coverage(affine_closed, squares, prefixes, refinement=()):
    original = json.loads((ROOT/'middle_shift_orders863_results.json').read_text())
    previous = json.loads((ROOT/'middle_shift_polynomial863_results.json').read_text())
    assert previous['all_representative_counts_excluded']
    assert len(squares) == 2 and {r['form'] for r in squares} == {'square-constant','square-input'}
    assert len(original) == 48 and len({tuple(r['sequence']) for r in original}) == 48
    new_mad_shift_mad = affine_closed and all(r['returncode'] == 0 and r['results']
        and r['results'][-1]['result'] == 'grammar_unsatisfiable' for r in squares)
    prefix_closed = {tuple(r['order']) for r in prefixes if r['result'] == 'unsat'}
    # A polynomial has zero mixed difference on the 8-by-256 rectangle
    # modulo2048; F has1024. Keep raw solver UNKNOWN records unchanged.
    prefix_closed.update(order for order in prefix_orders()
                         if all(op in ('+','-','*','mad') for op in order))
    refined = {tuple(r['sequence']) for r in refinement
               if r['returncode'] == 0 and r['results'] and
               r['results'][-1]['result'] == 'grammar_unsatisfiable'}
    rows = []
    for row in original:
        seq = tuple(row['sequence'])
        reasons = []
        if row['returncode'] == 0 and row['results'][-1]['result'] == 'grammar_unsatisfiable':
            reasons.append('previous full32 fixed-order UNSAT')
        if seq == ('>>','mad','mad'):
            reasons.append('previous complete first-SHR polynomial relaxation')
        if seq == ('mad','>>','mad') and new_mad_shift_mad:
            reasons.append('affine proof and two full32 square-first UNSAT queries')
        if seq[-1] == '>>' and seq[:2] in prefix_closed:
            reasons.append('high-output cover and nonshift prefix exclusion')
        # +,-,* each admit a single MAD with arbitrary constants. Their
        # replacements preserve the position and operands of the SHR.
        if new_mad_shift_mad and any(op in ('+','-','*') for op in seq):
            reasons.append('arithmetic opcode lowering to one of the three closed two-MAD/SHR orders')
        if seq in (('>>','mad','&'),('>>','mad','|')):
            reasons.append('permutation and bit-containment certificate for final AND/OR')
        if seq in refined:
            reasons.append('refined full32 fixed-order UNSAT')
        rows.append(dict(sequence=seq, excluded=bool(reasons), evidence=reasons,
                         raw_previous_result=row['results'][-1]['result']))
    return dict(orders=rows, closed=sum(r['excluded'] for r in rows),
        unresolved=sum(not r['excluded'] for r in rows),
        scope='One uint32 input; one MAD, one SHR and one operation from '
              '+,-,XOR,AND,OR,SHL,SHR,MUL,MAD. No extra live inputs or whole-kernel bound.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--timeout-ms',type=int,default=10000)
    ap.add_argument('--refinement',type=Path,
                    help='Include raw full32 fixed-order refinement results in coverage')
    args = ap.parse_args()
    witnesses = finite_witnesses()
    pair = top_pair_query()
    variable = variable_count_query(args.timeout_ms)
    print(json.dumps(dict(variable_count=variable['result'], top_pair=pair['result'])),flush=True)
    prefixes = []
    for order in prefix_orders():
        row = prefix_query(order,timeout_ms=args.timeout_ms)
        prefixes.append(row)
        print(json.dumps({k:row[k] for k in ('order','result','seconds')}),flush=True)
    squares = [square_query(form) for form in ('square-constant','square-input')]
    controls = [prefix_query(order,timeout_ms=args.timeout_ms,fn=fn) for order,fn in (
        (('mad','^'),lambda x: (3*x+5)^x),
        (('&','mad'),lambda x: 3*(x&0x36)+5),
        (('mad','mad'),lambda x: (3*x+5)*7+x))]
    count_control = lambda x: (x+full_shift(x,((1 << 31)*x+9)&MASK)) & MASK
    variable_control = variable_count_query(args.timeout_ms,count_control,
                                             fixed_affine=(1 << 31,9))
    report = dict(best_verified_cycles=864, new_executable=False,
        z3=z3.get_version_string(), literal_affine_witnesses=witnesses,
        top_pair=pair, variable_count=variable, variable_count_positive_control=variable_control,
        square_first=squares, final_shift_prefixes=prefixes, prefix_positive_controls=controls,
        final_bitwise=final_mask_certificate())
    refinement = json.loads(args.refinement.read_text()) if args.refinement else []
    report['coverage'] = coverage(pair['result']=='unsat' and variable['result']=='unsat',
                                  squares,prefixes,refinement)
    report['refinement_file'] = args.refinement.name if args.refinement else None
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(closed=report['coverage']['closed'],
        unresolved=report['coverage']['unresolved'], new_executable=False)),flush=True)


if __name__ == '__main__':
    main()
