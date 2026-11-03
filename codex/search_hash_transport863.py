"""Screen two-instruction replacements for three hash slices.

With encoded state e before the final XOR-shift, the next first MAD consumes
L(e XOR L(node)), where L(x)=x XOR (x>>16) is an involution. Eliminating one
instruction from MAD4097(L(x)) could save a per-lane operation, rather than
only rearranging startup. This search fixes the first instruction to SHR16
and allows one subsequent original-ISA operation with arbitrary uint32
constants and operands from x, the first result, K0, K1. Optional first-step
families search other right shifts, division, remainder, affine maps, square
plus a constant, bitwise operations with constants, input-dependent shifts,
comparisons, and square plus input.
The other objectives target the middle after an invertible affine change of
variable, and the XOR/right-shift/constant slice of the first hash stage.

UNSAT excludes only this grammar. SAT candidates require a full-width
counterexample query; UNKNOWN is not a proof. No submission is modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import z3


OPS = ('+', '-', '^', '&', '|', '*', 'mad', '<<', '>>', '//', '%', 'cdiv', '<', '==')
FIRST_FORMS = ('shr16','shr','//','%','cdiv','affine','square','^','&','|',
               'square-input','constant-shl','constant-shr','self-shl','self-shr',
               'less-than','greater-than','equal')
MASK = (1 << 32) - 1


def choose(index, values):
    result = values[-1]
    for i in reversed(range(len(values) - 1)):
        result = z3.If(index == i, values[i], result)
    return result


def operation(code, a, b, c):
    valid = z3.BoolVal(True)
    if code == '+': value = a + b
    elif code == '-': value = a - b
    elif code == '^': value = a ^ b
    elif code == '&': value = a & b
    elif code == '|': value = a | b
    elif code == '*': value = a * b
    elif code == 'mad': value = a * b + c
    elif code == '<<': value = a << b
    elif code == '>>': value = z3.LShR(a, b)
    elif code in ('//', '%', 'cdiv'):
        valid = b != 0
        if code == '//': value = z3.UDiv(a, b)
        elif code == '%': value = z3.URem(a, b)
        else:
            # The simulator's ceil division adds as an unbounded Python int.
            wide_a, wide_b = z3.ZeroExt(1, a), z3.ZeroExt(1, b)
            value = z3.Extract(31, 0, z3.UDiv(wide_a + wide_b - 1, wide_b))
    elif code == '<': value = z3.If(z3.ULT(a, b), z3.BitVecVal(1, 32), z3.BitVecVal(0, 32))
    elif code == '==': value = z3.If(a == b, z3.BitVecVal(1, 32), z3.BitVecVal(0, 32))
    else: raise ValueError(code)
    return value, valid


def target(x, control=None, objective='transport'):
    shifted = z3.LShR(x, 16)
    if control == 'xor': return x ^ shifted
    if control == 'mad': return x * 4097 + shifted
    if objective == 'head-xor':
        return x ^ z3.LShR(x, 19) ^ 0xc761c23c
    if objective == 'middle-tail':
        # q=33*b+A is bijective. The second original affine branch is
        # 512*q+K, with K=B-512*A modulo 2**32.
        k = (0xaccf6200 - 512*0xe9f8cc1d) & MASK
        return 9 * (x ^ (512*x+k)) + 0xfd7046c5
    if objective == 'transport':
        return (x ^ shifted) * 4097 + 0x7ed55d16
    raise ValueError(objective)


def samples():
    rng = random.Random(863)
    values = [0, 1, MASK, 0xffff, 0x10000, 0xffff0000, 0x7fffffff]
    values += [1 << k for k in range(32)]
    values += [rng.getrandbits(32) for _ in range(24)]
    return list(dict.fromkeys(values))


def first_operation(first, x, constant, offset):
    if first == 'affine': return x*constant+offset, z3.BoolVal(True)
    if first == 'square': return x*x+constant, z3.BoolVal(True)
    if first == 'square-input': return x*x+x, z3.BoolVal(True)
    if first in ('constant-shl','constant-shr'):
        return operation('<<' if first.endswith('shl') else '>>',constant,x,x)
    if first in ('self-shl','self-shr'):
        return operation('<<' if first.endswith('shl') else '>>',x,x,x)
    if first in ('less-than','greater-than','equal'):
        a,b = (constant,x) if first == 'greater-than' else (x,constant)
        return operation('==' if first == 'equal' else '<',a,b,x)
    return operation('>>' if first in ('shr16','shr') else first,x,constant,x)


def search(code, seconds=15, control=None, first='shr16', objective='transport'):
    if first not in FIRST_FORMS: raise ValueError(first)
    solver = z3.SolverFor('QF_BV')
    a, b, c = z3.BitVecs('a b c', 2)
    k0, k1 = z3.BitVecs('k0 k1', 32)
    divisor = z3.BitVec('first_constant', 32)
    offset = z3.BitVec('first_offset', 32)
    if first == 'shr16': solver.add(divisor == 16)
    elif first == 'shr': solver.add(z3.UGE(divisor, 1), z3.ULE(divisor, 31))
    elif first in ('//','%','cdiv'): solver.add(divisor != 0)
    if code != 'mad': solver.add(c == 0)
    if code in ('+', '^', '&', '|', '*', 'mad', '=='):
        solver.add(z3.ULE(a, b))

    def program(x, model=None):
        fields = (a, b, c, k0, k1, divisor, offset)
        aa, bb, cc, ka, kb, dd, ff = [model.eval(v, model_completion=True) if model is not None else v
                              for v in fields]
        first_value, first_valid = first_operation(first,x,dd,ff)
        pool = [x, first_value, ka, kb]
        result, valid = operation(code, choose(aa, pool), choose(bb, pool), choose(cc, pool))
        return result, z3.And(first_valid, valid)

    points = samples()
    for value in points:
        x = z3.BitVecVal(value, 32)
        result, valid = program(x)
        solver.add(valid, result == target(x, control, objective))
    start = time.monotonic()
    deadline = start + seconds
    iterations = 0
    while time.monotonic() < deadline:
        iterations += 1
        solver.set(timeout=max(1, int(1000 * (deadline - time.monotonic()))))
        status = solver.check()
        if status != z3.sat:
            return dict(op=code, control=control, status=str(status),
                classification='grammar_excluded' if status == z3.unsat else 'inconclusive',
                samples=len(points), iterations=iterations, seconds=time.monotonic()-start,
                reason=solver.reason_unknown() if status == z3.unknown else None)
        model = solver.model()
        x = z3.BitVec('verification_input', 32)
        result, valid = program(x, model)
        check = z3.SolverFor('QF_BV')
        check.set(timeout=max(1, int(1000 * (deadline - time.monotonic()))))
        check.add(z3.Or(z3.Not(valid), result != target(x, control, objective)))
        status = check.check()
        if status == z3.unsat:
            return dict(op=code, control=control, status='proved_equivalent',
                classification='full_uint32_equivalence', samples=len(points),
                iterations=iterations, seconds=time.monotonic()-start,
                operands=[model.eval(v).as_long() for v in (a,b,c)],
                first_constant=model.eval(divisor,model_completion=True).as_long(),
                first_offset=model.eval(offset,model_completion=True).as_long(),
                constants=[model.eval(v,model_completion=True).as_long() for v in (k0,k1)])
        if status == z3.unknown:
            return dict(op=code, control=control, status='verification_unknown',
                classification='inconclusive', reason=check.reason_unknown(),
                samples=len(points), iterations=iterations, seconds=time.monotonic()-start)
        value = check.model().eval(x).as_long()
        points.append(value)
        x = z3.BitVecVal(value, 32)
        result, valid = program(x)
        solver.add(valid, result == target(x, control, objective))
    return dict(op=code, control=control, status='deadline', classification='inconclusive',
                samples=len(points), iterations=iterations, seconds=time.monotonic()-start)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seconds', type=float, default=15)
    ap.add_argument('--objective', choices=('transport','middle-tail','head-xor'), default='transport')
    ap.add_argument('--first', choices=FIRST_FORMS, default='shr16',
        help='First operation; SHR searches every fixed shift 1..31; division/remainder use any nonzero uint32 divisor')
    args = ap.parse_args()
    if args.seconds <= 0: ap.error('--seconds must be positive')
    report = dict(z3=z3.get_version_string(), first=args.first, objective=args.objective,
        grammar='First operation x OP D followed by one ISA op; operand pool '
        'x, first_result, K0, K1; arbitrary uint32 K0/K1. D is 16 for SHR16, '
        '1..31 for SHR, any nonzero uint32 for division/remainder. Affine is '
        'x*D+E; square is x*x+D; bitwise families use arbitrary D. '
        'square-input is x*x+x; constant shifts are D OP x; self shifts are x OP x; '
        'comparisons are x<D, D<x, x==D. Controls use SHR16.', results=[])
    for code, control in [('^','xor'), ('mad','mad')] + [(op,None) for op in OPS]:
        first = 'shr16' if control else args.first
        row = search(code, args.seconds, control, first, args.objective)
        row['first'] = first
        report['results'].append(row)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(row), flush=True)
        if control is not None and row['status'] != 'proved_equivalent':
            raise RuntimeError('Positive control did not pass; inspect before interpreting search')


if __name__ == '__main__':
    main()
