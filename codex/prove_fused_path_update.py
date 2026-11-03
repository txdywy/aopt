"""Exclude single-ISA-op path updates on a concrete legal input subset.

Constants may be arbitrary uint32. UNSAT on this subset excludes a total
replacement; SAT would only be a candidate, never an equivalence proof.
"""
import json
import random
import z3


def check(op, physical=False):
    a, b, c = [z3.Int(name) for name in ('a', 'b', 'c')]
    k = [z3.BitVec('k' + str(i), 32) for i in range(3)]
    solver = z3.Solver()
    solver.set(timeout=3000)
    solver.add(*[z3.And(x >= 0, x < 5) for x in (a, b, c)])
    rng = random.Random(900)
    samples = [(m, q) for m in (0, 1, 2, 7) for q in
               (0, 1, 2, 3, 0xfffffffe, 0xffffffff, rng.getrandbits(32))]
    for m, q in samples:
        # Physical-address case uses legal depth-3 node addresses.
        x = m + 14 if physical else m
        args = [z3.BitVecVal(x, 32), z3.BitVecVal(q, 32)] + k
        def pick(index):
            out = args[-1]
            for i in range(3, -1, -1):
                out = z3.If(index == i, args[i], out)
            return out
        left, right, addend = pick(a), pick(b), pick(c)
        expr = {'+': lambda: left + right, '-': lambda: left - right,
                '*': lambda: left * right, '&': lambda: left & right,
                '|': lambda: left | right, '^': lambda: left ^ right,
                '>>': lambda: z3.LShR(left, right), '<<': lambda: left << right,
                'mad': lambda: left * right + addend}[op]()
        wanted = 2*x - 5 - (q & 1) if physical else 2*x + (q & 1)
        solver.add(expr == z3.BitVecVal(wanted & 0xffffffff, 32))
    status = solver.check()
    return {'operation': op, 'target': 'physical_address' if physical else 'packed_path',
            'status': str(status), 'samples': len(samples),
            'reason': solver.reason_unknown() if status == z3.unknown else 'finite necessary constraints'}


if __name__ == '__main__':
    m, q = z3.BitVecs('control_m control_q', 32)
    control = z3.Solver()
    control.add(2*m + (q & 1) != m + m + z3.URem(q, 2))
    assert control.check() == z3.unsat
    for physical in (False, True):
        for op in ('+', '-', '*', '&', '|', '^', '>>', '<<', 'mad'):
            print(json.dumps(check(op, physical)), flush=True)
