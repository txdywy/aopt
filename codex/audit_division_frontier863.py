"""Revisit the division gap in twoop32_closure863 without changing the kernel.

The historical enumerator omitted cdiv. Its 346 UNKNOWN results were also
closed by an injectivity argument which does not account for an original
input bypass (x ^ (x // 65536) is a counterexample to that general argument).
This tool retains bypasses, uses the frozen 33-bit cdiv semantics, requires
nonzero divisors, and leaves timeouts inconclusive.

Coverage: all two-instruction programs with cdiv in either position, and all
programs whose first instruction is // or %. A division in the first position
must have a constant nonzero denominator: a live-input denominator is zero on
the valid all-zero input. Constant-only first instructions are also admitted
by the general-second-cdiv queries. There are two private constants per
instruction, arbitrary operand choices, and all 14 frozen arithmetic opcodes.
The finite sample set is a necessary condition only. SAT models are subjected
to a separate full-uint32 equivalence query before being called candidates.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import random
import time

import z3

from codex.search_hash_transport863 import OPS, choose, operation

MASK = (1 << 32) - 1
C1 = 0xC761C23C


def objective(name, xs):
    x = xs[0]
    if name == 'g19C1':
        return x ^ z3.LShR(x, 19) ^ C1
    if name == 'g16n':
        return x ^ z3.LShR(x, 16) ^ xs[1]
    if name == 'bypass_control':
        return x ^ z3.LShR(x, 16)
    if name == 'node_control':
        return x ^ xs[1]
    if name == 'ceil_control':
        a = z3.ZeroExt(1, x)
        return z3.Extract(31, 0, z3.UDiv(a + 65535, z3.BitVecVal(65536, 33)))
    raise ValueError(name)


def points(name):
    n = 2 if name in ('g16n', 'node_control') else 1
    edge = [0, 1, MASK, 0xFFFF, 0x10000, 0xFFFF0000,
            0x7FFFFFFF, 0x80000000, C1]
    out = [tuple(v for _ in range(n)) for v in edge]
    if n == 2:
        out += [(v, 0) for v in edge[:6]] + [(0, v) for v in edge[:6]]
    rng = random.Random(863)
    out += [tuple(rng.getrandbits(32) for _ in range(n)) for _ in range(12)]
    return list(dict.fromkeys(out))


def search(name, op0, op1, input_index=None, seconds=6.0):
    start = time.monotonic()
    deadline = start + seconds
    n = 2 if name in ('g16n', 'node_control') else 1
    ks = z3.BitVecs('k0 k1 k2 k3', 32)
    selectors = []
    solver = z3.SolverFor('QF_BV')
    for i, size in enumerate((n + 2, n + 3)):
        width = (size - 1).bit_length()
        ss = z3.BitVecs(f'a{i} b{i} c{i}', width)
        for s in ss:
            if size < (1 << width):
                solver.add(z3.ULT(s, size))
        if (op0, op1)[i] != 'mad':
            solver.add(ss[2] == 0)
        selectors.append(ss)
    if input_index is not None:
        assert op0 in ('//', '%', 'cdiv')
        solver.add(selectors[0][0] == input_index,
                   selectors[0][1] == n, ks[0] != 0)

    def expr(xs, model=None):
        vals = list(xs)
        valid = []
        for i, op in enumerate((op0, op1)):
            pool = vals + list(ks[2*i:2*i+2])
            # vals after instr0 contains [inputs..., t0], matching pool1.
            ss = selectors[i]
            if model is not None:
                pool = [z3.simplify(z3.substitute(v, *[(k, model.eval(k, model_completion=True))
                                                      for k in ks])) for v in pool]
                ss = [model.eval(s, model_completion=True) for s in ss]
            args = [choose(s, pool) for s in ss]
            value, ok = operation(op, *args)
            vals.append(value)
            valid.append(ok)
        return vals[-1], z3.And(*valid)

    pts = points(name)
    for p in pts:
        xs = [z3.BitVecVal(v, 32) for v in p]
        got, ok = expr(xs)
        solver.add(ok, got == objective(name, xs))
    row = dict(target=name, op0=op0, op1=op1, first_input=input_index,
               first_denominator='constant' if input_index is not None else 'any')
    iterations = 0
    while time.monotonic() < deadline:
        solver.set(timeout=max(1, int(1000*(deadline-time.monotonic()))))
        result = solver.check()
        iterations += 1
        if result != z3.sat:
            row.update(status=str(result), reason=solver.reason_unknown() if result == z3.unknown else None)
            break
        model = solver.model()
        xs = z3.BitVecs(' '.join(f'x{i}' for i in range(n)), 32)
        got, ok = expr(xs, model)
        checker = z3.SolverFor('QF_BV')
        checker.set(timeout=max(1, int(1000*(deadline-time.monotonic()))))
        checker.add(z3.Or(z3.Not(ok), got != objective(name, xs)))
        checked = checker.check()
        if checked == z3.unsat:
            row.update(status='proved_equivalent',
                       constants=[model.eval(k, model_completion=True).as_long() for k in ks],
                       operands=[[model.eval(s, model_completion=True).as_long() for s in ss]
                                 for ss in selectors])
            break
        if checked == z3.unknown:
            row.update(status='verification_unknown', reason=checker.reason_unknown())
            break
        m = checker.model()
        p = tuple(m.eval(x, model_completion=True).as_long() for x in xs)
        pts.append(p)
        concrete = [z3.BitVecVal(v, 32) for v in p]
        got, ok = expr(concrete)
        solver.add(ok, got == objective(name, concrete))
    else:
        row.update(status='unknown', reason='deadline')
    row.update(samples=len(pts), iterations=iterations,
               seconds=round(time.monotonic()-start, 3))
    return row


def job(args):
    return search(*args)


def controls():
    # Verify both examples on the real frozen scalar ISA, including ceil overflow.
    from types import SimpleNamespace
    from codex.research_packet_870 import Machine
    machine = Machine.__new__(Machine)
    machine.trace = None
    core = SimpleNamespace(scratch=[0]*12)
    vectors = [0, 1, 65535, 65536, MASK, 0xFFFFFFFF, 0x80000000]
    for x in vectors:
        core.scratch[0:2] = [x, 65536]
        machine.scratch_write = {}
        machine.alu(core, '//', 2, 0, 1)
        core.scratch[2] = machine.scratch_write[2]
        machine.alu(core, '^', 3, 0, 2)
        assert machine.scratch_write[3] == x ^ (x >> 16)
        machine.alu(core, 'cdiv', 4, 0, 1)
        assert machine.scratch_write[4] == (x + 65535)//65536
    rows = [search('bypass_control', '//', '^', 0, 30),
            search('ceil_control', 'cdiv', '^', 0, 30),
            search('node_control', '//', '^', 0, 30)]
    assert all(r['status'] == 'proved_equivalent' for r in rows), rows
    return dict(frozen_isa_examples=len(vectors), full32_search_controls=rows)


def refine_remainder_mad():
    """Cover every D with a superset of all final-MAD wirings.

    P(x,x%D) is a bivariate quadratic, even if the MAD bypasses the
    remainder. For D>4, x%D=x at 0,1,2,4, so P is an ordinary quadratic on
    these four inputs. D=1..4 are checked separately with six arbitrary
    polynomial coefficients. UNSAT on finite samples is sufficient here.
    """
    rows = []
    for d in [None, 1, 2, 3, 4]:
        solver = z3.SolverFor('QF_BV')
        solver.set(timeout=30000)
        if d is None:
            xs = [0, 1, 2, 4]
            ks = z3.BitVecs('qa qb qc', 32)
        else:
            xs = list(range(65)) + [127, 128, 255, 256, 257, 1024,
                                    65535, 65536, 0x7FFFFFFF, MASK]
            ks = z3.BitVecs('qxx qxt qtt qx qt qc', 32)
        for x in xs:
            if d is None:
                basis = [x*x, x, 1]
            else:
                t = x % d
                basis = [x*x, x*t, t*t, x, t, 1]
            got = sum(k*z3.BitVecVal(b & MASK, 32) for k, b in zip(ks, basis))
            want = (x ^ (x >> 19) ^ C1) & MASK
            solver.add(got == want)
        status = solver.check()
        rows.append(dict(divisor='D>4' if d is None else d,
                         status=str(status), samples=xs,
                         coefficients=len(ks),
                         reason=solver.reason_unknown() if status == z3.unknown else None))
    return dict(target='g19C1', grammar='x%D followed by arbitrary MAD',
                covering_relaxation='quadratic in x and x%D',
                cases=rows, closed=all(r['status'] == 'unsat' for r in rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds', type=float, default=6)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--target', choices=['g19C1', 'g16n', 'both'], default='both')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--controls-only', action='store_true')
    ap.add_argument('--refine-only', action='store_true')
    args = ap.parse_args()
    result = dict(z3=z3.get_version_string(), controls=controls(), queries=[])
    print(json.dumps(result['controls']), flush=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    if args.refine_only:
        result['remainder_mad_refinement'] = refine_remainder_mad()
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result['remainder_mad_refinement']), flush=True)
        return
    if args.controls_only:
        return
    jobs = []
    for name in (('g19C1', 'g16n') if args.target == 'both' else (args.target,)):
        n = 2 if name == 'g16n' else 1
        for op0 in ('//', '%', 'cdiv'):
            for j in range(n):
                jobs += [(name, op0, op1, j, args.seconds) for op1 in OPS]
        # First cdiv was covered above; second cdiv for every other first op.
        jobs += [(name, op0, 'cdiv', None, args.seconds) for op0 in OPS if op0 != 'cdiv']
    with ProcessPoolExecutor(args.workers) as pool:
        for future in as_completed([pool.submit(job, j) for j in jobs]):
            row = future.result()
            result['queries'].append(row)
            print(json.dumps(row), flush=True)
            args.output.write_text(json.dumps(result, indent=2)+'\n')
    from collections import Counter
    result['counts'] = dict(Counter(r['status'] for r in result['queries']))
    result['new_executable'] = False
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['counts']), flush=True)


if __name__ == '__main__':
    main()
