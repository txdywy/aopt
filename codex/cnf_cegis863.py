"""Solve the stage-one re-encoding queries through z3 bit-blasting and an
external CDCL solver (pysat: CaDiCaL / Kissat).

The unknowns are built from named Boolean bits so that a SAT model can be
decoded; every candidate is re-checked on random inputs, and failing inputs
are added as new samples (CEGIS). UNSAT on a sample set is a full proof for
the corresponding form.
"""
from __future__ import annotations

import json
import random
import sys
import time

import z3
from pysat.solvers import Solver

from codex.affine_reencode_xmix863 import TARGETS, a_of, conc, sym, MASK


def bitvec(name):
    bits = [z3.Bool(f'{name}_{i}') for i in range(32)]
    bv = z3.Concat(*[z3.If(b, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)) for b in reversed(bits)])
    return bv, bits


def to_cnf(assertions):
    g = z3.Goal()
    g.add(*assertions)
    res = z3.Then('simplify', 'bit-blast', 'tseitin-cnf')(g)
    assert len(res) == 1
    var, clauses = {}, []

    def lit(e):
        neg = z3.is_not(e)
        a = e.arg(0) if neg else e
        k = a.get_id()
        if k not in var:
            var[k] = (len(var) + 1, a)
        return -var[k][0] if neg else var[k][0]

    for c in res[0]:
        if z3.is_false(c):
            return None, var
        if z3.is_or(c):
            clauses.append([lit(x) for x in c.children()])
        else:
            clauses.append([lit(c)])
    return clauses, var


def solve(fam, s, tname, normalized=True, engine='cadical195', seconds=3600, seed=1):
    target = TARGETS[tname]
    names = ('mu', 'nu', 'm', 'c', 'k')
    bvs = {n: bitvec(n) for n in names}
    mu, nu, m, c, k = (bvs[n][0] for n in names)
    rng = random.Random(seed)
    pts = [0, 1, 2, 1 << 12, 1 << 19, 1 << 31, MASK] + [rng.getrandbits(32) for _ in range(9)]
    t0 = time.monotonic()
    rounds = 0
    while time.monotonic() - t0 < seconds:
        rounds += 1
        cons = [z3.Extract(0, 0, mu) == 1, z3.Extract(0, 0, m) == 1]
        for x in pts:
            w = mu * z3.BitVecVal(a_of(x), 32) + nu
            lhs = sym(fam, s, x, w, k)
            tv = z3.BitVecVal(target(x), 32)
            cons.append(lhs == m * tv + c if normalized else m * lhs + c == tv)
        clauses, var = to_cnf(cons)
        if clauses is None:
            return dict(fam=fam, s=s, target=tname, result='unsat', how='simplify', rounds=rounds)
        with Solver(name=engine, bootstrap_with=clauses) as sol:
            ok = sol.solve()
            model = set(l for l in sol.get_model()) if ok else None
        if not ok:
            return dict(fam=fam, s=s, target=tname, result='unsat', engine=engine, rounds=rounds,
                        samples=len(pts), clauses=len(clauses), seconds=round(time.monotonic() - t0, 1))
        # decode named bits; bits eliminated by simplification default to 0
        byname = {str(a): idx for idx, a in var.values()}
        vals = {}
        for n in names:
            v = 0
            for i, b in enumerate(bvs[n][1]):
                idx = byname.get(str(b))
                if idx is not None and idx in model:
                    v |= 1 << i
            vals[n] = v
        chk = random.Random(99 + rounds)
        bad = None
        for _ in range(200000):
            x = chk.getrandbits(32)
            w = (vals['mu'] * a_of(x) + vals['nu']) & MASK
            lhs, rhs = conc(fam, s, x, w, vals['k']), target(x)
            good = lhs == (vals['m'] * rhs + vals['c']) & MASK if normalized else (vals['m'] * lhs + vals['c']) & MASK == rhs
            if not good:
                bad = x
                break
        if bad is None:
            return dict(fam=fam, s=s, target=tname, result='sat', model=vals, rounds=rounds,
                        seconds=round(time.monotonic() - t0, 1))
        pts.append(bad)
    return dict(fam=fam, s=s, target=tname, result='timeout', rounds=rounds, samples=len(pts))


if __name__ == '__main__':
    fam, s, tname = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    engine = sys.argv[4] if len(sys.argv) > 4 else 'cadical195'
    print(json.dumps(solve(fam, s, tname, engine=engine)), flush=True)
