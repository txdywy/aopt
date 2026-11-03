"""Round-0 four-instruction templates with the shift amounts pinned.

The plain search (grammar4_round0_863) leaves 228 templates undecided: z3 times
out on them with symbolic 5-bit shift amounts. Splitting each template on its
shift amount and solving every case with a concrete amount is much faster.

Single-shift templates. Every other form is a T-function, so with one shift
w >> s the output's bit 0 is a function of v0 and bit s of w, while the target's
bit 0 is v0 ^ y19 ^ const. For s < 19, bit s of w ignores input bit 19 and y19
flips with it, so only s = 19..31 remain (13 cases).

When the shift reads aff(xk(u)) = mu*(u ^ K) + nu, write mu = 2^d * odd. Bit s
of w is then bit s - d of odd*(u ^ K) + (nu >> d). For s - d < 19 it ignores
input bit 19; for s - d in 20..29, flipping input bit s - d flips it while y19
stays (y19 depends only on v mod 2^20), so for s <= 29 the multiplier has
exactly d = s - 19 trailing zeros. That side constraint is added.

Templates whose u feeds only an AND/OR with a constant are excluded outright:
the program must be injective on v < 2^30 (the target is), so u & K needs every
bit 0..29 of K (it is u) and u | K none of them (it is u + top bits of K). The
template is then one with an affine shift input, which the bit-0 lemma refutes.

For those aff(xk(u)) templates at s <= 29 the bit-0 condition at s - d = 19 is
solved exactly by grammar4_round0_bit0_863 (32 solutions for K mod 2^20 and the
odd parts of mu, nu); the case is solved under the constraint that bits
d..d+19 of the constants form one of them.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import random
import time

from codex import grammar3_head863 as g
from codex import grammar4_round0_863 as h

AUDIT = Path(__file__).with_name('grammar4_round0_audit863.json')


def open_templates():
    rows = json.loads(AUDIT.read_text())['rest']
    return [r['idx'] for r in rows if r['resolved'] == 'open']


def shifts(p):
    return [k for k, (f, _) in enumerate(p) if f == 'shrk']


def single_use_mask(p):
    """'andk'/'ork' if u feeds only instruction 1 and that is an AND/OR with a constant."""
    users = [k for k, (_, args) in enumerate(p) if 0 in args]
    if users == [0] and p[0][0] in ('andk', 'ork'):
        return p[0][0]
    return None


def xk_aff_input(p, k):
    """Index of the aff instruction if shift k reads aff(xk(u)), else None."""
    (j,) = p[k][1]
    if j and p[j - 1][0] == 'aff':
        (i,) = p[j - 1][1]
        if i and p[i - 1] == ('xk', (0,)):
            return j - 1
    return None


def solve_pinned(p, root, amounts, seconds=60, low=None, seed=863, hints=(), pins=()):
    """g.solve_domain with constant shift amounts {instruction: s}.

    low: optional list of (positions, allowed): positions are (instr, const, lo)
    triples, allowed a list of value tuples for bits lo..lo+19 of those constants.
    hints: (instr, const, t) triples meaning v2(constant) <= t, each proved by an exact rule
    of grammar4_round0_pinned_audit863 (which lists them with the case).
    pins: (instr, const, lo, width, value): bits lo..lo+width-1 of the constant equal value
    (one class of grammar4_round0_fiber863).
    """
    import z3
    sem = g.ops(z3)
    consts = []
    for n, (f, _) in enumerate(p):
        if f == 'shrk':
            consts.append([z3.BitVecVal(amounts[n], 32)])
        else:
            consts.append([z3.BitVec(f'k{n}_{j}', 32) for j in range(g.NCONST.get(f, 0))])
    m, c = z3.BitVec('m', 32), z3.BitVec('c', 32)
    solver = z3.SolverFor('QF_BV')
    solver.add(z3.Extract(0, 0, m) == 1)
    for k, s in amounts.items():
        j = xk_aff_input(p, k)
        if j is not None and s <= 29:
            d = s - 19
            solver.add(z3.Extract(d, 0, consts[j][0]) == 1 << d)
    for n, i, t in hints:
        solver.add(z3.Extract(t, 0, consts[n][i]) != 0)
    for n, i, lo, width, value in pins:
        solver.add(z3.Extract(lo + width - 1, lo, consts[n][i]) == value)
    for positions, allowed in low or []:
        solver.add(z3.Or([z3.And([z3.Extract(lo + 19, lo, consts[n][i]) == val
                                  for (n, i, lo), val in zip(positions, row)]) for row in allowed]))
    rng = random.Random(seed)
    pts = [0, 1, 2, 3, 1 << 12, 1 << 19, 1 << 20, 1 << 29, g.D30 - 1, 0x2AAAAAAA, 0x7FFFF] + \
          [rng.randrange(g.D30) for _ in range(9)]
    t0 = time.monotonic()

    def add(x):
        lhs = g.run_program(p, z3.BitVecVal(x ^ root, 32), consts, sem)
        solver.add(lhs == m * z3.BitVecVal(g.target(x), 32) + c)

    for x in pts:
        add(x)
    csem = g.ops(None)
    rounds = 0
    while True:
        left = seconds - (time.monotonic() - t0)
        if left <= 0:
            return dict(result='timeout', rounds=rounds)
        solver.set(timeout=int(left * 1000))
        rounds += 1
        st = solver.check()
        if st == z3.unsat:
            return dict(result='unsat', rounds=rounds, seconds=round(time.monotonic() - t0, 2))
        if st != z3.sat:
            return dict(result='timeout', rounds=rounds)
        mdl = solver.model()
        val = lambda v: mdl.eval(v, model_completion=True).as_long()
        kv = [[val(k) for k in ks] for ks in consts]
        mv, cv = val(m), val(c)
        chk = random.Random(rounds)
        bad = next((x for x in (chk.randrange(g.D30) for _ in range(4096))
                    if g.run_program(p, x ^ root, kv, csem) != (mv * g.target(x) + cv) & g.MASK), None)
        if bad is None:
            bad = g.verify_full(p, kv, mv, cv, root, domain=g.D30)
            if bad is None:
                return dict(result='sat', consts=kv, m=mv, c=cv, rounds=rounds,
                            seconds=round(time.monotonic() - t0, 2))
        add(bad)


def xk_aff_low(p, k, s, root):
    """Exact bit-0 pinning for a shift k by s that reads aff(xk(u)), else None."""
    j = xk_aff_input(p, k)
    if j is None or s > 29:
        return None
    from codex.grammar4_round0_bit0_863 import xk_affine_solutions
    (i,) = p[j][1]
    d = s - 19
    return [([(i - 1, 0, 0), (j, 0, d), (j, 1, d)], [list(r) for r in xk_affine_solutions(root)])]


def job(args):
    idx, amounts, seconds, root, low, *rest = args
    hints = rest[0] if rest else ()
    p = h.templates4()[idx] if idx not in _CACHE else _CACHE[idx]
    t0 = time.monotonic()
    try:
        r = solve_pinned(p, root, amounts, seconds, low, hints=hints)
    except Exception as exc:  # report, never hide
        r = dict(result='error', error=f'{type(exc).__name__}: {exc}'[:200])
    r.update(idx=idx, amounts={str(k): s for k, s in amounts.items()}, template=g.describe(p),
             pinned_low=bool(low), wall=round(time.monotonic() - t0, 2))
    if hints:
        r['hints'] = [list(x) for x in hints]
    return r


_CACHE: dict = {}


def init_worker():
    for i, p in enumerate(h.templates4()):
        _CACHE[i] = p


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--root', type=lambda s: int(s, 0), default=h.ROOTS[0])
    ap.add_argument('--only', type=str, default='', help='idx or idx:s list')
    ap.add_argument('--skip-done', action='store_true')
    ap.add_argument('--types', default='', help='only shift inputs of these kinds: xkaff')
    ap.add_argument('--combos', type=Path, help='JSON {idx: [[s, ...], ...]} of shift amounts per template')
    args = ap.parse_args()
    ts = h.templates4()
    todo = []
    hint_of = {}
    if args.combos:
        for idx, combos in json.loads(args.combos.read_text()).items():
            sh = shifts(ts[int(idx)])
            for c in combos:
                if isinstance(c, dict):        # {"amounts": [...], "hints": [[n, i, t], ...]}
                    hint_of[(int(idx), tuple(c['amounts']))] = tuple(map(tuple, c['hints']))
                    c = c['amounts']
                todo.append((int(idx), dict(zip(sh, c))))
    elif args.only:
        for item in args.only.split(','):
            idx, _, s = item.partition(':')
            sh = shifts(ts[int(idx)])
            assert len(sh) == 1
            for amount in ([int(s)] if s else range(19, 32)):
                todo.append((int(idx), {sh[0]: amount}))
    else:
        for idx in open_templates():
            p = ts[idx]
            if single_use_mask(p) or len(shifts(p)) != 1:
                continue
            if args.types == 'xkaff' and xk_aff_input(p, shifts(p)[0]) is None:
                continue
            todo += [(idx, {shifts(p)[0]: s}) for s in range(19, 32)]
    done = set()
    if args.skip_done and args.output.exists():
        for line in args.output.open():
            r = json.loads(line)
            done.add((r['idx'], tuple(sorted(r['amounts'].items()))))
    low = lambda i, a: xk_aff_low(ts[i], *next(iter(a.items())), args.root) if len(a) == 1 else None
    hints = lambda i, a: hint_of.get((i, tuple(s for _, s in sorted(a.items()))), ())
    jobs = [(i, a, args.seconds, args.root, low(i, a), hints(i, a)) for i, a in todo
            if (i, tuple(sorted((str(k), s) for k, s in a.items()))) not in done]
    print(json.dumps(dict(cases=len(todo), todo=len(jobs))), flush=True)
    with args.output.open('a' if args.skip_done else 'w') as out, \
            ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
        for f in as_completed([pool.submit(job, j) for j in jobs]):
            r = f.result()
            out.write(json.dumps(r) + '\n')
            out.flush()
            if r['result'] != 'unsat':
                print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
