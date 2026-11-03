"""Three-instruction grammar for the hash tail with the C5 XOR folded in.

The hash ends with MAD (x9 + C4), >>16, XOR and XOR C5. The 864 kernel moves C5
onto the node values of d0-d8, but it still spends one XOR per element where
that is impossible: the d9 and d10 recoveries and the output, 768 lanes. If the
four instructions

    out(a) = L16(9*a + C4) XOR C5,     L16(e) = e XOR (e >> 16)

fit in three, those XORs disappear (863 needs 53 lanes).

Both ends have some freedom:

  * input: a is the middle result p XOR q with q = 512*(33*z + C2). q's low nine
    bits are zero, so changing q's constant XORs any k < 512 into a; adding 2^31
    to it flips bit 31; and p's MAD with multiplier -33 complements a. The free
    input keys are therefore K = k ^ t*2^31 ^ s*(2^32-1), 2048 values.
  * output: at d9/d10 the next MAD0 absorbs an XOR key O in {0, 2^31, 2^32-1,
    2^31-1}; the stored output needs O = 0. Allowing all four is a superset.

Query: is there a program P of three instructions with

    P(a) == L16(9*(a ^ K) + C4) ^ C5 ^ O     for every uint32 a ?

Grammar: the 46 forms of grammar3_head863 (all VALU opcodes, with --full also
//, %, cdiv, <, ==), any SSA wiring, free constants. Sound pruning, as there:
every instruction is used; there is a non-T-function op (flipping input bit 16
always flips output bit 0); P is a bijection, so the injectivity rules apply.
Unlike the head, an affine last instruction is kept: the output must be exact.

K ranges over the 2048 free keys; --any-key lets it be any 32-bit value (a
superset, used for the basic pass). Templates that time out are split on their
shift amount. UNSAT on the sample set excludes the template for every constant.

Division templates left by z3 (division_rule; grammar3_division_valuation863 has
the same rules for the head). Flipping input bit j >= 16 moves the target by
2-adic order exactly j - 16 (e moves by an odd multiple of 2^j, so the lowest
changed bit of e ^ (e >> 16) is j - 16), for every K and O. Shapes: i1 = aff(u)
or u ^ K1, i2 = OP(i1), i3 = a combine of u, i1, i2.

  even multiplier  an even multiplier leaves i1, i2 fixed when u31 flips, so the
                   output moves by a multiple of 2^31, not order 15: i1 is a
                   bijection and the u31 flip maps it to i1 ^ 2^31.
  zero-divisor     OP divides by i1 (K // i1, K % i1, cdiv(K, i1)): i1 = 0 somewhere.
  The combine is A + B*i2 (A, B T-functions) or X ^ i2. The u16 flip moves the
  output by B*di2 mod 2^16 and needs order 0, so B is odd at every input.
  bijective-coef   B is u, u + 1, i1 or i1 + 1: even for half the inputs.
  For constant B the u31 flip needs order 15 for the move of OP(i1) at every u
  (q = 2^31 // K, r = 2^31 mod K; i1 mod K takes every residue when K <= 2^31):
  div-valuation    i1 // K moves by q or q + 1, both occurring unless r = 0; one of
                   them is odd. K = 2^k is the shift template of the basic grammar.
  mod-valuation    i1 % K moves by r or r - K; both of order 15 forces v2(K) >= 16,
                   but 2^v2(K) divides r. r = 0: no move; K > 2^31: i1 = 0 moves by 2^31.
  cdiv-pinned      cdiv(i1, K) moves by two values unless K = 2^k (exactly 2^(31-k)),
                   so K = 2^16; that case is solved with K pinned.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
from pathlib import Path
import random
import time

from codex import grammar3_head863 as g

MASK = g.MASK
C4, C5 = 0xFD7046C5, 0xB55A4F09
OKEYS = (0, 1 << 31, MASK, (1 << 31) - 1)


def L16(e):
    return e ^ (e >> 16)


def target(a, key=0, okey=0):
    return L16((9 * (a ^ key) + C4) & MASK) ^ C5 ^ okey


def free_keys():
    """XOR keys on a that the middle hash produces for free."""
    return sorted({k ^ (t << 31) ^ (MASK if s else 0) for k in range(512) for t in (0, 1) for s in (0, 1)})


def templates(full=False):
    """Canonical three-instruction templates after sound pruning (full: only those using //, %, cdiv, <, ==)."""
    out, seen = [], set()
    un = g.UNARY + (g.EXTRA_UNARY if full else ())
    bi = g.BINARY + (g.EXTRA_BINARY if full else ())
    i1s = [(f, (0,)) for f in un]
    i2s = [(f, (0,)) for f in un] + [(f, (1,)) for f in un] + [(f, (0, 1)) for f in bi]
    i3s = ([(f, (2,)) for f in un] + [(f, (0, 2)) for f in bi] +
           [(f, (1, 2)) for f in bi] + [(f, (0, 1, 2)) for f in g.TERNARY])
    for p in itertools.product(i1s, i2s, i3s):
        if not any(f in g.NONT for f, _ in p):
            continue
        used = {a for _, args in p[1:] for a in args}
        if 1 not in used:
            continue
        f3, a3 = p[2]
        if len(a3) == 1 and f3 not in ('xk', 'aff'):
            continue  # never a bijection
        if len(a3) == 1 and p[1][0] in g.NONINJ and len(p[1][1]) == 1:
            continue
        if p[0][0] in g.NONINJ and 0 not in used:
            continue
        if f3 in ('mod_ab', 'mod_ba', 'lt_ab', 'lt_ba', 'eq'):
            continue  # never surjective
        if len(a3) == 1 and p[1][0] in ('lt_ab', 'lt_ba', 'eq'):
            continue
        if any(f in g.DIVISOR and g.DIVISOR[f] != 'k' and args['ab'.index(g.DIVISOR[f])] == 0
               for f, args in p):
            continue  # zero divisor at a = 0
        key = g.canonical(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    if full:
        basic = {g.canonical(p) for p in templates(False)}
        out = [p for p in out if g.canonical(p) not in basic]
    return out


def solve(p, seconds=60, amounts=None, free_only=False, seed=863, pins=()):
    """CEGIS over the template's constants, K and O.

    amounts: optional {instruction: s} pinning the shift amounts. free_only restricts
    K to free_keys(); otherwise K is any 32-bit value. pins: (instr, const, value)
    triples fixing constants. Returns unsat, timeout or sat-sampled (a candidate that
    passed 65,536 random inputs; never seen so far).
    """
    import z3
    sem = g.ops(z3)
    consts = []
    for n, (f, _) in enumerate(p):
        if f == 'shrk':
            consts.append([z3.BitVecVal(amounts[n], 32) if amounts and n in amounts
                           else z3.ZeroExt(27, z3.BitVec(f'k{n}_0', 5))])
        else:
            consts.append([z3.BitVec(f'k{n}_{j}', 32) for j in range(g.NCONST.get(f, 0))])
    if free_only:
        k9, b31, bm = z3.BitVec('k9', 9), z3.BitVec('b31', 1), z3.BitVec('bm', 1)
        K = z3.ZeroExt(23, k9) ^ z3.Concat(b31, z3.BitVecVal(0, 31)) ^ z3.SignExt(31, bm)
    else:
        K = z3.BitVec('K', 32)
    osel = z3.BitVec('osel', 2)
    rng = random.Random(seed)
    pts = [0, 1, 2, 3, 1 << 16, 1 << 15, 1 << 31, MASK, 0x80000001, 0xFFFF, 0x10001] + \
          [rng.getrandbits(32) for _ in range(9)]
    solver = z3.SolverFor('QF_BV')
    for n, j, v in pins:
        solver.add(consts[n][j] == v)
    t0 = time.monotonic()

    def add(a):
        divs = []
        lhs = g.run_program(p, z3.BitVecVal(a, 32), consts, sem, divs)
        for d in divs:
            solver.add(d != 0)
        e = 9 * (z3.BitVecVal(a, 32) ^ K) + C4
        base = e ^ z3.LShR(e, 16) ^ C5
        rhs = z3.If(osel == 0, base, z3.If(osel == 1, base ^ OKEYS[1],
                    z3.If(osel == 2, base ^ OKEYS[2], base ^ OKEYS[3])))
        solver.add(lhs == rhs)

    for a in pts:
        add(a)
    rounds = 0
    csem = g.ops(None)
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
        kval, oval = val(K), OKEYS[val(osel)]
        chk = random.Random(rounds)
        bad = None
        for _ in range(1 << 16):
            a = chk.getrandbits(32)
            try:
                ok = g.run_program(p, a, kv, csem) == target(a, kval, oval)
            except ZeroDivisionError:
                ok = False
            if not ok:
                bad = a
                break
        if bad is None:
            return dict(result='sat-sampled', consts=kv, K=kval, O=oval, rounds=rounds)
        add(bad)


# combine of (u, i1, i2) -> coefficient of i2
COMBINE_B = {
    ('lin1', (0, 2)): '1', ('lin1', (1, 2)): '1', ('lin2', (0, 2)): 'K', ('lin2', (1, 2)): 'K',
    ('prod', (0, 2)): 'u', ('prod', (1, 2)): 'i1', ('aab', (0, 2)): '1', ('aab', (1, 2)): '1',
    ('aba', (0, 2)): 'u', ('aba', (1, 2)): 'i1', ('abb', (0, 2)): 'u+1', ('abb', (1, 2)): 'i1+1',
    ('mad_c', (0, 1, 2)): '1', ('mad_b', (0, 1, 2)): 'u', ('mad_a', (0, 1, 2)): 'i1',
    ('xor', (0, 2)): 'xor', ('xor', (1, 2)): 'xor',
}
FIRST = {('aff', (0,)), ('xk', (0,))}


def division_rule(p):
    """Exact rule excluding a division template of the shape i1 in FIRST, i2 = OP(i1),
    i3 = combine; 'cdiv-pinned' means only the divisor 2^16 is left. See the docstring."""
    (f1, a1), (f2, a2), (f3, a3) = p
    if (f1, a1) not in FIRST or a2 != (1,):
        return None
    if f2 in ('kdiv', 'kmod', 'kcdiv'):
        return 'zero-divisor'
    coef = COMBINE_B.get((f3, a3))
    if coef is None:
        return None
    if coef != 'xor' and coef not in ('1', 'K'):
        return 'bijective-coef'
    return {'divk': 'div-valuation', 'modk': 'mod-valuation', 'cdivk': 'cdiv-pinned'}.get(f2)


RESULTS = Path(__file__).with_name('grammar3_tail_c5_results.jsonl.gz')
AUDIT = Path(__file__).with_name('grammar3_tail_c5_audit.json')


def audit(results=RESULTS):
    """Per-template status from every stored run: solver (main pass UNSAT), split (all
    shift amounts UNSAT), rule (division_rule; cdiv-pinned and ork-bits also need their
    pinned runs UNSAT) or open."""
    import collections
    import gzip
    runs = collections.defaultdict(list)
    with gzip.open(results, 'rt') as f:
        for line in f:
            r = json.loads(line)
            runs[(r['full'], r['idx'])].append(r)
    out, tally = [], collections.Counter()
    for full in (False, True):
        for i, p in enumerate(templates(full)):
            rs = runs[(full, i)]
            if any(r['result'] == 'unsat' and r['run'] in ('main', 'rerun') for r in rs):
                method = 'solver'
            elif any(r['run'] == 'pin' for r in rs):
                pins = [r for r in rs if r['run'] == 'pin']
                need = 32 ** sum(f == 'shrk' for f, _ in p)
                method = 'split' if len(pins) == need and all(r['result'] == 'unsat' for r in pins) else 'open'
            else:
                rule = division_rule(p) if full else None
                pinned = [r for r in rs if r['run'] == 'pinned-rule']
                if rule == 'cdiv-pinned' or (rule is None and pinned):
                    rule = rule or 'ork-bits'
                    method = rule if pinned and all(r['result'] == 'unsat' for r in pinned) else 'open'
                else:
                    method = rule or 'open'
            tally[('division' if full else 'basic', method)] += 1
            if method == 'open':
                out.append(dict(full=full, idx=i, template=g.describe(p)))
    return dict(tally={f'{a}/{b}': n for (a, b), n in sorted(tally.items())}, open=out)


def job(args):
    idx, full, seconds, amounts, free_only = args
    p = templates(full)[idx] if idx not in _CACHE.get(full, {}) else _CACHE[full][idx]
    t0 = time.monotonic()
    try:
        r = solve(p, seconds, amounts, free_only)
    except Exception as exc:  # report, never hide
        r = dict(result='error', error=f'{type(exc).__name__}: {exc}'[:200])
    r.update(idx=idx, full=full, template=g.describe(p), wall=round(time.monotonic() - t0, 2))
    if amounts:
        r['amounts'] = {str(k): s for k, s in amounts.items()}
    r['keys'] = 'free' if free_only else 'any'
    return r


_CACHE: dict = {}


def init_worker():
    for full in (False, True):
        _CACHE[full] = dict(enumerate(templates(full)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--count', action='store_true')
    ap.add_argument('--full', action='store_true', help='only templates with //, %%, cdiv, <, ==')
    ap.add_argument('--output', type=Path)
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--skip-done', action='store_true', help='resume: skip indices already in --output')
    ap.add_argument('--pin', type=Path, help='rerun the open templates of this result file, one case per shift amount')
    ap.add_argument('--any-key', action='store_true', help='any 32-bit input key instead of the free ones')
    ap.add_argument('--audit', action='store_true', help=f'summarize {RESULTS.name} into {AUDIT.name}')
    args = ap.parse_args()
    if args.audit:
        res = audit()
        AUDIT.write_text(json.dumps(res, indent=1) + '\n')
        print(json.dumps(res['tally']), len(res['open']), 'open')
        return
    ts = templates(args.full)
    if args.count:
        print(len(ts))
        return
    if args.pin:
        rows = [json.loads(line) for line in args.pin.open()]
        todo = sorted(r['idx'] for r in rows if r['result'] != 'unsat' and r.get('full', args.full) == args.full)
        jobs = []
        for i in todo:
            sh = [n for n, (f, _) in enumerate(ts[i]) if f == 'shrk']
            for combo in itertools.product(range(32), repeat=len(sh)):
                jobs.append((i, args.full, args.seconds, dict(zip(sh, combo)), True))
    else:
        done = set()
        if args.skip_done and args.output.exists():
            done = {json.loads(line)['idx'] for line in args.output.open()}
        jobs = [(i, args.full, args.seconds, None, not args.any_key) for i in range(len(ts)) if i not in done]
    print(json.dumps(dict(templates=len(ts), jobs=len(jobs))), flush=True)
    mode = 'a' if args.skip_done else 'w'
    with args.output.open(mode) as out, \
            ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
        for f in as_completed([pool.submit(job, j) for j in jobs]):
            r = f.result()
            out.write(json.dumps(r) + '\n')
            out.flush()
            if r['result'] != 'unsat':
                print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
