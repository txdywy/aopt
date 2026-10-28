"""Four instructions for the C5 recovery plus the hash head at d9 and d10.

At d9 and d10 the node is a raw gathered value, so the kernel computes
u = h ^ node, then u ^ C5, then the four-instruction head T (MAD0, >>19, ^, ^C1):
five instructions. A four-instruction program with

    P(u) == m*T(u ^ C5) + c      for every uint32 u,  m odd,

would save one op per element at each of the two depths, 512 lanes (863 needs 53).
F's MADs absorb the odd affine image, exactly as in grammar3_head863. This is the
round-0 question of grammar4_round0_863 on the full 32-bit domain, with the key C5
in place of the root.

Grammar: the 14 core forms of grammar4_round0_863 with any SSA wiring. Sound
pruning: every instruction is used, there is a right shift, the last instruction
is not an affine map (absorbed) and can be a bijection (the target is one).

Lemma A (bit 0). Every core form except the right shift is a T-function. If a
template has one shift w >> s, output bit 0 is f(u0, w_s); the target's is
u0 ^ y19 ^ const, where y = 4097(u ^ C5) + C0, flipping u19 always flips y19 and
bits 20..31 never touch it. Hence w_s = y19 ^ g(u0) for every u. When w is
affine in u, w = a*u + b with a = 2^d * a', flipping u_j (j >= 20) must leave w_s
alone and flipping u19 must always flip it, which forces s = 19 + d and
w_s = bit 19 of (a'u + (b >> d)). So

    bit19(a' u + b') == y19(u) ^ g(u0)    for all u < 2^20,  a' odd,

and lemma_a.c shows no (a', b', g) satisfies it for the key C5 (the same check finds
8 solutions for each affine-compatible key 0, 2^31, 2^32-1). One XOR, AND or OR
with a constant around the affine value only complements w_s or fixes it, so those
inputs are covered too. Lemma A excludes 10,160 of the 12,254 templates.

Lemma B. If every shift reads u or the result of a shift that does, output bit 0
is a function of u0 and at most three single bits of u, but each of u1..u19 changes
y19 for some u. This excludes 774 more.

z3 decides the rest (CEGIS as in grammar3_head863, with a full 2^32 check of any
candidate). Its timeouts are split on the shift amounts. A single shift reading
aff(xk(u)) = mu*(u ^ K) + nu is pinned like the round-0 case: s = 19 + d with
mu = 2^d * odd (lemma A's argument), and (K mod 2^20, bits d..d+19 of mu and nu)
is one of the 32 solutions of grammar4_round0_bit0_863.xk_affine_solutions(C5).
Two shifts reading one affine value w: the same flips force the higher read to be
bit 19 + d of w and the output bit to be that bit XOR G(u0, lower bit), the
"two reads" condition lemma_a2() checks on 20 bits.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
from pathlib import Path
import random
import subprocess
import tempfile
import time

from codex import grammar3_head863 as g
from codex import grammar4_round0_863 as h

MASK = g.MASK
C5 = 0xB55A4F09
LEMMA_C = Path(__file__).with_name('grammar4_head_c5_lemma863.c')


def templates():
    out = []
    for p in h.templates4():
        f4, a4 = p[3]
        if len(a4) == 1 and f4 != 'xk':
            continue  # never a bijection
        if len(a4) == 1 and p[2][0] in g.NONINJ and len(p[2][1]) == 1:
            continue
        out.append(p)
    return out


def lemma_a(p):
    """True if the only shift reads an affine function of u, possibly through one
    XOR/AND/OR with a constant."""
    sh = [n for n, (f, _) in enumerate(p) if f == 'shrk']
    if len(sh) != 1:
        return False
    kind = {0: 'aff'}
    for n, (f, args) in enumerate(p, 1):
        ks = [kind.get(a) for a in args]
        if f == 'aff' and ks == ['aff'] or f in ('lin1', 'lin2') and ks == ['aff', 'aff']:
            kind[n] = 'aff'
        elif f in ('xk', 'andk', 'ork') and ks == ['aff']:
            kind[n] = 'bitk'
        else:
            kind[n] = None
    (src,) = p[sh[0]][1]
    return kind[src] in ('aff', 'bitk')


def lemma_b(p):
    """True if every shift reads u itself or the result of a shift that does."""
    chain, seen = {0}, False
    for n, (f, args) in enumerate(p, 1):
        if f == 'shrk':
            if args[0] not in chain:
                return False
            chain.add(n)
            seen = True
    return seen


def xk_aff_index(p, k):
    """Index of the aff instruction if shift k reads aff(xk(u)), else None."""
    (j,) = p[k][1]
    if j and p[j - 1][0] == 'aff':
        (i,) = p[j - 1][1]
        if i and p[i - 1] == ('xk', (0,)):
            return j - 1
    return None


def lemma_a2(q, key=C5, seconds=3600, seed=0):
    """Two reads of one affine value: is there a' odd, b' (mod 2^20) and H with
    bit19(a'u + b') == y19(u ^ key) ^ H(u0, bit_q(a'u + b')) for every u < 2^20?"""
    import z3
    M20 = (1 << 20) - 1
    y19 = lambda u: (((4097 * (u ^ key) + g.C0) & M20) >> 19) & 1
    a, b = z3.BitVec('a', 20), z3.BitVec('b', 20)
    hs = [z3.BitVec(f'H{i}', 1) for i in range(4)]
    solver = z3.SolverFor('QF_BV')
    solver.add(z3.Extract(0, 0, a) == 1)
    rng = random.Random(q + 1000 * seed)
    pts = [rng.randrange(1 << 20) for _ in range(48)]
    t0 = time.monotonic()

    def add(u):
        w = a * u + b
        hv = z3.If(z3.Extract(q, q, w) == 1, hs[2 + (u & 1)], hs[u & 1])
        solver.add(z3.Extract(19, 19, w) == z3.BitVecVal(y19(u), 1) ^ hv)

    for u in pts:
        add(u)
    while True:
        solver.set(timeout=int(max(1.0, seconds - (time.monotonic() - t0)) * 1000))
        st = solver.check()
        if st == z3.unsat:
            return 'unsat'
        if st != z3.sat:
            return 'timeout'
        mdl = solver.model()
        av, bv = mdl[a].as_long(), mdl[b].as_long()
        hv = [mdl.eval(x, model_completion=True).as_long() for x in hs]
        bad = next((u for u in range(1 << 20)
                    if ((av * u + bv) & M20) >> 19 & 1 != y19(u) ^ hv[2 * (((av * u + bv) & M20) >> q & 1) + (u & 1)]),
                   None)
        if bad is None:
            return 'sat'
        add(bad)


RIGID_LAST = ('lin1', 'lin2', 'prod', 'mad_c', 'mad_b', 'mad_a')


def valuation_full(p, s):
    """2-adic order rule on the full domain (round-0 rule 2.5 with j up to 31).

    Flipping input bit j >= 20 moves the target by m*(T' - T), of 2-adic order exactly
    j - 19. With one shift i3 = w >> s read once by the last instruction, everything else
    moves by a multiple of 2^j and 0 < |di3| < 2^(32-s). If the last instruction is
    A + B*i3, v2(B * di3) = j - 19: j = 20 gives v2(B) <= 1, j = 31 gives v2(B) >= s - 19,
    so s >= 21 is impossible. If it is X ^ i3, X | i3 or X & i3, the lowest moved output
    bit (j - 19) must lie below bit 32 - s for j = 31, so s >= 20 is impossible."""
    sh = [n for n, (f, _) in enumerate(p) if f == 'shrk']
    if len(sh) != 1 or sh[0] != len(p) - 2:
        return False
    f, args = p[-1]
    if args.count(sh[0] + 1) != 1:
        return False
    return f in RIGID_LAST and s >= 21 or f in ('xor', 'or', 'and') and s >= 20


def exact_rule(p, amounts, key=C5):
    """Name of the exact rule closing one shift assignment, or None."""
    from codex import grammar4_round0_bit0_863 as b0
    if len(amounts) == 1 and valuation_full(p, next(iter(amounts.values()))):
        return 'valuation'
    if b0.rigid_any(p, amounts, key):  # inputs below 2^30 are part of the domain
        return 'rigid'
    return None


FACTS = Path(__file__).with_name('grammar4_head_c5_facts.json')
TWO_CONST = {(('xk', (0,)), ('lin2', (0, 1))): 'xk_lin2', (('ork', (0,)), ('lin2', (0, 1))): 'ork_lin2'}
MASK_SUBS = {(('andk', (0,)), ('aff', (1,))), (('ork', (0,)), ('aff', (1,)))}


def compute_facts(key=C5):
    """The key-dependent exact facts the case rules use (minutes: two 2^40 enumerations)."""
    from codex import grammar4_round0_bit0_863 as b0
    from codex import grammar4_round0_pinned_audit863 as audit
    return dict(key=key, mask=audit.mask_rule_holds(key),
                two_const={n: len(b0.two_const_solutions(b0.TWO_CONST[n][1], key)) for n in TWO_CONST.values()},
                two_read={q: lemma_a2(q, key) for q in range(19)})


def double_read(p):
    """Both shifts read bits of one value aff(u), side by side or nested."""
    sh = [n for n, (f, _) in enumerate(p) if f == 'shrk']
    if len(sh) != 2 or p[0] != ('aff', (0,)):
        return False
    src = {1}
    for n in sh:
        if p[n][1][0] not in src:
            return False
        src.add(n + 1)
    return True


def case_rule(p, amounts, facts, key=C5):
    """Exact rule closing one shift assignment of a template z3 could not decide, or None."""
    sh = [n for n, (f, _) in enumerate(p) if f == 'shrk']
    if len(sh) == 1:
        s = amounts[sh[0]]
        if s < 19:
            return 'dependency'  # the shift input is a T-function: its bit s ignores u19
        r = exact_rule(p, amounts, key)
        if r:
            return r
        sub = tuple(p[:p[sh[0]][1][0]])
        if s == 19 and sub in MASK_SUBS and facts['mask']:
            return 'mask'
        if s == 19 and sub in TWO_CONST and facts['two_const'][TWO_CONST[sub]] == 0:
            return 'two-const'
        return None
    if double_read(p) and all(v == 'unsat' for v in facts['two_read'].values()):
        return 'two-read'
    from codex import grammar4_round0_pinned_audit863 as audit
    return audit.multi_rule(p, amounts, key, False)


def lemma_a_solutions(key=C5):
    """Run the exact 20-bit check of lemma A; returns the number of (a', b', g) solutions."""
    with tempfile.TemporaryDirectory() as d:
        exe = Path(d) / 'lemma'
        subprocess.run(['clang', '-O2', '-o', str(exe), str(LEMMA_C)], check=True)
        out = subprocess.run([str(exe), hex(key)], check=True, capture_output=True, text=True).stdout
    return int(out.split('solutions=')[1])


_XK_AFF: dict = {}


def solve(p, seconds=60, amounts=None, key=C5, seed=863, low=False):
    """CEGIS for P(u) == m*T(u ^ key) + c; amounts optionally pins shift amounts.

    low: the single shift reads aff(xk(u)); pin the low bits to the exact bit-0 solutions.
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
    m, c = z3.BitVec('m', 32), z3.BitVec('c', 32)
    rng = random.Random(seed)
    pts = [0, 1, 2, 3, 1 << 12, 1 << 19, 1 << 20, 1 << 31, MASK, 0x80000001, 0x7FFFF] + \
          [rng.getrandbits(32) for _ in range(9)]
    solver = z3.SolverFor('QF_BV')
    solver.add(z3.Extract(0, 0, m) == 1)
    if low:
        (k, sk), = amounts.items()
        if sk < 19:
            return dict(result='unsat', how='dependency')
        if key not in _XK_AFF:
            from codex import grammar4_round0_bit0_863 as b0
            _XK_AFF[key] = b0.xk_affine_solutions(key)
        d, j = sk - 19, xk_aff_index(p, k)
        (i,) = p[j][1]
        solver.add(z3.Extract(d, 0, consts[j][0]) == 1 << d)
        solver.add(z3.Or([z3.And(z3.Extract(19, 0, consts[i - 1][0]) == kk,
                                 z3.Extract(d + 19, d, consts[j][0]) == mu,
                                 z3.Extract(d + 19, d, consts[j][1]) == nu) for kk, mu, nu in _XK_AFF[key]]))
    t0 = time.monotonic()

    def add(u):
        lhs = g.run_program(p, z3.BitVecVal(u, 32), consts, sem)
        solver.add(lhs == m * z3.BitVecVal(g.target(u ^ key), 32) + c)

    for u in pts:
        add(u)
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
        mv, cv = val(m), val(c)
        chk = random.Random(rounds)
        bad = next((u for u in (chk.getrandbits(32) for _ in range(4096))
                    if g.run_program(p, u, kv, csem) != (mv * g.target(u ^ key) + cv) & MASK), None)
        if bad is None:
            bad = g.verify_full(p, kv, mv, cv, key)
            if bad is None:
                return dict(result='sat', consts=kv, m=mv, c=cv, rounds=rounds,
                            seconds=round(time.monotonic() - t0, 2))
        add(bad)


_TS: list = []


def init_worker():
    _TS[:] = templates()
    if FACTS.exists():
        _FACTS.update(json.loads(FACTS.read_text()))


_FACTS: dict = {}


def job(args):
    idx, seconds, amounts, low = args
    p = _TS[idx] if _TS else templates()[idx]
    t0 = time.monotonic()
    rule = case_rule(p, amounts, _FACTS or json.loads(FACTS.read_text())) if amounts else None
    try:
        r = dict(result='unsat', rule=rule) if rule else solve(p, seconds, amounts, low=low)
    except Exception as exc:  # report, never hide
        r = dict(result='error', error=f'{type(exc).__name__}: {exc}'[:200])
    r.update(idx=idx, template=g.describe(p), wall=round(time.monotonic() - t0, 2))
    if amounts:
        r['amounts'] = {str(k): s for k, s in amounts.items()}
        r['low'] = low
    return r


RUNS = Path(__file__).with_name('grammar4_head_c5_runs.jsonl.gz')
AUDIT = Path(__file__).with_name('grammar4_head_c5_audit.json')


def audit(runs=RUNS, facts=None):
    """One row per template: lemma-a, lemma-b, solver (z3 UNSAT in the main pass) or
    split (every shift assignment closed by a pinned z3 run or a case rule)."""
    import collections
    import gzip
    facts = facts or json.loads(FACTS.read_text())
    main_unsat, pinned = set(), {}
    with gzip.open(runs, 'rt') as f:
        for line in f:
            r = json.loads(line)
            if r['pass'] == 'main':
                if r['result'] == 'unsat':
                    main_unsat.add(r['idx'])
                continue
            key = (r['idx'], tuple(sorted((int(k), v) for k, v in r['amounts'].items())))
            if r['result'] == 'unsat':
                pinned[key] = r.get('rule') or 'solver'
    rows, tally, cases = [], collections.Counter(), collections.Counter()
    for i, p in enumerate(templates()):
        row = dict(idx=i, template=g.describe(p))
        if lemma_a(p):
            row['method'] = 'lemma-a'
        elif lemma_b(p):
            row['method'] = 'lemma-b'
        elif i in main_unsat:
            row['method'] = 'solver'
        else:
            sh = [n for n, (f, _) in enumerate(p) if f == 'shrk']
            rules, left = collections.Counter(), []
            for combo in itertools.product(range(32), repeat=len(sh)):
                am = dict(zip(sh, combo))
                r = pinned.get((i, tuple(sorted(am.items())))) or case_rule(p, am, facts)
                rules[r or 'open'] += 1
                if r is None:
                    left.append(combo)
            row.update(method='open' if left else 'split', cases=dict(rules), open=left)
            cases.update(rules)
        tally[row['method']] += 1
        rows.append(row)
    return dict(key=C5, templates=dict(tally), cases=dict(cases), rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--count', action='store_true')
    ap.add_argument('--output', type=Path)
    ap.add_argument('--seconds', type=float, default=30)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--pin', type=Path, help='rerun the open templates of this file, one case per shift amount')
    ap.add_argument('--facts', action='store_true', help=f'compute the key facts and write {FACTS.name}')
    ap.add_argument('--audit', action='store_true', help=f'combine {RUNS.name} and the rules into {AUDIT.name}')
    args = ap.parse_args()
    ts = templates()
    if args.audit:
        res = audit()
        res['rows'] = [r for r in res['rows'] if r['method'] in ('split', 'open')]  # the rest is in templates
        AUDIT.write_text(json.dumps(res, indent=0) + '\n')
        print(json.dumps(dict(templates=res['templates'], cases=res['cases'])))
        return
    if args.facts:
        FACTS.write_text(json.dumps(compute_facts(), indent=1) + '\n')
        print(FACTS.read_text())
        return
    if args.count:
        print(len(ts), sum(map(lemma_a, ts)), sum(lemma_b(p) and not lemma_a(p) for p in ts))
        return
    if args.pin:
        rows = [json.loads(line) for line in args.pin.open()]
        todo = sorted(r['idx'] for r in rows if r['result'] != 'unsat')
        jobs = []
        for i in todo:
            sh = [n for n, (f, _) in enumerate(ts[i]) if f == 'shrk']
            if len(sh) == 1 and xk_aff_index(ts[i], sh[0]) is not None:
                jobs += [(i, args.seconds, {sh[0]: s}, True) for s in range(32)]
            else:
                jobs += [(i, args.seconds, dict(zip(sh, combo)), False)
                         for combo in itertools.product(range(32), repeat=len(sh))]
    else:
        jobs = [(i, args.seconds, None, False) for i, p in enumerate(ts) if not (lemma_a(p) or lemma_b(p))]
    print(json.dumps(dict(templates=len(ts), lemma_a=sum(map(lemma_a, ts)),
                          lemma_b=sum(lemma_b(p) and not lemma_a(p) for p in ts), jobs=len(jobs))), flush=True)
    with args.output.open('w') as out, \
            ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
        for f in as_completed([pool.submit(job, j) for j in jobs]):
            r = f.result()
            out.write(json.dumps(r) + '\n')
            out.flush()
            if r['result'] != 'unsat':
                print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
