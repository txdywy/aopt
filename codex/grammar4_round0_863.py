"""Round 0 with the node XOR merged in: four instructions from the raw input value.

Round 0 computes x = v XOR R (input value XOR root, both below 2^30) and then the
first hash segment: five instructions from v to the middle hash F. The kernel
learns R only at run time, but it can derive any constants from R once during
setup, and F's MADs absorb any odd affine image of the result. So a fixed
skeleton P with four instructions and R-dependent constants such that

    P(v) == m*T(v XOR R) + c     for every v < 2^30,   m odd,

would save one VALU op per element in round 0, 256 lanes, against the 53 that
863 needs. A skeleton has to work for every root value, so failing for a single
random R excludes it.

Grammar: the 14 core forms of grammar3_head863 (MAD; x^K, x&K, x|K; x>>s;
K*a+b, K*b+a, a*b+K, a^b, a&b, a|b; three-operand MAD) with any SSA wiring.
That includes "node XOR with any key, then a three-instruction head", the
case the three-instruction search left open. Squares, runtime shift amounts,
division and comparisons are left out: they never mattered in the
three-instruction searches, and with them the four-instruction grammar has
over a million templates.

Sound pruning: every instruction is used; there is a right shift (T's bit 0
depends on input bit 19); the last instruction is not a single-operand MAD
(absorbed by the output encoding); a first-instruction shift that is the only
use of v is excluded, because the domain is closed under v ^ 1, so s >= 1
collides and s = 0 is the twin template with an identity MAD.

Bit-0 lemma. Every core form except the right shift is a T-function, whose
bit 0 depends only on the operands' bits 0. With a single shift w >> s, the
output's bit 0 is therefore a Boolean function of v0 and w_s. The target's
bit 0 is v0 ^ y19 ^ const (m is odd), where y19 is bit 19 of 4097(v^R)+C0,
and y19 is not a function of v0. Hence bit_s(w(v)) = y19(v) ^ g(v0) for every
v, where w is the sub-program feeding the shift. If no constants satisfy
that, every template whose only shift reads such a w is excluded.
"""
from __future__ import annotations

import argparse
import functools
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
from pathlib import Path
import random
import time

from codex import grammar3_head863 as g

U4 = ('aff', 'xk', 'andk', 'ork', 'shrk')
B4 = ('lin1', 'lin2', 'prod', 'xor', 'and', 'or')
T4 = ('mad_c', 'mad_b', 'mad_a')
ROOTS = [random.Random(863).randrange(g.D30), random.Random(864).randrange(g.D30)]


def templates4():
    out, seen = [], set()
    unary = lambda ids: [(f, (i,)) for i in ids for f in U4]
    binary = lambda prs: [(f, pr) for pr in prs for f in B4]
    ternary = lambda trs: [(f, tr) for tr in trs for f in T4]
    i1s = unary([0])
    i2s = unary([0, 1]) + binary([(0, 1)])
    i3s = unary([0, 1, 2]) + binary([(0, 1), (0, 2), (1, 2)]) + ternary([(0, 1, 2)])
    i4s = unary([3]) + binary([(0, 3), (1, 3), (2, 3)]) + ternary([(0, 1, 3), (0, 2, 3), (1, 2, 3)])
    for p in itertools.product(i1s, i2s, i3s, i4s):
        if not any(f == 'shrk' for f, _ in p):
            continue
        used = {a for _, args in p[1:] for a in args}
        if not {1, 2} <= used or p[3][0] == 'aff':
            continue
        if p[0][0] == 'shrk' and 0 not in used:
            continue
        key = g.canonical(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def solve_domain_cnf(p, root, engine='cadical195', seed=863):
    """CaDiCaL version of g.solve_domain (no time limit: run it in a killable process)."""
    import z3
    from pysat.solvers import Solver
    sem = g.ops(z3)
    named, consts = {}, []
    for n, (f, _) in enumerate(p):
        ks = []
        for j in range(g.NCONST.get(f, 0)):
            w = 5 if f == 'shrk' else 32
            bv, bits = g._named(z3, f'k{n}_{j}', w)
            named[f'k{n}_{j}'] = bits
            ks.append(z3.ZeroExt(27, bv) if w == 5 else bv)
        consts.append(ks)
    m, named['m'] = g._named(z3, 'm', 32)
    c, named['c'] = g._named(z3, 'c', 32)
    ids, nxt = {}, [0]

    def var():
        nxt[0] += 1
        return nxt[0]

    for bits in named.values():
        for b in bits:
            ids[b.get_id()] = var()
    tactic = z3.Then('simplify', 'bit-blast', 'tseitin-cnf')

    def clauses(x):
        lhs = g.run_program(p, z3.BitVecVal(x ^ root, 32), consts, sem)
        goal = z3.Goal()
        goal.add(lhs == m * z3.BitVecVal(g.target(x), 32) + c)
        local, out = {}, []

        def lit(e):
            neg = z3.is_not(e)
            a = e.arg(0) if neg else e
            k = a.get_id()
            v = ids.get(k) or local.get(k)
            if v is None:
                v = local[k] = var()
            return -v if neg else v

        for cl in tactic(goal)[0]:
            if z3.is_false(cl):
                return None
            if not z3.is_true(cl):
                out.append([lit(e) for e in cl.children()] if z3.is_or(cl) else [lit(cl)])
        return out

    rng = random.Random(seed)
    pts = [0, 1, 2, 3, 1 << 12, 1 << 19, 1 << 20, 1 << 29, g.D30 - 1, 0x2AAAAAAA, 0x7FFFF] + \
          [rng.randrange(g.D30) for _ in range(9)]
    t0 = time.monotonic()
    sol = Solver(name=engine)
    sol.add_clause([ids[named['m'][0].get_id()]])
    try:
        for x in pts:
            cl = clauses(x)
            if cl is None:
                return dict(result='unsat', how='simplify')
            sol.append_formula(cl)
        rounds = 0
        while True:
            rounds += 1
            if not sol.solve():
                return dict(result='unsat', engine=engine, rounds=rounds, samples=len(pts),
                            seconds=round(time.monotonic() - t0, 2))
            model = set(l for l in sol.get_model() if l > 0)
            val = lambda name: sum(1 << i for i, b in enumerate(named[name]) if ids[b.get_id()] in model)
            kv = [[val(f'k{n}_{j}') for j in range(g.NCONST.get(f, 0))] for n, (f, _) in enumerate(p)]
            mv, cv = val('m'), val('c')
            csem = g.ops(None)
            chk = random.Random(rounds)
            bad = next((x for x in (chk.randrange(g.D30) for _ in range(4096))
                        if g.run_program(p, x ^ root, kv, csem) != (mv * g.target(x) + cv) & g.MASK), None)
            if bad is None:
                bad = g.verify_full(p, kv, mv, cv, root, domain=g.D30)
                if bad is None:
                    return dict(result='sat', consts=kv, m=mv, c=cv, rounds=rounds,
                                seconds=round(time.monotonic() - t0, 2))
            pts.append(bad)
            cl = clauses(bad)
            if cl is None:
                return dict(result='unsat', how='simplify')
            sol.append_formula(cl)
    finally:
        sol.delete()


def shift_input(p, k):
    """The sub-program computing the operand of shift instruction k, re-indexed."""
    need = set()

    def visit(val):
        if val and val not in need:
            need.add(val)
            for a in p[val - 1][1]:
                visit(a)

    (j,) = p[k][1]
    visit(j)
    remap, sub = {0: 0}, []
    for n, val in enumerate(sorted(need), 1):
        remap[val] = n
        f, args = p[val - 1]
        sub.append((f, tuple(remap[a] for a in args)))
    return sub


def bit0_width(sub, root, sh, seconds=60, nsamples=96, seed=865):
    """One shift amount: bit sh of w(v) == y19(v) ^ g(v0), solved at width sh + 1.

    w is built from T-functions only, so its bit sh depends on v and the
    constants modulo 2^(sh+1); everything is truncated to that width.
    """
    import z3
    width = sh + 1
    mask = (1 << width) - 1
    sem = g.ops(z3)
    consts = [[z3.BitVec(f'k{n}_{j}', width) for j in range(g.NCONST.get(f, 0))]
              for n, (f, _) in enumerate(sub)]
    g0, g1 = z3.Bool('g0'), z3.Bool('g1')
    solver = z3.SolverFor('QF_BV')
    t0 = time.monotonic()

    def y19(v):
        return ((4097 * (v ^ root) + g.C0) & g.MASK) >> 19 & 1

    def add(v):
        w = g.run_program(sub, z3.BitVecVal(v & mask, width), consts, sem)
        gv = g1 if v & 1 else g0
        solver.add((z3.Extract(sh, sh, w) == 1) == (z3.Not(gv) if y19(v) else gv))

    rng = random.Random(seed)
    pts = [0, 1, 1 << 19, (1 << 19) | 1] + [rng.randrange(g.D30) for _ in range(nsamples - 4)]
    # bits 20..sh must not matter: pair samples that differ in one of them
    pts += [v ^ (1 << j) for v, j in zip(pts[4:], range(20, min(sh, 29) + 1))]
    for v in pts:
        add(v)
    csem = g.ops(None)
    for rounds in range(1, 200):
        left = seconds - (time.monotonic() - t0)
        if left <= 0:
            return 'timeout'
        solver.set(timeout=int(left * 1000))
        st = solver.check()
        if st == z3.unsat:
            return 'unsat'
        if st != z3.sat:
            return 'timeout'
        mdl = solver.model()
        kv = [[mdl.eval(k, model_completion=True).as_long() for k in ks] for ks in consts]
        gg = [z3.is_true(mdl.eval(x, model_completion=True)) for x in (g0, g1)]
        chk = random.Random(rounds)
        bad = None
        for _ in range(20000):
            v = chk.randrange(g.D30)
            if (g.run_program(sub, v, kv, csem) >> sh & 1) != (y19(v) ^ gg[v & 1]):
                bad = v
                break
        if bad is None:
            return 'survives'  # holds on 20,000 random inputs: the full solver decides
        add(bad)
    return 'timeout'


def sub_class(sub):
    """'bitwise' if the shift input is built from u by XOR/AND/OR (with constants or each
    other), 'affine' if it is an affine map of u (MAD chains, a*K+b of affine values),
    possibly followed by one XOR/AND/OR with a constant; else None.

    Bit s of a bitwise value is a function of v_s alone. Bit s of (a op K) is bit s of
    a or a constant. Both reduce the lemma to the exact checks below.
    """
    kind = ['bitwise']  # u itself
    for f, args in sub:
        ks = [kind[a] for a in args]
        if f in ('xk', 'andk', 'ork', 'xor', 'and', 'or') and all(k == 'bitwise' for k in ks):
            kind.append('bitwise')
        elif f in ('aff', 'lin1', 'lin2') and all(k in ('affine', 'bitwise') for k in ks) and \
                all(k == 'affine' or a == 0 for k, a in zip(ks, args)):
            kind.append('affine')
        else:
            kind.append(None)
    last = kind[-1]
    if sub and last is None and sub[-1][0] in ('xk', 'andk', 'ork'):
        last = kind[sub[-1][1][0]]  # trailing bitwise op with a constant
    return last


def y19_of(root):
    return lambda v: ((4097 * (v ^ root) + g.C0) & g.MASK) >> 19 & 1


@functools.lru_cache(maxsize=None)
def bitwise_refuted(root):
    """For every s, a pair of inputs agreeing on bits 0, s, 19 but not on y19.

    Then y19 is not a function of (v0, v_s), so bit s of a bitwise value cannot
    equal y19 ^ g(v0). Returns the witnesses.
    """
    y19 = y19_of(root)
    rng = random.Random(866)
    wit = {}
    for sh in range(32):
        keep = (1 | 1 << sh | 1 << 19) & (g.D30 - 1)
        while sh not in wit:
            v1 = rng.randrange(g.D30)
            v2 = (v1 & keep) | (rng.randrange(g.D30) & ~keep)
            if y19(v1) != y19(v2):
                wit[sh] = (v1, v2)
    return wit


AFFINE_C = r"""
#include <stdint.h>
#include <stdio.h>
#define N (1u << 20)
#define H (1u << 19)
static uint32_t R;
static int y19(uint32_t v) { return ((4097u * (v ^ R) + %(C0)uu) >> 19) & 1; }
int main(void) {
  R = %(R)uu;
  static uint32_t vs[256]; uint32_t x = 12345;
  for (int i = 0; i < 256; i++) { x = x * 1103515245u + 12345u; vs[i] = (x >> 7) & (N - 1); }
  long found = 0;
  for (uint32_t mu = 1; mu < N; mu += 2)
    for (int g = 0; g < 4; g++) {
      /* feasible nu: arc [lo, lo + len) on Z/2^20 */
      uint32_t lo = 0, len = N;
      for (int i = 0; i < 256 && len; i++) {
        uint32_t v = vs[i];
        int t = y19(v) ^ ((g >> (v & 1)) & 1);
        uint32_t a = (t * H - mu * v) & (N - 1);   /* arc [a, a + H) */
        if (len == N) { lo = a; len = H; continue; }
        uint32_t d = (a - lo) & (N - 1);          /* offset of a from lo */
        if (d < len) { len -= d; lo = a; if (len > H) len = H; }
        else if (((lo - a) & (N - 1)) < H) { uint32_t e = H - ((lo - a) & (N - 1)); if (e < len) len = e; }
        else len = 0;
      }
      for (uint32_t k = 0; k < len; k++) {      /* survivors: check every v */
        uint32_t nu = (lo + k) & (N - 1), v, ok = 1;
        for (v = 0; v < N && ok; v++)
          ok = ((((mu * v + nu) & (N - 1)) >> 19) & 1) == (uint32_t)(y19(v) ^ ((g >> (v & 1)) & 1));
        if (ok) { printf("%%u %%u %%d\n", mu, nu, g); found++; }
      }
    }
  printf("done %%ld\n", found);
  return 0;
}
"""


@functools.lru_cache(maxsize=None)
def affine_top_refuted(root):
    """Shift amounts 30 and 31 for an affine input with odd multiplier mu.

    y19 depends only on r = v mod 2^20, and v + j*2^20 stays in the domain for
    j < 1024, so the output bit is constant on each such fiber. Along a fiber
    the top 12 bits of mu*v + beta walk by mu mod 4096. Bit 31 constant forces
    a walk of 1023 steps to stay in a half circle, so mu = +-1 mod 4096; the top
    bits of mu*r + beta then move by (mu >> 20) or (mu >> 20) + 1 per step in r,
    and staying in the allowed arc leaves only mu = +-1. Bit 30 constant forces
    every fiber to start on one residue mod 1024, which the carry out of the low
    20 bits breaks as r varies, so s' = 30 is impossible. For mu = +-1 the bit
    is a function of one carry (or borrow) threshold in r: a step function of r
    with at most one step. Returns True if y19 ^ g(v0) is never such a function.
    """
    y19 = y19_of(root)
    f = [y19(r) for r in range(1 << 20)]
    for g0 in (0, 1):
        for g1 in (0, 1):
            vals = [f[r] ^ (g1 if r & 1 else g0) for r in range(1 << 20)]
            if sum(vals[r] != vals[r + 1] for r in range((1 << 20) - 1)) <= 1:
                return False
    return True


@functools.lru_cache(maxsize=None)
def affine_refuted(root):
    """Exact check that no odd mu, nu (mod 2^20), g give bit19(mu*v+nu) == y19(v) ^ g(v0).

    Any affine shift input reduces to this: write the multiplier as 2^d * odd; bit s
    of 2^d*mu*v + nu is bit s-d of mu*v + (nu >> d); s-d < 19 ignores v19, s-d > 19
    depends on v_(s-d), which y19 does not; a zero multiplier is constant.
    Returns the list of solutions (empty means refuted).
    """
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        src, exe = Path(d) / 'aff.c', Path(d) / 'aff'
        src.write_text(AFFINE_C % dict(C0=g.C0, R=root))
        subprocess.run(['clang', '-O2', '-o', str(exe), str(src)], check=True)
        out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout.split('\n')
    assert out[-2].startswith('done') if out[-1] == '' else out[-1].startswith('done')
    return tuple(tuple(map(int, l.split())) for l in out if l and not l.startswith('done'))


def bit0_lemma(sub, root, seconds=60, exhaustive=False):
    """'unsat' if no constants and shift give bit_s(w(v)) == y19(v) ^ g(v0) on v < 2^30.

    Bitwise and affine shift inputs use the exact checks above. Otherwise s < 19
    is impossible outright (bit s of a T-function ignores input bit 19, y19 flips
    with it) and each s >= 19 is solved at width s + 1.
    """
    cls = sub_class(sub)
    if cls == 'bitwise':
        return 'unsat' if bitwise_refuted(root) else 'timeout'
    if cls == 'affine':
        return 'unsat' if not affine_refuted(root) and affine_top_refuted(root) else 'survives'
    if not exhaustive:
        return 'generic'
    worst = 'unsat'
    for sh in range(19, 32):
        r = bit0_width(sub, root, sh, seconds)
        if r == 'survives':
            return 'survives'
        if r == 'timeout':
            worst = 'timeout'
    return worst


def lemma_key(p):
    """Canonical shift-input sub-program of a single-shift template, else None."""
    shifts = [k for k, (f, _) in enumerate(p) if f == 'shrk']
    if len(shifts) != 1:
        return None
    sub = shift_input(p, shifts[0])
    return (g.canonical(sub) if sub else 'u'), sub


def lemma_job(args):
    key, sub, root = args
    t0 = time.monotonic()
    return key, bit0_lemma(sub, root), round(time.monotonic() - t0, 2)


def lemma_table(ts, root, workers, path):
    """Bit-0 lemma verdict per distinct shift input, cached in a JSON file."""
    table = json.loads(path.read_text()) if path.exists() else {}
    subs = {}
    for p in ts:
        lk = lemma_key(p)
        if lk:
            subs.setdefault(lk[0], lk[1])
    todo = [(k, sub, root) for k, sub in subs.items() if k not in table]
    # exact classes are cheap and shared; computing them in the parent fills the caches once
    for k, sub, _ in list(todo):
        if sub_class(sub) in ('bitwise', 'affine'):
            table[k] = bit0_lemma(sub, root)
    todo = [t for t in todo if t[0] not in table]
    print(json.dumps(dict(lemma_inputs=len(subs), todo=len(todo))), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, f in enumerate(as_completed([pool.submit(lemma_job, t) for t in todo]), 1):
            k, verdict, secs = f.result()
            table[k] = verdict
            if verdict != 'unsat' or n % 100 == 0:
                print(json.dumps(dict(done=n, key=k, verdict=verdict, seconds=secs)), flush=True)
            if n % 25 == 0:
                path.write_text(json.dumps(table, indent=0, sort_keys=True))
    path.write_text(json.dumps(table, indent=0, sort_keys=True))
    return table


def job(args):
    idx, p, seconds, root, lem = args
    t0 = time.monotonic()
    try:
        if lem == 'unsat':
            r = dict(result='unsat', how='bit0-lemma')
        else:
            r = g.solve_domain(p, seconds, root)
            if lem is not None:
                r['lemma'] = lem
    except Exception as exc:  # report, never hide
        r = dict(result='error', error=f'{type(exc).__name__}: {exc}'[:200])
    r.update(idx=idx, template=g.describe(p), root=root, wall=round(time.monotonic() - t0, 2))
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--count', action='store_true')
    ap.add_argument('--output', type=Path)
    ap.add_argument('--seconds', type=float, default=30)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--root', type=lambda s: int(s, 0), default=ROOTS[0])
    ap.add_argument('--only', type=str, default='')
    ap.add_argument('--skip-done', action='store_true')
    ap.add_argument('--one-cnf', type=int, default=None, help='CaDiCaL on one template, print JSON')
    ap.add_argument('--lemma', type=Path, default=None, help='bit-0 lemma table (JSON, computed if missing)')
    args = ap.parse_args()
    ts = templates4()
    if args.one_cnf is not None:
        t0 = time.monotonic()
        r = solve_domain_cnf(ts[args.one_cnf], args.root)
        r.update(idx=args.one_cnf, template=g.describe(ts[args.one_cnf]), root=args.root,
                 wall=round(time.monotonic() - t0, 2))
        print(json.dumps(r))
        return
    if args.count:
        print(len(ts), 'roots', [hex(r) for r in ROOTS])
        return
    idxs = [int(i) for i in args.only.split(',')] if args.only else list(range(len(ts)))
    done = set()
    if args.skip_done and args.output.exists():
        done = {json.loads(l)['idx'] for l in args.output.open()}
    table = lemma_table(ts, args.root, args.workers, args.lemma) if args.lemma else {}
    verdict = lambda p: table.get(lemma_key(p)[0]) if lemma_key(p) else None
    todo = [(i, ts[i], args.seconds, args.root, verdict(ts[i])) for i in idxs if i not in done]
    print(json.dumps(dict(templates=len(ts), todo=len(todo), root=args.root)), flush=True)
    with args.output.open('a' if args.skip_done else 'w') as out, \
            ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(job, t) for t in todo]):
            r = f.result()
            out.write(json.dumps(r) + '\n')
            out.flush()
            if r['result'] != 'unsat':
                print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
