"""Fiber and low-bit checks for the round-0 combos the pinned audit leaves open.

Both use the target's bit 0, x0 ^ y19 ^ const with y = 4097x + C0 (m is odd): it depends on input
bits 0..19 only, and flipping input bit 19 always flips it (y moves by 4097*2^19).

Low-bit projection. Every opcode but the right shift is a T-function, so the output mod 2^k needs
only the low k bits of its operands, and w >> s needs bits s..s+k-1 of w. The case is solved mod
2^k at sample points with m and c cut to k bits; that is implied by the full equation, so UNSAT
refutes the case. Samples come in pairs x, x ^ 2^j (j = 20..29), across which the target's bit 0
does not change.

Fiber filter, for a = aff(u) = mu*u + nu read by
  mad     a*(a >> p) + (a >> q): template 928, and 1082 / 1083, whose a >> s1 >> s2 is
          a >> (s1 + s2) (solved in the 928 form); bit 0 is (a0 & a_p) ^ a_q.
  window  (a + K*(a >> s1)) >> s2 (template 1377); bit 0 is bit s2 of a + K*(a >> s1).
Fix u mod 2^19 and let g = u >> 19 < 2048. Along those inputs the target's bit 0 is
const ^ (g & 1), and a = L + mu*2^19*g = L + 2^e*step*g with e = 19 + v2(mu) and step odd, which
depend on r = mu mod 2^13 only. The bits of a below e stay fixed along g: those below e - 19 are
bits of nu, those in [e - 19, e) take every pattern as u mod 2^19 varies (mu*u mod 2^e runs over
all multiples of 2^(e - 19)). The bits from e up are bits of W(g) = W(0) + step*g. FIBER_C keeps a
class (r, nu's fixed bits, K mod 2^(s2+1)) only if every pattern of the middle bits has some W(0)
with bit 0 ^ (g & 1) constant for g < 2048; z3 then solves each class with those bits pinned.
For mad, the residues with v2(mu) = q - 19 and nu0 = 0 (bit 0 is then a_q alone) are one class.

Window lemma. If every good W(0), whatever the pattern, is v mod 2^t, then bits e..e+t-1 of
mu*u + nu equal v for every u < 2^19: the progression stays in one block of 2^e mod 2^(e+t), so
its step as a signed residue has |step| * (2^19 - 1) < 2^e. A residue r that no such step matches
is refuted; otherwise mu mod 2^(e+t) and bits e..e+t-1 of nu are pinned as well.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
import json
from pathlib import Path
import random
import subprocess
import tempfile
import time

from codex import grammar3_head863 as g
from codex import grammar4_round0_863 as h
from codex import grammar4_round0_bit0_863 as b0
from codex import grammar4_round0_pinned863 as q

MAD = (('aff', (0,)), ('shrk', (1,)), ('shrk', (1,)), ('mad_c', (1, 2, 3)))
WINDOW = (('aff', (0,)), ('shrk', (1,)), ('lin2', (1, 2)), ('shrk', (3,)))
T_OPS = {'aff', 'lin1', 'lin2', 'prod', 'mad_a', 'mad_b', 'mad_c', 'xor', 'and', 'or', 'xk', 'andk',
         'ork', 'aab', 'aba', 'abb', 'bba', 'sq', 'sqx'}

FIBER_C = r"""/* shape 0: out0 = (a0 & a_p) ^ a_q.  shape 1: out0 = bit s2 of a + K*(a >> s1).
   Prints: r e nu_fixed_mask nu_fixed_bits k_low t v for every surviving class, where every good
   offset W(0), whatever the pattern, is v mod 2^t. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
static int SHAPE, P1, P2;
static uint32_t KK;
static inline uint32_t f0(uint32_t a) {
    if (SHAPE == 0) return ((a & (a >> P1)) ^ (a >> P2)) & 1;
    return ((a + KK * (a >> P1)) >> P2) & 1;
}
int main(int argc, char **argv) {
    SHAPE = atoi(argv[1]); P1 = atoi(argv[2]); P2 = atoi(argv[3]);
    int rel[40], nrel = 0;
    if (SHAPE == 0) { rel[nrel++] = 0; rel[nrel++] = P1; rel[nrel++] = P2; }
    else {
        for (int b = 0; b <= P2; b++) rel[nrel++] = b;
        for (int b = P1; b <= P1 + P2 && b < 32; b++) rel[nrel++] = b;
    }
    uint32_t kcount = SHAPE == 1 ? (1u << (P2 + 1)) : 1;
    for (uint32_t r = 1; r < 8192; r++) {
        int e = 19 + __builtin_ctz(r);
        uint32_t step = r >> (e - 19);
        int fixed[40], nf = 0, all[40], na = 0, hmax = -1;
        for (int i = 0; i < nrel; i++) {
            int b = rel[i];
            if (b < e - 19) fixed[nf++] = b;
            else if (b < e) all[na++] = b;
            else if (b - e > hmax) hmax = b - e;
        }
        if (hmax < 0) continue;                 /* out0 constant along g: cannot follow g & 1 */
        uint32_t smask = (1u << (hmax + 1)) - 1;
        for (uint32_t kl = 0; kl < kcount; kl++) {
            KK = kl;
            for (uint32_t fa = 0; fa < (1u << nf); fa++) {
                uint32_t base = 0, fmask = 0;
                for (int i = 0; i < nf; i++) { fmask |= 1u << fixed[i]; if ((fa >> i) & 1) base |= 1u << fixed[i]; }
                int ok = 1, any = 0;
                uint32_t first = 0, diff = 0;          /* low bits shared by every good offset */
                for (uint32_t pa = 0; pa < (1u << na) && ok; pa++) {
                    uint32_t lfix = base;
                    for (int i = 0; i < na; i++) if ((pa >> i) & 1) lfix |= 1u << all[i];
                    int found = 0;
                    for (uint32_t P = 0; P <= smask; P++) {
                        uint32_t w = P, ref = f0(lfix | (w << e)), good = 1;
                        for (uint32_t gg = 1; gg < 2048; gg++) {
                            w = (w + step) & smask;
                            if ((f0(lfix | (w << e)) ^ (gg & 1)) != ref) { good = 0; break; }
                        }
                        if (good) {
                            found = 1;
                            if (!any) { first = P; any = 1; }
                            diff |= P ^ first;
                        }
                    }
                    if (!found) ok = 0;
                }
                int t = diff ? __builtin_ctz(diff) : hmax + 1;
                if (ok) printf("%u %d %u %u %u %d %u\n", r, e, fmask, base, kl, t, first & ((1u << t) - 1));
            }
        }
    }
    return 0;
}
"""


def mad_pq(p, amounts):
    """(p, q) if the case computes a*(a >> p) + (a >> q) with a = aff(u), else None."""
    if p[0] != ('aff', (0,)):
        return None
    off = {1: 0}
    for n, (f, args) in enumerate(p[1:], 1):
        if f == 'shrk' and args[0] in off:
            off[n + 1] = off[args[0]] + amounts[n]
    f, args = p[-1]
    if f not in ('mad_c', 'mad_b') or args[0] != 1 or not {args[1], args[2]} <= off.keys():
        return None
    x, y = off[args[1]], off[args[2]]
    pq = (x, y) if f == 'mad_c' else (y, x)
    return pq if max(pq) < 32 else None


def window_steps(r, e, t):
    """Multipliers mod 2^(e+t) left by the window lemma, for mu = r mod 2^13.

    Every good offset is v mod 2^t, so bits e..e+t-1 of mu*u + nu equal v for all u < 2^19: the
    progression stays in one block of 2^e, so its step, read as a signed residue mod 2^(e+t), has
    |step| * (2^19 - 1) < 2^e.
    """
    n, bound = e + t, ((1 << e) - 1) // ((1 << 19) - 1)
    return [s % (1 << n) for s in range(-bound, bound + 1) if s and s % 8192 == r]


@functools.lru_cache(maxsize=None)
def fiber_classes(shape, x, y):
    """Classes kept by FIBER_C and the window lemma: dicts with mu = (bits, value), nu_mask and
    nu_bits (low bits of nu), nu_hi = (lo, width, value) or None, k (K mod 2^(y+1)) for shape 1;
    or {'mass': d}. Classes the window lemma empties are counted in 'window'."""
    with tempfile.TemporaryDirectory() as d:
        out = subprocess.run([str(b0._build(FIBER_C, 'fiber', d)), str(shape), str(x), str(y)],
                             check=True, capture_output=True, text=True).stdout.split()
    rows = [tuple(map(int, out[i:i + 7])) for i in range(0, len(out), 7)]
    mass, groups, emptied = [], {}, 0
    for r, e, fmask, fbits, k, t, v in rows:
        if shape == 0 and e == y and fmask == 1 and fbits == 0:   # v2(mu) = q - 19, nu0 = 0
            mass.append(r)
            continue
        refined = [((e + t, s), (e, t, v)) for s in window_steps(r, e, t)] if t else [((13, r), None)]
        emptied += not refined
        for mu, nu_hi in refined:
            groups.setdefault((mu, nu_hi, fmask, k if shape == 1 else None), set()).add(fbits)
    classes = []
    for (mu, nu_hi, fmask, k), bits in sorted(groups.items(), key=lambda t: (t[0][0], t[0][2], t[0][3] or 0)):
        if len(bits) == 1 << bin(fmask).count('1'):     # every value of nu's fixed bits survives
            classes.append(dict(mu=mu, nu_hi=nu_hi, nu_mask=0, nu_bits=0, k=k))
        else:
            classes += [dict(mu=mu, nu_hi=nu_hi, nu_mask=fmask, nu_bits=b, k=k) for b in sorted(bits)]
    if mass:
        classes.append(dict(mass=y - 19))
    return dict(kept=len(rows), window=emptied, classes=tuple(classes))


def class_pins(cls, k_at=None, k_width=0):
    """solve_pinned pins for one class: mu = consts[0][0], nu = consts[0][1], K = consts[k_at][0]."""
    if 'mass' in cls:
        d = cls['mass']
        return ((0, 0, 0, d + 1, 1 << d), (0, 1, 0, 1, 0))
    pins = [(0, 0, 0, *cls['mu'])]
    pins += [(0, 1, b, 1, cls['nu_bits'] >> b & 1) for b in range(32) if cls['nu_mask'] >> b & 1]
    if cls['nu_hi']:
        pins.append((0, 1, *cls['nu_hi']))
    if cls['k'] is not None:
        pins.append((k_at, 0, 0, k_width, cls['k']))
    return tuple(pins)


def projection_widths(p, amounts, k):
    """Low bits of each value (0 = u) that the output mod 2^k depends on."""
    need = [0] * (len(p) + 1)
    need[-1] = k
    for n in range(len(p), 0, -1):
        f, args = p[n - 1]
        if need[n]:
            if f != 'shrk' and f not in T_OPS:
                raise ValueError(f)
            for a in args:
                need[a] = max(need[a], min(32, amounts[n - 1] + need[n]) if f == 'shrk' else need[n])
    return need


def run_projected(p, amounts, u, consts, need, sem):
    """The output mod 2^need[-1] from truncated values (z3 bit-vectors)."""
    import z3
    cut = lambda x, w: z3.Extract(w - 1, 0, x) if x.size() > w else x
    vals = [z3.BitVecVal(u & ((1 << need[0]) - 1), need[0]) if need[0] else None]
    for n, (f, args) in enumerate(p, 1):
        w = need[n]
        if not w:
            vals.append(None)
        elif f == 'shrk':
            s, hi = amounts[n - 1], min(31, amounts[n - 1] + w - 1)
            v = z3.Extract(hi, s, vals[args[0]]) if s <= hi else z3.BitVecVal(0, w)
            vals.append(z3.ZeroExt(w - v.size(), v) if v.size() < w else v)
        else:
            vals.append(sem[f](*[cut(vals[a], w) for a in args], [cut(c, w) for c in consts[n - 1]]))
    return vals[-1]


def solve_projected(p, root, amounts, k, seconds=120, npts=400, seed=1):
    """Solve the case mod 2^k; 'unsat' refutes it."""
    import z3
    sem, csem = g.ops(z3), g.ops(None)
    need = projection_widths(p, amounts, k)
    consts = [[] if f == 'shrk' else [z3.BitVec(f'k{n}_{j}', 32) for j in range(g.NCONST.get(f, 0))]
              for n, (f, _) in enumerate(p)]
    m, c = z3.BitVec('m', k), z3.BitVec('c', k)
    solver = z3.SolverFor('QF_BV')
    solver.add(z3.Extract(0, 0, m) == 1)
    mask = (1 << k) - 1
    add = lambda x: solver.add(run_projected(p, amounts, x ^ root, consts, need, sem) ==
                               m * z3.BitVecVal(g.target(x) & mask, k) + c)
    rng = random.Random(seed)
    for _ in range(npts // 2):
        x = rng.randrange(g.D30)
        add(x)
        add(x ^ (1 << rng.randrange(20, 30)))
    t0, rounds = time.monotonic(), 0
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
        kv = [[amounts[n]] if f == 'shrk' else [val(x) for x in consts[n]] for n, (f, _) in enumerate(p)]
        mv, cv = val(m), val(c)
        chk = random.Random(rounds)
        bad = next((x for x in (chk.randrange(g.D30) for _ in range(20000))
                    if (g.run_program(p, x ^ root, kv, csem) - mv * g.target(x) - cv) & mask), None)
        if bad is None:   # the low bits fit: no refutation at this k
            return dict(result='sat-mod', rounds=rounds)
        add(bad)


def job(args):
    kind, p, amounts, extra, seconds = args
    t0 = time.monotonic()
    try:
        if kind == 'project':
            runs = []
            for k in (1, 2):
                r = solve_projected(p, h.ROOTS[0], amounts, k, seconds)
                runs.append(dict(k=k, **r))
                if r['result'] != 'sat-mod':
                    break
            r = dict(result=runs[-1]['result'], runs=runs)
        else:
            r = q.solve_pinned(p, h.ROOTS[0], amounts, seconds, pins=extra)
    except Exception as exc:  # report, never hide
        r = dict(result='error', error=f'{type(exc).__name__}: {exc}'[:200])
    r['wall'] = round(time.monotonic() - t0, 2)
    return r


def plan(cases, seconds):
    """{case: (method, [job key], filter summary)} and {job key: job args}."""
    ts = h.templates4()
    jobs, out = {}, {}
    for idx, am in cases:
        p = ts[idx]
        amounts = dict(zip(q.shifts(p), am))
        pq = mad_pq(p, amounts)
        keys = []
        if pq:
            fc = fiber_classes(0, *pq)
            for cls in fc['classes']:
                key = json.dumps(['mad', pq, cls], sort_keys=True)
                jobs[key] = ('solve', MAD, {1: pq[0], 2: pq[1]}, class_pins(cls), seconds)
                keys.append(key)
            out[(idx, am)] = ('fiber', keys, dict(shape='mad', pq=pq, kept=fc['kept'], window=fc['window']))
        elif tuple(p) == WINDOW:
            s1, s2 = amounts[1], amounts[3]
            fc = fiber_classes(1, s1, s2)
            for cls in fc['classes']:
                key = json.dumps(['window', idx, am, cls], sort_keys=True)
                jobs[key] = ('solve', p, amounts, class_pins(cls, 2, s2 + 1), seconds)
                keys.append(key)
            out[(idx, am)] = ('fiber', keys, dict(shape='window', kept=fc['kept'], window=fc['window']))
        else:
            key = json.dumps(['project', idx, am])
            jobs[key] = ('project', p, amounts, None, seconds)
            out[(idx, am)] = ('projection', [key], None)
    return out, jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--combos', type=Path, required=True, help='open combos, as the audit --open-todo writes')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seconds', type=float, default=300)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    cases = [(int(i), tuple(c['amounts'] if isinstance(c, dict) else c))
             for i, combos in json.loads(args.combos.read_text()).items() for c in combos]
    cases_plan, jobs = plan(cases, args.seconds)
    print(json.dumps(dict(cases=len(cases), jobs=len(jobs))), flush=True)
    done = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(job, a): key for key, a in jobs.items()}
        for f in as_completed(futs):
            done[futs[f]] = f.result()
            if done[futs[f]]['result'] == 'sat':
                raise SystemExit(f'SAT: {futs[f]} {done[futs[f]]}')
    ts = h.templates4()
    with args.output.open('w') as out:
        for (idx, am), (method, keys, summary) in sorted(cases_plan.items()):
            runs = [dict(job=json.loads(k), **done[k]) for k in keys]
            ok = all(r['result'] == 'unsat' for r in runs)
            row = dict(result='unsat' if ok else 'timeout', method=method, idx=idx,
                       amounts={str(n): s for n, s in zip(q.shifts(ts[idx]), am)},
                       template=g.describe(ts[idx]), filter=summary, runs=runs)
            out.write(json.dumps(row) + '\n')
            print(json.dumps(dict(idx=idx, amounts=am, result=row['result'], method=method,
                                  jobs=len(runs))), flush=True)


if __name__ == '__main__':
    main()
