"""2-adic arguments for the full-domain division templates still open after
grammar3_division_audit863.

Full domain: P(u) = m*T(u ^ e) + c for every u < 2^32, m odd, e one of the four
keys. Flipping input bit j (20..31) flips bit j of x = u ^ e, and the target
moves by m*(T' - T), whose 2-adic valuation is exactly j - 19 (sampled in the
tests). The templates read i1 in {aff(u), xk(u)}, i2 = OP(i1), i3 = combine of u,
i1, i2; every combine below moves by a multiple of 2^j when only u and i1 move by
multiples of 2^j.

even multiplier  i1 = aff(u) with an even multiplier does not move when u31 flips,
                 nor does i2, so the output moves by a multiple of 2^31, not
                 valuation 12. Hence i1 is a bijection of u (odd multiplier, or XOR
                 with a constant) and flipping u31 maps i1 to i1 ^ 2^31.
zero-divisor     OP divides by i1 (K // i1, K % i1, cdiv(K, i1)): i1 = 0 for some u,
                 and the simulator raises.
For the other rules the combine is A + B*i2 with A, B T-functions or constants
(lin1, lin2, prod, mad_c, mad_a, aab, aba, abb), or X ^ i2 with X in {u, i1}.
bijective-coef   B is i1 or i1 + 1: some u has v2(B) >= 2, so under the bit-20 flip
                 every term of the output's move has valuation >= 2, not 1.
For constant B the bit-20 flip gives v2(B) <= 1, and the bit-31 flip needs the move
of i2 to have valuation 12 - v2(B) (for X ^ i2 the output's lowest changed bit is
that of i2, so valuation 12) at every u. With q = 2^31 // K, r = 2^31 mod K and
rho = i1 mod K over i1 < 2^31 (all residues when K <= 2^31):
div-valuation    i1 // K moves by q + [rho >= K - r]: both values occur unless r = 0,
                 and consecutive integers differ in valuation. K = 2^k makes i2 a
                 right shift: that template of the base grammar is UNSAT (looked up).
                 K > 2^31: the move is 0 or 1.
mod-valuation    i1 % K moves by r or r - K, both occurring unless r = 0; equal
                 valuations v would make v2(K) > v, but 2^v2(K) divides r. r = 0: no
                 move. K > 2^31: rho = 0 moves by 2^31.
Combines of i1 and i2 alone (i1 a bijection as above):
shift-le         i1 >> i2 <= i1 pointwise; a bijection below the identity is the identity,
                 which is affine in u while T is not.
and-image        i1 & (i1 % K) < K <= 2^32 - 1: too few values for a bijection.
or-collision     i1 | (i1 // K): K even gives P(K) = P(K + 1); K odd >= 3 an even x in
                 [K + 1, 2K - 2] gives P(x) = P(x + 1) (x // K = 1 is odd); K >= 2^31 an even
                 x >= K does the same; K = 1 or 2^32 - 1 leaves P = i1. i1 | cdiv(i1, K):
                 P(2) = P(3) = 3 for every K >= 2, and K = 1 leaves P = i1.
cdiv-valuation   cdiv(i1, K) moves by q + [rho = 0 or rho > K - r], two values unless
                 K = 2^k, where it moves by exactly 2^(31-k): k = 19 + v2(B) (X ^ i2:
                 k = 19). K > 2^31: moves of at most 1. The K = 2^19, 2^20 cases are
                 solved with K pinned (cdiv-pinned).
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import time

from codex import grammar3_head863 as g

AUDIT = Path(__file__).with_name('grammar3_division_audit863.json')
BASE_RESULTS = Path(__file__).with_name('grammar3_head863_results.jsonl.gz')

KDIV = {'kdiv', 'kmod', 'kcdiv'}
COMBINE = {   # combine of (u, i1, i2) -> (A, B) description; B in {'1', 'K', 'i1', 'i1+1'}
    ('lin1', (0, 2)): '1', ('lin1', (1, 2)): '1', ('lin2', (0, 2)): 'K', ('lin2', (1, 2)): 'K',
    ('prod', (1, 2)): 'i1', ('mad_c', (0, 1, 2)): '1', ('mad_a', (0, 1, 2)): 'i1',
    ('aab', (1, 2)): '1', ('aab', (0, 2)): '1', ('aba', (1, 2)): 'i1', ('abb', (1, 2)): 'i1+1',
    ('xor', (1, 2)): 'xor', ('xor', (0, 2)): 'xor',
}
FIRST = {('aff', (0,)), ('xk', (0,))}


def open_items():
    rows = json.loads(AUDIT.read_text())['rest']
    return [r['idx'] for r in rows if r['resolved'] == 'open']


def base_shift_unsat():
    """Canonical keys of base-grammar templates whose run was UNSAT."""
    out = set()
    ts = g.templates()
    for line in gzip.open(BASE_RESULTS, 'rt'):
        r = json.loads(line)
        if r['result'] == 'unsat':
            out.add(g.canonical(ts[r['idx']]))
    return out


def classify(p, base_unsat):
    """(rule, pinned divisors to solve) for a full-grammar template, or (None, ())."""
    if len(p) != 3 or p[0] not in FIRST or p[1][1] != (1,):
        return None, ()
    op = p[1][0]
    kind = COMBINE.get(p[2])
    # for zero-divisor any combine of i1, i2 alone also qualifies: it does not move at all
    # when i1 and i2 do not
    if op in KDIV and (kind is not None or set(p[2][1]) <= {1, 2}):
        return 'zero-divisor', ()
    if kind in ('i1', 'i1+1'):
        return 'bijective-coef', ()
    if p[2] == ('shr_ab', (1, 2)):
        return 'shift-le', ()
    if p[2] == ('and', (1, 2)) and op == 'modk':
        return 'and-image', ()
    if p[2] == ('or', (1, 2)) and op in ('divk', 'cdivk'):
        return 'or-collision', ()
    if kind is None:
        return None, ()
    if op == 'modk':
        return 'mod-valuation', ()
    if op == 'divk':
        shifted = [p[0], ('shrk', (1,)), p[2]]
        if g.canonical(shifted) in base_unsat:
            return 'div-valuation', ()
        return None, ()
    if op == 'cdivk':
        return 'cdiv-valuation', ((1 << 19,) if kind in ('1', 'xor') else (1 << 19, 1 << 20))
    return None, ()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--seconds', type=float, default=300)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    ts = g.templates(True)
    base_unsat = base_shift_unsat()
    rows, tally = [], {}
    for idx in open_items():
        p = ts[idx]
        rule, pins = classify(p, base_unsat)
        runs = []
        for k in pins:
            t0 = time.monotonic()
            r = g.solve(p, args.seconds, fix={'k1_0': k})
            runs.append(dict(divisor=k, result=r['result'], seconds=round(time.monotonic() - t0, 2)))
            print(json.dumps(dict(idx=idx, divisor=k, result=r['result'])), flush=True)
            if r['result'] == 'sat':
                raise SystemExit(f'SAT: {idx} {r}')
        if pins and not all(x['result'] == 'unsat' for x in runs):
            rule = None
        tally[rule or 'open'] = tally.get(rule or 'open', 0) + 1
        rows.append(dict(idx=idx, template=g.describe(p), rule=rule, pinned_runs=runs))
    summary = dict(items=len(rows), by_rule=tally)
    print(json.dumps(summary))
    args.output.write_text(json.dumps(dict(summary=summary, items=rows), indent=1) + '\n')


if __name__ == '__main__':
    main()
