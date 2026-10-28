"""Exact arguments for full-domain templates whose last instruction divides.

grammar3_head863 --full times out on many templates ending in P = a // b or
P = cdiv(a, b) with runtime a, b, because z3's divider circuits are slow. P has
to be a bijection of all 2^32 inputs (m is odd and T is a bijection), and b >= 1
everywhere (a zero divisor makes the simulator raise). Then P <= a pointwise,
and P >= 2^31 (cdiv: P > 2^31) forces b = 1 and P = a. The rules below use only
these facts; each holds for every choice of constants:

  input-numerator  a is the raw input u. P <= u pointwise and P is a bijection,
                   so P is the identity by induction from 0; T is not affine.
  single-source    a and b are i1 and i2 = g(i1). P depends on u only through i1,
                   so i1 is a bijection; if a = i1 the previous rule applies with
                   i1 for u, and if b = i1 it is zero for some input.
  injective-a      a is one instruction on u (i1, or i2 reading only u):
                   MAD(u,K1,K2) with odd K1 or u XOR K: a bijection, and
                   P <= a forces P = a, a single affine or XOR instruction.
  even-mad-a       a is MAD(u,K1,K2) with even K1: a hits at most 2^30 values
                   that are >= 2^31, but P needs 2^31 such values (cdiv: 2^31 - 1).
  small-image      a is u >> s (s >= 1), u & K (K != all ones), a square,
                   a shifted constant, u // K, K // u, u % K, K % u, cdiv by or of a
                   constant, or a comparison: a takes fewer than 2^31 - 1 values
                   above 2^31, or, for K = all ones, s = 0 or u // 1, the case is the input or
                   a constant, both covered above.
Whatever no rule settles is reported as open.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from codex.grammar3_head863 import OUTDIV, templates, describe

SMALL_IMAGE = {'shrk', 'andk', 'sq', 'sqx', 'kshl', 'kshr', 'sshl', 'sshr', 'divk', 'kdiv',
               'modk', 'kmod', 'cdivk', 'kcdiv', 'ltk', 'klt', 'eqk'}


def classify(p):
    f3, a3 = p[2]
    if f3 not in OUTDIV:
        return None
    num, den = (a3[i] for i in OUTDIV[f3])
    if num == 0:
        return 'input-numerator'
    if {num, den} == {1, 2} and p[1][1] == (1,):
        return 'single-source'
    if p[num - 1][1] == (0,):  # the numerator is one instruction applied to u
        f1 = p[num - 1][0]
        if f1 == 'xk':
            return 'injective-a'
        if f1 == 'aff':
            return 'injective-a|even-mad-a'
        if f1 in SMALL_IMAGE:
            return 'small-image'
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    rows = {}
    for path in args.inputs:
        opener = gzip.open if path.endswith('.gz') else open
        for line in opener(path, 'rt'):
            r = json.loads(line)
            if r['idx'] not in rows or r['result'] == 'unsat':
                rows[r['idx']] = r
    ts = templates(True)
    out = []
    for idx, r in sorted(rows.items()):
        if r['result'] == 'unsat':
            continue
        out.append(dict(idx=idx, template=describe(ts[idx]), result=r['result'],
                        resolved=classify(ts[idx]) or 'open'))
    summary = dict(queries=len(rows), unsat=sum(r['result'] == 'unsat' for r in rows.values()),
                   other=len(out), by_argument={})
    for r in out:
        summary['by_argument'][r['resolved']] = summary['by_argument'].get(r['resolved'], 0) + 1
    print(json.dumps(summary))
    if args.output:
        args.output.write_text(json.dumps(dict(summary=summary, rest=out), indent=1) + '\n')


if __name__ == '__main__':
    main()
