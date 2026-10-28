"""Exact arguments for round-0 four-instruction templates the solver did not decide.

The target's bit 0 is v0 ^ y19 ^ const, where y19 = bit 19 of 4097*(v^R) + C0.
For each bit j < 20 a single flip (v, v ^ 2^j) changes y19 (witnesses below), so
y19 depends on every bit 0..19 of v.

bit0-dependency: for every choice of the template's shift amounts, track which
input bits the output's bit 0 can depend on. Bitwise forms pass bit j through,
arithmetic forms (MAD, a*K+b, a*b+K) let bit j depend on bits 0..j of their
operands, a shift by s reads bit j+s, and input bits 30 and 31 are zero on the
domain. The sets over-approximate the true dependence. If a choice leaves out
some bit in 0..19, the flip witness for that bit agrees on the whole set but not
on y19, so that choice is impossible. A last instruction that shifts by s >= 3
leaves at most 2^29 values for 2^30 inputs and is impossible too. A template is
excluded when every choice is.

Retries with a longer limit override a timeout; the rest is reported as open.
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
from pathlib import Path

from codex import grammar3_head863 as g
from codex import grammar4_round0_863 as h

BITWISE = {'xk', 'andk', 'ork', 'xor', 'and', 'or'}


def flip_witnesses(root):
    """For each j < 20 an input v with y19(v) != y19(v ^ 2^j)."""
    y19 = h.y19_of(root)
    out = {}
    for j in range(20):
        out[j] = next(v for v in range(g.D30) if y19(v) != y19(v ^ (1 << j)))
    return out


def bit0_support(p, amounts):
    """Input bits that bit 0 of the output may depend on, or None if impossible."""
    deps = [[frozenset([j]) if j < 30 else frozenset() for j in range(32)]]
    for k, (f, args) in enumerate(p):
        if f == 'shrk':
            s, a = amounts[k], deps[args[0]]
            if k == len(p) - 1 and s >= 3:
                return None
            deps.append([a[j + s] if j + s < 32 else frozenset() for j in range(32)])
        elif f in BITWISE:
            deps.append([frozenset().union(*(deps[a][j] for a in args)) for j in range(32)])
        else:
            acc, cur = [], frozenset()
            for j in range(32):
                cur = cur.union(*(deps[a][j] for a in args))
                acc.append(cur)
            deps.append(acc)
    return deps[-1][0] | {0}


def dependency_excluded(p):
    shifts = [k for k, (f, _) in enumerate(p) if f == 'shrk']
    need = frozenset(range(20))
    for combo in itertools.product(range(32), repeat=len(shifts)):
        sup = bit0_support(p, dict(zip(shifts, combo)))
        if sup is not None and need <= sup:
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    wit = flip_witnesses(h.ROOTS[0])
    assert len(wit) == 20
    rows = {}
    for path in args.inputs:
        opener = gzip.open if path.endswith('.gz') else open
        for line in opener(path, 'rt'):
            r = json.loads(line)
            if r['idx'] not in rows or r['result'] == 'unsat':
                rows[r['idx']] = r
    ts = h.templates4()
    rest = []
    for idx, r in sorted(rows.items()):
        if r['result'] == 'unsat':
            continue
        p = ts[idx]
        rest.append(dict(idx=idx, template=g.describe(p),
                         resolved='bit0-dependency' if dependency_excluded(p) else 'open'))
    how = {}
    for r in rows.values():
        if r['result'] == 'unsat':
            k = r.get('how', 'solver')
            how[k] = how.get(k, 0) + 1
    summary = dict(templates=len(ts), queries=len(rows), unsat=how, timeouts=len(rest), by_argument={})
    for r in rest:
        summary['by_argument'][r['resolved']] = summary['by_argument'].get(r['resolved'], 0) + 1
    print(json.dumps(summary))
    if args.output:
        args.output.write_text(json.dumps(dict(summary=summary, flip_witnesses=wit, rest=rest),
                                          indent=1) + '\n')


if __name__ == '__main__':
    main()
