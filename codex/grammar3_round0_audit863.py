"""Resolve round-0 grammar timeouts by exact arguments, list what is left.

Reads the round-0 result files of grammar3_head863 and, for every query that
timed out, applies one of these arguments; each is valid for every constant:

  few-values   the last instruction shifts a constant by a runtime amount, so it
               has at most 33 results and cannot be injective on 2^30 inputs.
  drops-bit0   the input is used only through i1 = u >> s; the domain (all u with
               the key's top two bits) is closed under u ^ 1, so u and u ^ 1
               collide unless s = 0, and s = 0 is the same template with the
               identity MAD in place of the shift.
  same-as-key0 the key is affine on the domain (u = +-x + const) and u is used
               only by i1 = MAD(u,K1,K2): the MAD absorbs the key, so the query
               equals the key-0 query of the same template.
  linear-out   as above, except that u is also an operand of the last instruction
               K*u + w or K*w + u. The sign of u moves into K, or the whole
               equation is negated (m stays odd); the constant moves into c.

A later retry with a longer limit overrides a timeout. Whatever no argument or
retry settles is reported as open.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from codex.grammar3_head863 import D30, MASK, templates, describe

AFFINE_KEYS = {t << 30 | low for t in range(4) for low in (0, D30 - 1)}


def load(paths):
    rows = {}
    for path in paths:
        opener = gzip.open if str(path).endswith('.gz') else open
        for line in opener(path, 'rt'):
            r = json.loads(line)
            k = (r['idx'], r['key'])
            # an UNSAT anywhere settles the query; otherwise keep the first result
            if k not in rows or r['result'] == 'unsat':
                rows[k] = r
    return rows


def audit(rows):
    ts = templates(False, True)
    by_desc = {describe(p): i for i, p in enumerate(ts)}
    out = []
    for (idx, key), r in sorted(rows.items()):
        if r['result'] == 'unsat':
            continue
        p = ts[idx]
        (f1, a1), (f2, a2), (f3, a3) = p
        uses_u = [n for n, (_, args) in enumerate(p) if 0 in args]
        why = None
        if f3 in ('kshl', 'kshr'):
            why = 'few-values'
        elif f1 == 'shrk' and uses_u == [0]:
            twin = describe([('aff', a1), (f2, a2), (f3, a3)])
            if rows.get((by_desc[twin], key), {}).get('result') == 'unsat':
                why = 'drops-bit0'
        key0 = rows.get((idx, 0), {}).get('result') == 'unsat'
        if why is None and f1 == 'aff' and uses_u == [0] and key in AFFINE_KEYS and key0:
            why = 'same-as-key0'
        if (why is None and f1 == 'aff' and uses_u == [0, 2] and f3 in ('lin1', 'lin2')
                and key in AFFINE_KEYS and key0):
            why = 'linear-out'
        out.append(dict(idx=idx, key=hex(key), template=r['template'], resolved=why or 'open'))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    rows = load(args.inputs)
    res = audit(rows)
    summary = dict(queries=len(rows), unsat=sum(r['result'] == 'unsat' for r in rows.values()),
                   timeouts=len(res), by_argument={})
    for r in res:
        summary['by_argument'][r['resolved']] = summary['by_argument'].get(r['resolved'], 0) + 1
    print(json.dumps(summary))
    for r in res:
        if r['resolved'] == 'open':
            print(json.dumps(r))
    if args.output:
        args.output.write_text(json.dumps(dict(summary=summary, timeouts=res), indent=1) + '\n')


if __name__ == '__main__':
    main()
