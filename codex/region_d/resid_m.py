"""Residual closer for multi-SHR (and k-type) wirings left open by drive_g2: exact rdg with
the FIRST constant shift literal fixed, one query per s in [0, 31] plus one for s >= 32,
other literals free. A part that is SAT at c=8 is re-run at c=11, 13, 15 (same part);
UNSAT at any width closes the part. Wirings without a constant literal get one long rdg
query per width. usage: resid_m.py SRC.jsonl OUT.jsonl SECONDS"""
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import z3
import rdg
import rdm
from coverage2 import load


def literal_slots(order, wiring):
    out = []
    for i, op in enumerate(order):
        if op == '>>':
            a, b = wiring[i]
            if b > i:
                out.append((i, b - i - 1))
    return out


def part_verdict(o, w, extra, sec):
    log = []
    for c in (8, 11, 13, 15):
        r = rdg.solve(o, w, c, sec, extra=extra)
        log.append((c, r['result']))
        if r['result'] != 'sat':
            return r['result'], log
    return 'sat', log


def close_one(args):
    r, sec = args
    o = tuple(r['order']); w = tuple(tuple(x) for x in r['wiring'])
    t = time.time()
    try:
        if rdm.applicable(o, w):
            x = rdm.solve_all(o, w, 11, 3 * sec)
            return dict(i=r['i'], order=o, wiring=w, result=x['result'], how='rdm', parts=x['parts'],
                        sec=round(time.time() - t, 1))
        slots = literal_slots(o, w)
        parts = []
        if slots:
            i0, sl = slots[0]
            for s in range(33):
                if s < 32:
                    ex = lambda S, ks, s=s: S.add(ks[i0][sl] == s)
                else:
                    ex = lambda S, ks: S.add(z3.UGE(ks[i0][sl], 32))
                v, log = part_verdict(o, w, ex, sec)
                parts.append((s, v, log))
        else:
            v, log = part_verdict(o, w, None, sec * 4)
            parts.append(('all', v, log))
        verdict = 'unsat' if parts and all(p[1] == 'unsat' for p in parts) else (
            'sat' if any(p[1] == 'sat' for p in parts) else 'timeout')
        return dict(i=r['i'], order=o, wiring=w, result=verdict, how='fixed-literal', parts=parts,
                    sec=round(time.time() - t, 1))
    except Exception as e:
        return dict(i=r['i'], order=o, wiring=w, result='error', error=repr(e)[:200], sec=round(time.time() - t, 1))


def main():
    src, out, sec = sys.argv[1], sys.argv[2], float(sys.argv[3])
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    done = {r['i'] for r in load(out) if r.get('result') == 'unsat'}
    best = {}
    for r in load(src):
        if best.get(r['i'], {}).get('result') == 'unsat':
            continue
        best[r['i']] = r
    todo = [r for i, r in sorted(best.items()) if r['result'] != 'unsat' and i not in done]
    print('residuals', len(todo), 'workers', workers, flush=True)
    from multiprocessing import Pool
    with Pool(workers) as pool, open(out, 'a') as f:
        for rec in pool.imap_unordered(close_one, [(r, sec) for r in todo], chunksize=1):
            f.write(json.dumps(rec, default=list) + '\n')
            f.flush()
            print(rec['i'], rec['order'], rec['wiring'], rec.get('how'), rec['result'], rec['sec'], flush=True)
    print('RESIDM DONE', flush=True)


if __name__ == '__main__':
    main()
