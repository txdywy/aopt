"""Exact computation for variable-shift wirings that Theorem V does not exclude (v2).

usage: drive_v2.py OUT.jsonl C SECONDS WORKERS [T_FAST]
v2: rdv at c=C with a short budget T_FAST first; if that is not UNSAT and the shift
amount is affine (rdvf.applicable), the fiber reduction rdvf at c=11 (exact for every
constant choice, see rdvf.py) decides it; otherwise rdv gets the full budget.
Selects single-SHR variable-shift wirings whose shift amount b = t_i has an escaping
subprogram in varshift3_cache.jsonl, and solves each with rdv (exact, both shift
operands at full width). Index i is the position among all single-SHR
variable-shift wirings in enumeration order. Resumable.
"""
import json
import os
import sys
import time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/user/aopt')
from count import wirings, orders_with_shr


def all_v():
    out = []
    for o in orders_with_shr():
        if o[:3].count('>>') != 1:
            continue
        for w in wirings(o, True):
            out.append((o, w))
    return out


def work(args):
    i, o, w, c, sec, t_fast = args
    import rdv
    import rdvf
    import rdvq
    t = time.time()
    try:
        r = rdv.solve(o, w, c, t_fast)
        if r['result'] != 'unsat':
            if rdvf.applicable(o, w):
                x = rdvf.solve(o, w, 11, sec)
                r = dict(result=x['result'], rdv_fast=r['result'], rdvf=True, c_rdvf=11, it=x.get('it'))
            elif rdvq.applicable(o, w):
                x = rdvq.solve_all(o, w, 11, min(sec, 30.0))
                r = dict(result=x['result'], rdv_fast=r['result'], rdvq=True, c_rdvq=11, parts=x['parts'])
            else:
                r = rdv.solve(o, w, c, sec)
    except Exception as e:
        r = dict(result='error', error=repr(e)[:200])
    r.update(i=i, order=o, wiring=w, c=c, sec=round(time.time() - t, 3))
    return r


def main():
    out, c, sec = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    t_fast = float(sys.argv[5]) if len(sys.argv) > 5 else 8.0
    esc = set()
    for line in open('varshift3_cache.jsonl'):
        r = json.loads(line)
        if r['result'] != 'unsat':
            esc.add((tuple(r['order']), tuple(tuple(x) for x in r['wiring'])))
    allw = all_v()
    sel = []
    for i, (o, w) in enumerate(allw):
        j = o.index('>>')
        b = w[j][1]
        if b and (o[:b], tuple(tuple(x) for x in w[:b])) in esc:
            sel.append(i)
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                done.add(json.loads(line)['i'])
            except Exception:
                pass
    todo = [(i, allw[i][0], allw[i][1], c, sec, t_fast) for i in sel if i not in done]
    print(f'variable-shift wirings {len(allw)} escaping {len(sel)} done {len(done)} todo {len(todo)}', flush=True)
    t0 = time.time()
    with Pool(workers) as pool, open(out, 'a') as f:
        for n, r in enumerate(pool.imap_unordered(work, todo, chunksize=4), 1):
            f.write(json.dumps(r, default=list) + '\n')
            f.flush()
            if r['result'] != 'unsat':
                print('NOTE', r['result'], r['i'], r['order'], r['wiring'], r['sec'], flush=True)
            if n % 2000 == 0:
                f.flush()
                print(f'progress {n}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
    print(f'ALL DONE V2 {len(todo)} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
