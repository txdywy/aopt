"""General exact driver (rdg) over the complete SHR operand space.

usage: drive_g2.py MODE OUT.jsonl C SECONDS WORKERS [T_FAST]
v2: rdg at c=C with budget T_FAST first; if not UNSAT and rdvf applies (single SHR with
an affine runtime shift amount), rdvf at c=11 decides it; otherwise rdg full budget.
MODE multi    : every wiring with >= 2 SHRs except all-constant kinds (cc, ccc),
                which the campaign covers with rd3.
MODE kescape  : single 'k' wirings (constant >> runtime b) whose b-subprogram escapes
                Theorem V according to varshift3_cache.jsonl.
Index i = position in the full tagged enumeration (orders_with_shr x wirings_tagged).
"""
import json
import os
import sys
import time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/user/aopt')
from count import orders_with_shr
from count2 import wirings_tagged


def enumerate_all():
    out = []
    for o in orders_with_shr():
        for w, kinds in wirings_tagged(o):
            out.append((o, w, kinds))
    return out


def work(args):
    i, o, w, c, sec, t_fast = args
    import rdg
    import rdvf2 as rdvf
    t = time.time()
    try:
        r = rdg.solve(o, w, c, t_fast)
        if r['result'] != 'unsat':
            if rdvf.applicable(o, w):
                x = rdvf.solve(o, w, 11, sec)
                r = dict(result=x['result'], rdg_fast=r['result'], rdvf=True, c_rdvf=11, it=x.get('it'))
            else:
                r = rdg.solve(o, w, c, sec)
    except Exception as e:
        r = dict(result='error', error=repr(e)[:200])
    r.update(i=i, order=o, wiring=w, c=c, sec=round(time.time() - t, 3))
    return r


def main():
    mode, out, c, sec = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    t_fast = float(sys.argv[6]) if len(sys.argv) > 6 else 8.0
    allw = enumerate_all()
    if mode == 'multi':
        sel = [i for i, (o, w, k) in enumerate(allw) if len(k) >= 2 and k not in ('cc', 'ccc')]
    elif mode == 'kescape':
        esc = set()
        for line in open('varshift3_cache.jsonl'):
            r = json.loads(line)
            if r['result'] != 'unsat':
                esc.add((tuple(r['order']), tuple(tuple(x) for x in r['wiring'])))
        sel = []
        for i, (o, w, k) in enumerate(allw):
            if k != 'k':
                continue
            j = o.index('>>')
            b = w[j][1]
            if b and (o[:b], tuple(tuple(x) for x in w[:b])) in esc:
                sel.append(i)
    else:
        raise SystemExit('unknown mode')
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                done.add(json.loads(line)['i'])
            except Exception:
                pass
    todo = [(i, allw[i][0], allw[i][1], c, sec, t_fast) for i in sel if i not in done]
    print(f'mode {mode} selected {len(sel)} done {len(done)} todo {len(todo)}', flush=True)
    t0 = time.time()
    with Pool(workers) as pool, open(out, 'a') as f:
        for n, r in enumerate(pool.imap_unordered(work, todo, chunksize=8), 1):
            f.write(json.dumps(r, default=list) + '\n')
            f.flush()
            if r['result'] != 'unsat':
                print('NOTE', r['result'], r['i'], r['order'], r['wiring'], r['sec'], flush=True)
            if n % 5000 == 0:
                f.flush()
                print(f'progress {n}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
    print(f'ALL DONE {mode} {len(todo)} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
