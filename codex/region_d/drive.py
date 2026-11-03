"""Parallel, resumable driver for region D constant-shift wirings.

usage: drive.py OUT.jsonl S_LO S_HI C SECONDS [WORKERS] [NSHR] [POS]
Each line of OUT.jsonl: {"i":..., "order":..., "wiring":..., "band":[lo,hi], "c":..,
"result":..., "Wmax":..., "sec":..., ["consts":..,"K":..]}
NSHR selects wirings with exactly that many SHRs (default 1).
"""
import json, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/user/aopt')
from count import wirings, orders_with_shr


def all_wirings(nshr, pos=None):
    out = []
    for o in orders_with_shr():
        if o[:3].count('>>') != nshr:
            continue
        if pos is not None and o.index('>>') != pos:
            continue
        for w in wirings(o, False):
            out.append((o, w))
    return out


def work(args):
    i, o, w, lo, hi, c, sec = args
    from rd3 import solve
    t = time.time()
    try:
        r = solve(o, w, lo, hi, c, sec)
    except Exception as e:                          # keep going, record the failure
        r = dict(result='error', error=repr(e)[:200])
    r.update(i=i, order=o, wiring=w, band=[lo, hi], c=c, sec=round(time.time() - t, 3))
    return r


def main():
    out, lo, hi, c, sec = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
    workers = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    nshr = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    pos = int(sys.argv[8]) if len(sys.argv) > 8 and sys.argv[8] != '-' else None
    allw = all_wirings(nshr, pos)
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)['i'])
                except Exception:
                    pass
    # Theorem C0: a single shift whose input is z is excluded for every s >= 11, so a band
    # starting at 11 or later needs no computation for those wirings (indices are unchanged).
    skip_c0 = nshr == 1 and lo >= 11
    todo = [(i, o, w, lo, hi, c, sec) for i, (o, w) in enumerate(allw)
            if i not in done and not (skip_c0 and w[o.index('>>')][0] == 0)]
    print(f'total {len(allw)} done {len(done)} todo {len(todo)} (C0 skip={skip_c0})', flush=True)
    t0 = time.time()
    n = 0
    with Pool(workers) as pool, open(out, 'a') as f:
        for r in pool.imap_unordered(work, todo, chunksize=8):
            f.write(json.dumps(r, default=list) + '\n')
            n += 1
            if r['result'] not in ('unsat',):
                print('NOTE', r['result'], r['i'], r['order'], r['wiring'], r.get('Wmax'), r['sec'], flush=True)
            if n % 2000 == 0:
                f.flush()
                print(f'progress {n}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
    print(f'ALL DONE {n} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
