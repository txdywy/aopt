"""Hybrid driver v2 for single constant-shift wirings at SHR position POS.

v2: when the rdg fast path is not UNSAT and the shifted value is affine in z, the
fiber reduction rda (exact at c bits, every shift, see rda.py) is used instead of
rd3 bands: c = 11, escalated to 13 and 15 while some case is SAT.

usage: drive_h.py POS OUT.jsonl T_FAST T_BAND WORKERS
For each wiring: rdg at full width with a free 32-bit shift amount (covers every s
in one query, exact). If it is UNSAT the wiring is closed for all shifts. Otherwise
(timeout or low-bit SAT) fall back to rd3 bands; when the shift input is z, shifts
s >= 11 are covered by Theorem C0, so only [1,8] and [9,10] are computed.
Index i = position in drive.all_wirings(1, POS), the same as the band campaign.
"""
import json
import os
import sys
import time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/user/aopt')
from drive import all_wirings


def work(args):
    i, o, w, pos, t_fast, t_band = args
    import rdg
    import rd3
    import rda
    import rda2
    import rdq
    t = time.time()
    rec = dict(i=i, order=o, wiring=w)
    try:
        r = rdg.solve(o, w, 8, t_fast)
        rec['rdg'] = r['result']
        if r['result'] == 'sat':
            rec['rdg_consts'], rec['rdg_K'] = r.get('consts'), r.get('K')
        if r['result'] != 'unsat':
            if rda.applicable(o, w):
                rec['rda'] = {}
                for c in (11, 13, 15):
                    x = rda.solve(o, w, c, t_band)
                    ent = dict(result=x['result'], cases={k: v['result'] for k, v in x['cases'].items()})
                    for k, v in x['cases'].items():
                        if v['result'] == 'sat':
                            ent.setdefault('sat', {})[k] = dict(params=v['params'], K=v['K'], consts=v['consts'])
                    rec['rda'][c] = ent
                    if x['result'] != 'sat':
                        break
            elif rda2.applicable(o, w):
                rec['rda2'] = {}
                for c in (11, 13, 15):
                    x = rda2.solve(o, w, c, t_band)
                    ent = dict(result=x['result'], cases={k: v['result'] for k, v in x['cases'].items()})
                    rec['rda2'][c] = ent
                    if x['result'] != 'sat':
                        break
            elif rdq.applicable(o, w):
                x = rdq.solve_all(o, w, 11, t_band)
                rec['rdq'] = dict(result=x['result'], parts=x['parts'])
            else:
                bands = [(1, 8), (9, 10)] if w[pos][0] == 0 else [(1, 8), (9, 16), (17, 31)]
                rec['bands'] = {}
                for lo, hi in bands:
                    b = rd3.solve(o, w, lo, hi, 8, t_band)
                    ent = dict(result=b['result'])
                    if b['result'] == 'sat':
                        ent.update(consts=b.get('consts'), K=b.get('K'))
                    rec['bands'][f'{lo}-{hi}'] = ent
    except Exception as e:
        rec['error'] = repr(e)[:200]
    rec['sec'] = round(time.time() - t, 3)
    return rec


def main():
    pos, out = int(sys.argv[1]), sys.argv[2]
    t_fast, t_band = float(sys.argv[3]), float(sys.argv[4])
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    allw = all_wirings(1, pos)
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                done.add(json.loads(line)['i'])
            except Exception:
                pass
    todo = [(i, o, w, pos, t_fast, t_band) for i, (o, w) in enumerate(allw) if i not in done]
    print(f'pos {pos} total {len(allw)} done {len(done)} todo {len(todo)}', flush=True)
    t0 = time.time()
    with Pool(workers) as pool, open(out, 'a') as f:
        for n, r in enumerate(pool.imap_unordered(work, todo, chunksize=8), 1):
            f.write(json.dumps(r, default=list) + '\n')
            f.flush()
            bad = [k for k, v in (r.get('bands') or {}).items() if v['result'] != 'unsat']
            for key in ('rda', 'rda2'):
                if key in r:
                    last = r[key][max(r[key])]
                    if last['result'] != 'unsat':
                        bad.append('%s c=%d %s' % (key, max(r[key]), last['result']))
            if 'rdq' in r and r['rdq']['result'] != 'unsat':
                bad.append('rdq %s' % r['rdq']['result'])
            if 'error' in r or bad:
                print('NOTE', r['i'], r['order'], r['wiring'], 'rdg', r.get('rdg'), 'open bands', bad, r.get('error', ''), flush=True)
            if n % 2000 == 0:
                f.flush()
                print(f'progress {n}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
    print(f'ALL DONE H pos{pos} {len(todo)} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
