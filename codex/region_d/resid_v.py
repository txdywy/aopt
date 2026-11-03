"""rdvf (c=11) on variable-shift records of cv_c8.jsonl that are not UNSAT and were not
decided by rdvf. usage: resid_v.py OUT.jsonl SECONDS WORKERS"""
import json, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rdvf
import rdvq


def work(a):
    i, o, w, sec = a
    t = time.time()
    try:
        if rdvf.applicable(o, w):
            x = rdvf.solve(o, w, 11, sec)
            r = dict(result=x['result'], it=x.get('it'), solver='rdvf')
        else:
            x = rdvq.solve_all(o, w, 11, sec)
            r = dict(result=x['result'], parts=x['parts'], solver='rdvq')
    except Exception as e:
        r = dict(result='error', error=repr(e)[:200])
    r.update(i=i, order=o, wiring=w, c=11, sec=round(time.time() - t, 2))
    return r


if __name__ == '__main__':
    out, sec, workers = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try: done.add(json.loads(l)['i'])
            except Exception: pass
    items, skip = [], []
    for l in open('cv_c8.jsonl'):
        try: r = json.loads(l)
        except Exception: continue
        if r['result'] == 'unsat' or r.get('rdvf') or r.get('rdvq') or r['i'] in done:
            continue
        o = tuple(r['order']); w = tuple(tuple(x) for x in r['wiring'])
        (items if (rdvf.applicable(o, w) or rdvq.applicable(o, w)) else skip).append((r['i'], o, w, sec))
    print('todo', len(items), 'not rdvf-applicable', len(skip), flush=True)
    for s in skip:
        print('SKIP', s[0], s[1], s[2], flush=True)
    with Pool(workers) as pool, open(out, 'a') as f:
        for r in pool.imap_unordered(work, items):
            f.write(json.dumps(r, default=list) + '\n'); f.flush()
            print(r['i'], r['order'], r['wiring'], r['result'], r['sec'], flush=True)
    print('RESIDV DONE', flush=True)
