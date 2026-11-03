"""Run rda2 (bitwise-path fiber reduction, c = 11 -> 13 -> 15 while SAT) on the
non-affine residual list resid_nonapplicable.json. usage: resid_b.py OUT.jsonl SECONDS WORKERS"""
import json, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rda2


def work(args):
    pos, i, o, w, sec = args
    t = time.time()
    rec = dict(pos=pos, i=i, order=o, wiring=w, rda2={})
    try:
        for c in (11, 13, 15):
            x = rda2.solve(o, w, c, sec)
            rec['rda2'][c] = dict(result=x['result'], cases={k: v['result'] for k, v in x['cases'].items()})
            if x['result'] != 'sat':
                break
    except Exception as e:
        rec['error'] = repr(e)[:200]
    rec['sec'] = round(time.time() - t, 2)
    return rec


if __name__ == '__main__':
    out, sec, workers = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    items = json.load(open(sys.argv[4] if len(sys.argv) > 4 else 'resid_nonapplicable.json'))
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                r = json.loads(line); done.add((r['pos'], r['i']))
            except Exception:
                pass
    todo, skip = [], []
    for pos, i, o, w in items:
        o = tuple(o); w = tuple(tuple(x) for x in w)
        if (pos, i) in done:
            continue
        (todo if rda2.applicable(o, w) else skip).append((pos, i, o, w, sec))
    print('todo', len(todo), 'not rda2-applicable', len(skip), flush=True)
    for s in skip:
        print('SKIP', s[0], s[1], s[2], s[3], flush=True)
    with Pool(workers) as pool, open(out, 'a') as f:
        for r in pool.imap_unordered(work, todo):
            f.write(json.dumps(r, default=list) + '\n'); f.flush()
            last = r['rda2'][max(r['rda2'])] if r['rda2'] else {'result': 'error'}
            print(r['pos'], r['i'], r['order'], r['wiring'], {c: v['result'] for c, v in r['rda2'].items()}, r['sec'], r.get('error', ''), flush=True)
    print('RESIDB DONE', flush=True)
