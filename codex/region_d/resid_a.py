"""Residual closer: run rda (c = 11, then 13, 15 while SAT) on every rda-applicable
wiring left open by earlier passes (ch_p1/ch_p2 hybrid records whose rdg was not UNSAT
and whose bands are not all UNSAT; cs_p2 band records that are not UNSAT).
usage: resid_a.py OUT.jsonl SECONDS WORKERS"""
import json, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rda


def load(fn):
    out = []
    if os.path.exists(fn):
        for line in open(fn):
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def residuals():
    res = {}
    for pos, fn in ((1, 'ch_p1.jsonl'), (2, 'ch_p2.jsonl')):
        for r in load(fn):
            if r.get('rdg') == 'unsat' or 'rda' in r:
                continue
            bands = r.get('bands') or {}
            if bands and all(v['result'] == 'unsat' for v in bands.values()):
                continue
            res[(pos, r['i'])] = (tuple(r['order']), tuple(tuple(x) for x in r['wiring']))
    for r in load('cs_p2_s1_8.jsonl'):
        if r.get('result') != 'unsat':
            res[(2, r['i'])] = (tuple(r['order']), tuple(tuple(x) for x in r['wiring']))
    return res


def work(args):
    key, o, w, sec = args
    t = time.time()
    rec = dict(pos=key[0], i=key[1], order=o, wiring=w, rda={})
    try:
        for c in (11, 13, 15):
            x = rda.solve(o, w, c, sec)
            ent = dict(result=x['result'], cases={k: v['result'] for k, v in x['cases'].items()})
            rec['rda'][c] = ent
            if x['result'] != 'sat':
                break
    except Exception as e:
        rec['error'] = repr(e)[:200]
    rec['sec'] = round(time.time() - t, 2)
    return rec


def main():
    out, sec, workers = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    done = {(r['pos'], r['i']) for r in load(out)}
    res = residuals()
    app = {k: v for k, v in res.items() if rda.applicable(*v)}
    non = sorted(k for k in res if k not in app)
    todo = [(k, o, w, sec) for k, (o, w) in sorted(app.items()) if k not in done]
    print(f'residuals {len(res)} applicable {len(app)} non-applicable {len(non)} todo {len(todo)}', flush=True)
    json.dump([[k[0], k[1], res[k][0], res[k][1]] for k in non], open('resid_nonapplicable.json', 'w'))
    with Pool(workers) as pool, open(out, 'a') as f:
        for r in pool.imap_unordered(work, todo):
            f.write(json.dumps(r, default=list) + '\n')
            f.flush()
            last = r['rda'][max(r['rda'])] if r['rda'] else {'result': 'error'}
            if last['result'] != 'unsat':
                print('NOTE', r['pos'], r['i'], r['order'], r['wiring'], r['rda'], r.get('error', ''), flush=True)
    print('RESID DONE', len(todo), flush=True)


if __name__ == '__main__':
    main()
