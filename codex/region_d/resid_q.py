"""rdq.solve_all (square path w*w+w, every e as its own query) on the non-affine residual
list. usage: resid_q.py OUT.jsonl SECONDS"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rdq
out, sec = sys.argv[1], float(sys.argv[2])
done = set()
if os.path.exists(out):
    for l in open(out):
        try:
            r = json.loads(l); done.add((r['pos'], r['i']))
        except Exception:
            pass
items = json.load(open('resid_nonapplicable.json'))
with open(out, 'a') as f:
    for pos, i, o, w in items:
        o = tuple(o); w = tuple(tuple(x) for x in w)
        if (pos, i) in done or not rdq.applicable(o, w):
            continue
        t = time.time()
        x = rdq.solve_all(o, w, 11, sec)
        rec = dict(pos=pos, i=i, order=o, wiring=w, rdq=dict(result=x['result'], parts=x['parts']), sec=round(time.time() - t, 1))
        f.write(json.dumps(rec, default=list) + '\n'); f.flush()
        print(pos, i, o, w, x['result'], [p for p in x['parts'] if p[1] != 'unsat'], rec['sec'], flush=True)
print('RESIDQ DONE', flush=True)
