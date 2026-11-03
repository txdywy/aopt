"""Soundness cross-check: every wiring with a 32-bit-verified SAT at c=8 (rd3 band
[1,8], from cs_p2_s1_8.jsonl and cs_p0 files) that rda can handle must be SAT under
rda at c=8 as well; an rda UNSAT there would expose an unsound reduction."""
import json, sys, time, glob
sys.path.insert(0, '.')
import rda
seen = set()
rows = []
for fn in ['cs_p2_s1_8.jsonl'] + sorted(glob.glob('cs_p0_*.jsonl')):
    for line in open(fn):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get('result') != 'sat':
            continue
        o = tuple(r['order']); w = tuple(tuple(x) for x in r['wiring'])
        if (o, w) in seen or not rda.applicable(o, w):
            continue
        seen.add((o, w))
        rows.append((fn, r['i'], o, w))
print('applicable verified c=8 SAT wirings:', len(rows), flush=True)
bad = 0
for fn, i, o, w in rows:
    t = time.time()
    res = rda.solve(o, w, 8, 120)
    tag = res['result']
    if tag == 'unsat':
        bad += 1
    print(fn, i, o, w, tag, {k: v['result'] for k, v in res['cases'].items()}, '%.1fs' % (time.time() - t), flush=True)
print('UNSOUND (rda unsat on a real SAT):', bad)
