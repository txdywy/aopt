"""Stricter soundness check: classify each verified c=8 solution into its fiber case
(from its own constants and shift) and require that rda's case of that kind is SAT."""
import json, sys, glob, random
sys.path.insert(0, '.')
import rda, test_rda
from rd2 import K1, K2, C4, D
M = (1 << 32) - 1
def ev(order, wiring, consts, z, upto):
    vals = [z]
    for i in range(upto):
        pool = vals + list(consts[i])
        a = [pool[j] for j in wiring[i]]
        op = order[i]
        if op == 'mad': v = (a[0] * a[1] + a[2]) & M
        elif op == '^': v = a[0] ^ a[1]
        elif op == '&': v = a[0] & a[1]
        elif op == '|': v = a[0] | a[1]
        else: v = (a[0] >> a[1]) if a[1] < 32 else 0
        vals.append(v)
    return vals
rows = []
seen = set()
for fn in ['cs_p2_s1_8.jsonl'] + sorted(glob.glob('cs_p0_*.jsonl')):
    for line in open(fn):
        try: r = json.loads(line)
        except Exception: continue
        if r.get('result') != 'sat': continue
        o = tuple(r['order']); w = tuple(tuple(x) for x in r['wiring'])
        if not rda.applicable(o, w): continue
        rows.append((fn, r, o, w))
print('records', len(rows))
bad = 0; checked = 0; kinds = {}
cache = {}
rng = random.Random(5)
for fn, r, o, w in rows:
    c = r['c']; cm = (1 << c) - 1
    consts = [tuple(x) for x in r['consts']]
    pos = o.index('>>'); vi, ki = w[pos]
    s = consts[pos][ki - pos - 1]
    # sanity: the stored solution matches the target mod 2^c on random 32-bit z
    ok = True
    for _ in range(300):
        z = rng.getrandbits(32)
        out = ev(o, w, consts, z, 4)[-1] & cm
        y = (z ^ r['K']) & cm
        t = ((9 * (((33 * y + K1) & cm) ^ ((16896 * y + K2) & cm)) + C4) & cm) ^ (D & cm)
        ok &= out == t
    v0 = ev(o, w, consts, 0, pos)[vi]; v1 = ev(o, w, consts, 1, pos)[vi]
    A, B = (v1 - v0) & M, v0
    kind, p = test_rda.classify(A, B, s, c)
    kinds[kind] = kinds.get(kind, 0) + 1
    key = (o, w, kind, c)
    if key not in cache:
        cache[key] = rda.solve_case(o, w, kind, c, 120)['result']
    res = cache[key]
    checked += 1
    if res != 'sat' or not ok:
        bad += 1
        print('PROBLEM', fn, r['i'], o, w, 'kind', kind, 'rda', res, 'record ok', ok)
print('checked', checked, 'kinds', kinds, 'problems', bad)
