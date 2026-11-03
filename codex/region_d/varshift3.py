"""Variable-shift classifier, parallel and resumable (same logic as varshift.py).

Step 1 collects the distinct shift-amount subprograms (b = t_i), step 2 solves each
escape query once with a persistent cache, step 3 classifies every wiring.
See varshift.py for Theorem V and the soundness of the escape query.
"""
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/user/aopt')
from count import wirings, orders_with_shr


def solve_key(args):
    o, w, i, sec = args
    from varshift import escape
    t = time.time()
    return o, w, escape(o, w, i, sec), round(time.time() - t, 2)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'varshift3.json'
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    sec = float(sys.argv[3]) if len(sys.argv) > 3 else 120
    cache_fn = out.replace('.json', '_cache.jsonl')
    cache = {}
    if os.path.exists(cache_fn):
        for line in open(cache_fn):
            try:
                r = json.loads(line)
                cache[(tuple(r['order']), tuple(tuple(x) for x in r['wiring']))] = r['result']
            except Exception:
                pass
    t0 = time.time()
    rows = []                      # (order, wiring, j, i or None)
    keys = {}
    for o in orders_with_shr():
        if o[:3].count('>>') != 1:
            continue
        j = o.index('>>')
        for w in wirings(o, True):
            b = w[j][1]
            if b == 0:
                rows.append((o, None))
                continue
            i = b - 1
            k = (o[:i + 1], tuple(tuple(x) for x in w[:i + 1]))
            keys.setdefault(k, i)
            rows.append((o, k))
    todo = [(k[0], k[1], i, sec) for k, i in keys.items() if k not in cache]
    print(f'wirings {len(rows)} distinct subprograms {len(keys)} cached {len(keys) - len(todo)} todo {len(todo)}', flush=True)
    with Pool(workers) as pool, open(cache_fn, 'a') as cf:
        for n, (o, w, res, dt) in enumerate(pool.imap_unordered(solve_key, todo), 1):
            cache[(o, w)] = res
            cf.write(json.dumps(dict(order=o, wiring=w, result=res, sec=dt)) + '\n')
            cf.flush()
            if res != 'unsat' or n % 25 == 0:
                print(f'[{n}/{len(todo)}] {res} {dt}s {o} {w}', flush=True)
    stats = Counter()
    for o, k in rows:
        if k is None:
            stats['b=z excluded (Theorem V)'] += 1
        elif cache[k] == 'unsat':
            stats['b=t_i excluded (Theorem V, escape UNSAT)'] += 1
        else:
            stats['escape ' + cache[k]] += 1
    sub = Counter(cache[k] for k in keys)
    res = dict(stats=dict(stats), distinct_subprograms=len(keys), subprogram_results=dict(sub),
               escape_subprograms=[dict(order=k[0], wiring=k[1], result=cache[k]) for k in keys if cache[k] != 'unsat'],
               seconds=round(time.time() - t0, 1))
    json.dump(res, open(out, 'w'), indent=1, default=list)
    print('VARSHIFT DONE', json.dumps(dict(stats=res['stats'], subprograms=res['subprogram_results'])), flush=True)


if __name__ == '__main__':
    main()
