"""Validation of rda2 (bitwise path).
L: lemma check u = ((A z + B) op k) >> s mod 2^c lies in {G(u_w(r), ks)} for random
   32-bit A, B, k, s, op chains, and the true ks satisfies the tie constraints.
R: every 32-bit-verified c=8 SAT record (cs_p0/cs_p2 band files) that rda2 can handle
   must be SAT under rda2 in the case its own constants fall into.
"""
import glob, json, random, sys
import numpy as np
sys.path.insert(0, '.')
import rda2, test_rda
from rd2 import K1, K2, C4, D
M = (1 << 32) - 1


def lemma(trials=3000, c=11, seed=11):
    rng = random.Random(seed)
    cm = (1 << c) - 1
    ops = {'^': lambda a, b: a ^ b, '&': lambda a, b: a & b, '|': lambda a, b: a | b}
    for _ in range(trials):
        e = rng.choice(range(0, 33))
        A = 0 if e >= 32 else ((rng.getrandbits(32 - e) | 1) << e) & M
        B = rng.getrandbits(32)
        chain = [(rng.choice('^&|'), rng.getrandbits(32)) for _ in range(rng.choice((1, 2)))]
        s = rng.choice(list(range(0, 33)))
        kind, p = test_rda.classify(A, B, s, c)
        z0 = rng.getrandbits(c)
        uws, _ = test_rda.model_values(kind, p, z0, c)
        kss = [((k >> s) & cm) if s < 32 else 0 for _, k in chain]
        for (op, k), ksv in zip(chain, kss):
            kc = k & cm
            assert (ksv & (cm >> s if s < 64 else 0)) == (kc >> s if s < 64 else 0)
            if s > 32 - c:
                assert ksv >> (32 - s) == 0
        model = set()
        for uw in uws:
            x = uw
            for (op, _), ksv in zip(chain, kss):
                x = ops[op](x, ksv)
            model.add(x)
        for _ in range(5):
            z = z0 + (rng.getrandbits(32 - c) << c)
            x = (A * z + B) & M
            for op, k in chain:
                x = ops[op](x, k)
            u = (x >> s) & cm if s < 32 else 0
            assert u in model, (kind, A, B, s, chain, z)
    return trials


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


def records():
    rng = random.Random(5)
    probs = checked = 0
    cache = {}
    kinds = {}
    for fn in ['cs_p2_s1_8.jsonl'] + sorted(glob.glob('cs_p0_*.jsonl')) + sorted(glob.glob('cs_p1_*.jsonl')):
        for line in open(fn):
            try: r = json.loads(line)
            except Exception: continue
            if r.get('result') != 'sat': continue
            o = tuple(r['order']); w = tuple(tuple(x) for x in r['wiring'])
            if not rda2.applicable(o, w): continue
            c = r['c']; cm = (1 << c) - 1
            consts = [tuple(x) for x in r['consts']]
            pos = o.index('>>'); vi, ki = w[pos]
            s = consts[pos][ki - pos - 1]
            ok = True
            for _ in range(300):
                z = rng.getrandbits(32)
                out = ev(o, w, consts, z, 4)[-1] & cm
                y = (z ^ r['K']) & cm
                t = ((9 * (((33 * y + K1) & cm) ^ ((16896 * y + K2) & cm)) + C4) & cm) ^ (D & cm)
                ok &= out == t
            root, path = rda2.bitwise_path(o, w, pos)
            w0 = ev(o, w, consts, 0, pos)[root]; w1 = ev(o, w, consts, 1, pos)[root]
            A, B = (w1 - w0) & M, w0
            kind, _ = test_rda.classify(A, B, s, c)
            kinds[kind] = kinds.get(kind, 0) + 1
            key = (o, w, kind, c)
            if key not in cache:
                cache[key] = rda2.solve_case(o, w, kind, c, 120)['result']
            checked += 1
            if cache[key] != 'sat' or not ok:
                probs += 1
                print('PROBLEM', fn, r['i'], o, w, kind, cache[key], ok, flush=True)
    return checked, kinds, probs


if __name__ == '__main__':
    print('lemma trials ok:', lemma(), flush=True)
    print('records checked, kinds, problems:', records(), flush=True)
