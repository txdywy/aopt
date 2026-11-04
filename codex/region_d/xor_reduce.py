"""Input-XOR absorption. If I1 = XOR(z, k) and no later instruction reads z, then with
z' = z ^ k the program reads only z', and F(z ^ K) ^ D = F(z' ^ (k ^ K)) ^ D has the same
form (K is free). So the wiring computes exactly the functions of the 3-instruction
program in z', padded with a trailing XOR by a private constant (= 0): a wiring with the
SHR one position earlier. It is excluded iff that reduced wiring is excluded."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drive import all_wirings


def canon(op, i, idx):
    ka, kb = i + 1, i + 2
    consts = {ka, kb}
    if op == 'mad':
        a, b, c = idx
        a, b = min(a, b), max(a, b)
        used = [v for v in (a, b, c) if v in consts]
        if used and used[0] != ka:            # rename so the first constant is k0
            sw = {ka: kb, kb: ka}
            a, b, c = [sw.get(v, v) for v in (a, b, c)]
            a, b = min(a, b), max(a, b)
        return (a, b, c)
    if op == '>>':
        return tuple(idx)
    a, b = sorted(idx)
    if b == kb and a not in consts:
        b = ka
    return (a, b)


def reduce(order, wiring):
    if order[0] != '^':
        return None
    if sorted(wiring[0]) not in ([0, 1], [0, 2]):
        return None
    n = len(order)
    if any(0 in wiring[j] for j in range(1, n)):
        return None
    new_w = []
    for j in range(1, n):
        def m(x, j=j):
            if x == 1:
                return 0
            if 2 <= x <= j:
                return x - 1
            if x == j + 1:
                return j
            return j + 1
        new_w.append(canon(order[j], j - 1, tuple(m(x) for x in wiring[j])))
    new_o = tuple(order[1:]) + ('^',)
    new_w.append((n - 1, n))                  # pad: t_last ^ k (k = 0)
    return new_o, tuple(new_w)


_index = {}


def lookup(o, w):
    pos = o.index('>>')
    if pos not in _index:
        _index[pos] = {(tuple(oo), tuple(tuple(x) for x in ww)): i for i, (oo, ww) in enumerate(all_wirings(1, pos))}
    return pos, _index[pos].get((tuple(o), tuple(tuple(x) for x in w)))


if __name__ == '__main__':
    import json, collections
    from coverage2 import load, final
    v1 = {}
    for r in load('ch_p1.jsonl'):
        if v1.get(r['i']) != 'unsat':
            v1[r['i']] = final(r)
    for fn in ('ca_res.jsonl', 'cb_res.jsonl', 'cc_res.jsonl', 'cq_res.jsonl'):
        for r in load(fn):
            if r.get('pos') == 1 and (final(r) == 'unsat' or r.get('result') == 'unsat'):
                v1[r['i']] = 'unsat'
    allw2 = all_wirings(1, 2)
    stat = collections.Counter(); closed = []
    for i, (o, w) in enumerate(allw2):
        red = reduce(o, w)
        if red is None:
            continue
        pos, j = lookup(*red)
        if j is None:
            stat['reduced wiring not in enumeration'] += 1
            continue
        verdict = v1.get(j)
        stat['reducible, pos1 target ' + str(verdict)] += 1
        if verdict == 'unsat':
            closed.append((i, j))
    print(dict(stat))
    json.dump(closed, open('xor_reduce_p2.json', 'w'))
    print('pos2 wirings closed by input-XOR absorption:', len(closed))
