import json, random, sys
sys.path.insert(0, '.')
import xor_reduce
from drive import all_wirings
M = (1 << 32) - 1
rng = random.Random(9)


def run(order, wiring, consts, z):
    vals = [z]
    for i, op in enumerate(order):
        pool = vals + list(consts[i])
        a = [pool[j] for j in wiring[i]]
        if op == 'mad': v = (a[0] * a[1] + a[2]) & M
        elif op == '^': v = a[0] ^ a[1]
        elif op == '&': v = a[0] & a[1]
        elif op == '|': v = a[0] | a[1]
        else: v = (a[0] >> a[1]) if a[1] < 32 else 0
        vals.append(v)
    return vals[-1]


def reduced_consts(order, wiring, consts):
    """Constants for the reduced program: old instruction j -> new j-1, swapped when the
    canonical renaming swapped k0/k1; pad constant 0."""
    out = []
    for j in range(1, len(order)):
        k0, k1 = consts[j]
        ka, kb = j + 1, j + 2
        # recompute the remapped (pre-canonical) tuple to detect a swap
        def m(x, j=j):
            if x == 1: return 0
            if 2 <= x <= j: return x - 1
            if x == j + 1: return j
            return j + 1
        raw = tuple(m(x) for x in wiring[j])
        can = xor_reduce.canon(order[j], j - 1, raw)
        nka, nkb = j, j + 1
        swapped = False
        if order[j] == 'mad':
            a, b, c = raw
            a, b = min(a, b), max(a, b)
            used = [v for v in (a, b, c) if v in (nka, nkb)]
            swapped = bool(used) and used[0] != nka
        elif order[j] != '>>':
            a, b = sorted(raw)
            swapped = (b == nkb and a not in (nka, nkb))
        out.append((k1, k0) if swapped else (k0, k1))
    out.append((0, 0))
    return out


pairs = json.load(open('xor_reduce_p2.json'))
allw2 = all_wirings(1, 2)
allw1 = all_wirings(1, 1)
bad = 0
for i, j in pairs:
    o, w = allw2[i]
    o1, w1 = allw1[j]
    for _ in range(40):
        consts = [(rng.getrandbits(32), rng.getrandbits(32)) for _ in range(4)]
        if o[2] == '>>':
            consts[2] = (rng.randrange(0, 40), 0)
        k0 = consts[0][wiring_slot] if False else None
        # XOR constant of I1: the non-z operand of wiring[0]
        kidx = [x for x in w[0] if x != 0][0] - 1        # 0 -> k0, 1 -> k1
        k = consts[0][kidx]
        rc = reduced_consts(o, w, consts)
        z = rng.getrandbits(32)
        if run(o, w, consts, z) != run(o1, w1, rc, z ^ k):
            bad += 1
            print('MISMATCH', i, o, w, '->', j, o1, w1)
            break
print('pairs', len(pairs), 'mismatches', bad)
