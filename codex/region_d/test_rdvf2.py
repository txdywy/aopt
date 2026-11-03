"""Validation of rdvf2.
(1) rdvf's positive controls (affine amount) must be SAT under rdvf2.
(2) bitwise-amount controls: u = t0 >> (t0 & 7) with v2(A) >= 7 (HB = 0, b <= e), and
    u = t0 >> (t0 ^ k) with v2(A) >= 21 (no free part); realizable, must be SAT.
(3) real c=8 solutions (rdv SAT records in cv_c8.jsonl) whose own constants fall in the
    searched branch (ii) must be SAT under rdvf2 at c=8 (branch (i) is not used at c=8).
"""
import json, random, sys
import numpy as np
sys.path.insert(0, '.')
import rdvf2, test_rdvf, test_rdq
from rda import affine_kinds
M = (1 << 32) - 1
c = 11; cm = (1 << c) - 1
rng = random.Random(55)


def controls():
    res = []
    # (1) reuse rdvf's constructions
    o = ('mad', 'mad', '>>', '^'); w = ((0, 1, 2), (0, 2, 3), (1, 2), (3, 4))
    for t in range(3):
        A = ((rng.getrandbits(24) | 1) << 8) & M
        consts = [(A, rng.getrandbits(32)), (1 << 31, rng.randrange(0, 8)), (0, 0), (rng.getrandbits(32), 0)]
        res.append(('affine-free', o, w, consts))
    # (2a) AND amount, same root
    o2 = ('mad', '&', '>>', '^'); w2 = ((0, 1, 2), (1, 2), (1, 2), (3, 4))
    for t in range(3):
        e = rng.randrange(7, 21)
        A = ((rng.getrandbits(32 - e) | 1) << e) & M
        consts = [(A, rng.getrandbits(32)), (7, 0), (0, 0), (rng.getrandbits(32), 0)]
        res.append(('and-same', o2, w2, consts))
    # (2b) XOR amount, e >= 21
    o3 = ('mad', '^', '>>', '^'); w3 = ((0, 1, 2), (1, 2), (1, 2), (3, 4))
    for t in range(3):
        e = rng.randrange(21, 32)
        A = ((rng.getrandbits(32 - e) | 1) << e) & M
        consts = [(A, rng.getrandbits(32)), (rng.getrandbits(32), 0), (0, 0), (rng.getrandbits(32), 0)]
        res.append(('xor-e21', o3, w3, consts))
    out = []
    for kind, o, w, consts in res:
        tgt = np.array([test_rdq.run32(o, w, consts, z) & cm for z in range(1 << c)], dtype=np.uint64)
        for _ in range(300):
            z = rng.getrandbits(32)
            assert test_rdq.run32(o, w, consts, z) & cm == tgt[z & cm], kind
        r = rdvf2.solve(o, w, c, 300, 1, tgt)
        out.append((kind, r['result'], r.get('it')))
    return out


def real():
    rows = []
    for l in open('cv_c8.jsonl'):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get('result') != 'sat' or 'consts' not in r:
            continue
        o = tuple(r['order']); w = tuple(tuple(x) for x in r['wiring'])
        an = rdvf2.analyse(o, w)
        if an is None:
            continue
        consts = [tuple(x) for x in r['consts']]
        cc = r['c']; ccm = (1 << cc) - 1
        # branch classification from the solution's own constants
        def val(idx, z):
            return test_rdq_eval(o, w, consts, z)[idx]
        rows.append((r['i'], o, w, consts, r['K'], cc))
    return rows


def test_rdq_eval(o, w, consts, z):
    vals = [z]
    for i, op in enumerate(o):
        pool = vals + list(consts[i])
        a = [pool[j] for j in w[i]]
        if op == 'mad': v = (a[0] * a[1] + a[2]) & M
        elif op == '^': v = a[0] ^ a[1]
        elif op == '&': v = a[0] & a[1]
        elif op == '|': v = a[0] | a[1]
        else: v = (a[0] >> a[1]) if a[1] < 32 else 0
        vals.append(v)
    return vals


def real_check():
    rows = real()
    checked = bad = skipped = 0
    for i, o, w, consts, K, cc in rows:
        an = rdvf2.analyse(o, w)
        pos = an['pos']
        vals0 = test_rdq_eval(o, w, consts, 0); vals1 = test_rdq_eval(o, w, consts, 1)
        wb0, wb1 = vals0[an['b_root']], vals1[an['b_root']]
        Ab = (wb1 - wb0) & M
        e = (Ab & -Ab).bit_length() - 1 if Ab else 32
        # HB from the solution's bitwise path: G(0), G(~0)
        def G(x):
            sv = {an['b_root']: x}
            for ii in an['b_path']:
                args = [sv[j] if j <= ii else consts[ii][j - ii - 1] for j in w[ii]]
                op = o[ii]
                sv[ii + 1] = args[0] ^ args[1] if op == '^' else (args[0] & args[1] if op == '&' else args[0] | args[1])
            return sv[an['bi']]
        hb = ((G(0) | G(M)) >> (cc + e)) if cc + e < 32 else 0
        if not (e >= 32 - cc or hb == 0):
            skipped += 1
            continue
        r = rdvf2.solve(o, w, cc, 300, 1, None, allow_low_c=True)
        checked += 1
        if r['result'] != 'sat':
            bad += 1
            print('PROBLEM', i, o, w, r['result'], flush=True)
    return checked, skipped, bad


if __name__ == '__main__':
    out = controls()
    print('controls', out, 'all SAT:', all(x[1] == 'sat' for x in out), flush=True)
    print('real c=8 solutions (checked, outside branch ii, problems):', real_check(), flush=True)
