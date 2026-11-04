import random, sys
import numpy as np
sys.path.insert(0, '.')
import rdp2, test_rdq
M = (1 << 32) - 1
c = 13; cm = (1 << c) - 1
rng = random.Random(77)
o, w = ('mad', 'mad', '>>', '^'), ((0, 0, 1), (1, 2, 3), (2, 3), (3, 4))
out = []
for t in range(4):
    k2 = ((rng.getrandbits(26) | 1) << 6) & M
    s = rng.randrange(0, 8)
    consts = [(0, rng.getrandbits(32)), (k2, rng.getrandbits(32)), (s, 0), (rng.getrandbits(32), 0)]
    tgt = np.array([test_rdq.run32(o, w, consts, z) & cm for z in range(1 << c)], dtype=np.uint64)
    for _ in range(300):
        z = rng.getrandbits(32)
        assert test_rdq.run32(o, w, consts, z) & cm == tgt[z & cm]
    r = rdp2.solve(o, w, c, 300, 1, tgt)
    out.append((s, r['result'], r.get('it')))
print('c=13 controls', out, 'all SAT:', all(x[1] == 'sat' for x in out), flush=True)
