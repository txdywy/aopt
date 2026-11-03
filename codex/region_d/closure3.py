"""Independent check: no <=3-instruction {MAD,XOR,AND,OR} program computes F(z^K)^D mod 2^11.
No right shift, so width-11 UNSAT excludes the full uint32 program for every K.
Fewer instructions are covered by padding with an identity MAD (t*1+0)."""
import sys, json, time
sys.path.insert(0, '/home/user/aopt')
from itertools import product
from codex.cegis_c1_absorb863 import wirings_general
from codex.cegis_middle_c5_863 import solve
from collections import Counter
out = open('closure3.jsonl', 'a')
res = Counter(); t0 = time.time()
for o in product(('mad', '^', '&', '|'), repeat=3):
    for w in wirings_general(o):
        r = solve(o, w, 11, 60)
        res[r['result']] += 1
        out.write(json.dumps(dict(order=o, wiring=w, result=r['result'])) + '\n'); out.flush()
        if r['result'] != 'unsat':
            print('NOTE', r['result'], o, w, flush=True)
print('CLOSURE3 DONE', dict(res), '%.0fs' % (time.time() - t0), flush=True)
