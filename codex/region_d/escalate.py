"""Escalate low-width SAT candidates of region D.

A width-c SAT only says some constants match the low c output bits for every
input. Re-solve the same wiring and shift band with more exact output bits.
UNSAT at any larger c excludes the wiring for that band (it is a superset of
the constraints). A candidate that survives c = 15 is reported for full-width
treatment (beyond bit 15 the four absorbable output keys differ).
"""
import glob
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/user/aopt')
from rd3 import solve

LADDER = (11, 13, 15)


def escalate(order, wiring, band, seconds):
    steps = []
    for c in LADDER:
        t = time.time()
        r = solve(tuple(order), tuple(tuple(x) for x in wiring), band[0], band[1], c, seconds)
        steps.append([c, r['result'], round(time.time() - t, 2)])
        if r['result'] == 'unsat':
            return 'closed', steps, None
        if r['result'] != 'sat':
            return 'undecided', steps, None
        last = r
    return 'survives_15', steps, last


def main():
    out = sys.argv[1]
    pattern = sys.argv[2] if len(sys.argv) > 2 else 'cs_*.jsonl'
    seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 120
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                r = json.loads(line)
                done.add((r['file'], r['i']))
            except Exception:
                pass
    with open(out, 'a') as f:
        for fn in sorted(glob.glob(pattern)):
            for line in open(fn):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get('result') != 'sat' or (fn, r['i']) in done:
                    continue
                status, steps, last = escalate(r['order'], r['wiring'], r['band'], seconds)
                rec = dict(file=fn, i=r['i'], order=r['order'], wiring=r['wiring'], band=r['band'],
                           status=status, ladder=steps)
                if last is not None:
                    rec.update(consts=last.get('consts'), K=last.get('K'))
                f.write(json.dumps(rec) + '\n')
                f.flush()
                print(status, fn, r['i'], r['order'], r['wiring'], steps, flush=True)


if __name__ == '__main__':
    main()
