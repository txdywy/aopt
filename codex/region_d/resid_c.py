"""Residual closer for single constant-shift wirings left open in ch_p1/ch_p2 (and the
cs_p2 band leftovers): escalating budgets and case splits.

  rda2-applicable (affine or bitwise path): each fiber case at c=11 with SECONDS; a case
      that times out is split (rda2.split_plan) and each part gets SECONDS; a SAT case is
      re-run at c=13 then c=15 (all three cases at the new width).
  rdq-applicable (w*w + w): rdq.solve_all, every e separately.
  otherwise: exact rdg at c=11 with the shift literal fixed, one query per s in [0, 32]
      (s = 32 stands for all s >= 32: the literal is constrained to >= 32).
UNSAT is recorded only when every part is UNSAT. usage: resid_c.py OUT.jsonl SECONDS
"""
import json
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import z3
import rda2
import rdq
import rdp2
import rdg
from coverage2 import load, final


def residuals():
    fixed = set()
    for fn in ('ca_res.jsonl', 'cb_res.jsonl', 'cc_res.jsonl', 'cq_res.jsonl'):
        for r in load(fn):
            if final(r) == 'unsat' or r.get('result') == 'unsat':
                fixed.add((r['pos'], r['i']))
    res = {}
    for pos in (1, 2):
        verdict, shape = {}, {}
        for r in load(f'ch_p{pos}.jsonl'):
            v = final(r)
            shape[r['i']] = (tuple(r['order']), tuple(tuple(x) for x in r['wiring']))
            if verdict.get(r['i']) != 'unsat':
                verdict[r['i']] = v
        for i, v in verdict.items():
            if v != 'unsat':
                res[(pos, i)] = shape[i] + (v,)
    for r in load('cs_p2_s1_8.jsonl'):
        if r.get('result') != 'unsat':
            k = (2, r['i'])
            if k not in res:
                res[k] = (tuple(r['order']), tuple(tuple(x) for x in r['wiring']), r['result'])
    return {k: v for k, v in res.items() if k not in fixed}


def run_rda2(o, w, sec):
    log = []
    for c in (11, 13, 15):
        verdicts = {}
        for kind in (1, 2, 3):
            r = rda2.solve_case(o, w, kind, c, sec)
            v = r['result']
            if v == 'timeout':
                sp = rda2.solve_split(o, w, kind, c, sec)
                v = sp['result']
                log.append((c, kind, 'split', sp['parts']))
            else:
                log.append((c, kind, v))
            verdicts[kind] = v
        if all(v == 'unsat' for v in verdicts.values()):
            return 'unsat', log
        if not any(v == 'sat' for v in verdicts.values()):
            return 'timeout', log
    return 'sat', log


def run_rdg_fixed_s(o, w, sec):
    pos = o.index('>>')
    ki = w[pos][1]
    slot = ki - pos - 1
    log = []
    verdict = 'unsat'
    for s in range(0, 33):
        if s < 32:
            ex = lambda S, ks, s=s: S.add(ks[pos][slot] == s)
        else:
            ex = lambda S, ks: S.add(z3.UGE(ks[pos][slot], 32))
        r = rdg.solve(o, w, 11, sec, extra=ex)
        log.append((s, r['result']))
        if r['result'] != 'unsat':
            verdict = 'sat' if r['result'] == 'sat' else 'timeout'
    return verdict, log


def main():
    out, sec = sys.argv[1], float(sys.argv[2])
    done = {(r['pos'], r['i']) for r in load(out) if r.get('result') == 'unsat'}
    res = residuals()
    todo = sorted(k for k in res if k not in done)
    print('residuals', len(res), 'todo', len(todo), flush=True)
    with open(out, 'a') as f:
        for key in todo:
            o, w, prev = res[key]
            t = time.time()
            if rda2.applicable(o, w):
                v, log = run_rda2(o, w, sec)
                how = 'rda2'
            elif rdq.applicable(o, w):
                x = rdq.solve_all(o, w, 11, sec)
                v, log, how = x['result'], x['parts'], 'rdq'
            elif rdp2.applicable(o, w):
                x = rdp2.solve_split(o, w, 11, sec)
                v, log, how = x['result'], x['parts'], 'rdp2'
                if v != 'unsat':
                    v2, log2 = run_rdg_fixed_s(o, w, sec)
                    v, log, how = v2, [log, log2], 'rdp2+rdg_fixed_s'
            else:
                v, log = run_rdg_fixed_s(o, w, sec)
                how = 'rdg_fixed_s'
            rec = dict(pos=key[0], i=key[1], order=o, wiring=w, prev=prev, result=v, how=how, log=log,
                       sec=round(time.time() - t, 1))
            f.write(json.dumps(rec, default=list) + '\n')
            f.flush()
            print(key, o, w, how, v, rec['sec'], flush=True)
    print('RESIDC DONE', flush=True)


if __name__ == '__main__':
    main()
