"""Region D coverage accounting from all result files (resumable runs, duplicates OK).

A wiring counts as CLOSED when one exact argument excludes it for every 32-bit constant
choice and every shift amount:
  single constant shift : rdg UNSAT (all shifts); or rda/rda2/rdq UNSAT (all shifts);
                          or every band UNSAT, plus s = 0 / s >= 32 (closure3) and, for
                          pos 0, s >= 11 by Theorem C0; residual files (ca/cb/cc_res).
  single variable shift : Theorem V (b = z, or b-subprogram escape UNSAT); or computed
                          UNSAT (rdv c=8, rdvf/rdvq c=11; cvf_res).
  single k-type         : Theorem V as above; or computed UNSAT (cg_kescape, rdvf).
  multi cc / ccc        : band [1,31] UNSAT (cs_m2/cs_m3), escalations; s = 0 / s >= 32
                          reduce to programs with fewer SHRs.
  multi other           : rdg / rdvf UNSAT (cg_multi).
Everything else is OPEN and listed.
"""
import json
import os
import sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load(fn):
    out = []
    if os.path.exists(fn):
        for line in open(fn):
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def final(rec):
    """Return 'unsat' / 'sat' / 'timeout' / 'error' for one record of any driver."""
    if rec.get('rdg') == 'unsat':
        return 'unsat'
    if 'rda2_final' in rec:
        return rec['rda2_final']
    for key in ('rda', 'rda2'):
        if key in rec and rec[key]:
            last = rec[key][max(rec[key], key=int)]
            return last['result']
    if 'rdq' in rec:
        return rec['rdq']['result']
    if 'rdp2' in rec:
        return rec['rdp2']['result']
    if 'bands' in rec and rec['bands']:
        vals = [v['result'] for v in rec['bands'].values()]
        if all(v == 'unsat' for v in vals):
            return 'unsat'
        return 'sat' if 'sat' in vals else 'timeout'
    if 'result' in rec:
        return rec['result']
    return 'error'


def best(recs, key):
    """Best verdict per key: unsat > sat > timeout > error."""
    rank = {'unsat': 3, 'sat': 2, 'timeout': 1, 'error': 0}
    out = {}
    for r in recs:
        k = key(r)
        v = final(r)
        if k not in out or rank.get(v, 0) > rank.get(out[k], 0):
            out[k] = v
    return out


def single_constant():
    from drive import all_wirings
    res = {}
    # pos 0: bands [1,8], [9,10] (+ C0 for s >= 11) with escalations of band SATs
    esc = defaultdict(dict)
    for r in load('escalated.jsonl'):
        esc[r.get('file', r.get('fn'))][r['i']] = r
    for pos in (0, 1, 2):
        allw = all_wirings(1, pos)
        n = len(allw)
        verdict = {}
        if pos == 0:
            b18 = best(load('cs_p0_s1_8.jsonl'), lambda r: r['i'])
            b910 = best(load('cs_p0_s9_10.jsonl'), lambda r: r['i'])
            escal = {}
            for r in load('escalated.jsonl'):
                ok = any(x[1] == 'unsat' for x in r.get('ladder', r.get('results', [])))
                escal[(r.get('file') or r.get('fn'), r['i'])] = ok
            for i in range(n):
                v1 = b18.get(i, 'missing')
                v2 = b910.get(i, 'missing')
                if v1 == 'sat' and escal.get(('cs_p0_s1_8.jsonl', i)):
                    v1 = 'unsat'
                if v2 == 'sat' and escal.get(('cs_p0_s9_10.jsonl', i)):
                    v2 = 'unsat'
                verdict[i] = 'unsat' if (v1 == 'unsat' and v2 == 'unsat') else (v1 if v1 != 'unsat' else v2)
        else:
            verdict = best(load(f'ch_p{pos}.jsonl'), lambda r: r['i'])
            for fn in ('ca_res.jsonl', 'cb_res.jsonl', 'cc_res.jsonl', 'cq_res.jsonl'):
                for r in load(fn):
                    if r.get('pos') != pos:
                        continue
                    v = final(r) if ('rda' in r or 'rda2' in r or 'rdq' in r) else r.get('result')
                    if v == 'unsat':
                        verdict[r['i']] = 'unsat'
        c = Counter(verdict.get(i, 'missing') for i in range(n))
        res[pos] = dict(total=n, **c)
        res[pos]['open'] = sorted(i for i in range(n) if verdict.get(i) != 'unsat' and i in verdict)[:2000]
    return res


def variable():
    st = json.load(open('varshift3.json'))['stats']
    computed = best(load('cv_c8.jsonl'), lambda r: r['i'])
    for r in load('cvf_res.jsonl'):
        if r.get('result') == 'unsat':
            computed[r['i']] = 'unsat'
    c = Counter(computed.values())
    return dict(theorem_bz=st['b=z excluded (Theorem V)'], theorem_escape_unsat=st['b=t_i excluded (Theorem V, escape UNSAT)'],
                escaping=46788, computed=dict(c), open=sorted(i for i, v in computed.items() if v != 'unsat'))


def ktype():
    k = best(load('cg_kescape.jsonl'), lambda r: r['i'])
    for r in load('ck_res.jsonl'):
        if r.get('result') == 'unsat':
            k[r['i']] = 'unsat'
    c = Counter(k.values())
    return dict(theorem_bz=71580, theorem_escape_unsat=20538, escaping=16632, computed=dict(c),
                open=sorted(i for i, v in k.items() if v != 'unsat'))


def multi():
    out = {}
    for name, fn, fix, tot in (('cc', 'cs_m2_s1_31.jsonl', 'cm2_res.jsonl', 6534),
                               ('ccc', 'cs_m3_s1_31.jsonl', 'cm3_res.jsonl', 116)):
        b = best(load(fn), lambda r: r['i'])
        for r in load(fix):
            if r.get('result') == 'unsat':
                b[r['i']] = 'unsat'
        out[name] = dict(total=tot, **Counter(b.values()))
        out[name + '_open'] = sorted(i for i, v in b.items() if v != 'unsat')
    g = best(load('cg_multi.jsonl'), lambda r: r['i'])
    for r in load('cm_res.jsonl'):
        if r.get('result') == 'unsat':
            g[r['i']] = 'unsat'
    out['other'] = dict(total=113866, **Counter(g.values()))
    out['other_open'] = sorted(i for i, v in g.items() if v != 'unsat')
    return out


if __name__ == '__main__':
    rep = dict(single_constant=single_constant(), variable=variable(), ktype=ktype(), multi=multi())
    json.dump(rep, open('coverage2.json', 'w'), indent=1, default=list)
    for k, v in rep.items():
        if k == 'single_constant':
            for pos, d in v.items():
                print('constant pos', pos, {a: b for a, b in d.items() if a != 'open'}, 'open sample', d['open'][:8])
        else:
            print(k, {a: b for a, b in v.items() if a != 'open'}, 'open', len(v.get('open', [])) if isinstance(v, dict) and 'open' in v else '')
