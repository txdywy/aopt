"""Fixed three-op structures, with arbitrary input and output XOR keys.

These queries reduce selector overhead. They cover parallel affine/XOR
and nested affine/XOR/affine only; neither is the whole three-op grammar.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import z3

from codex.search_cross_encoding863 import middle


def evaluate(x, coefficients, family, mask):
    a,b,d,e = coefficients[:4]
    if family == 'parallel':
        return (((a*x+b)&mask)^((d*x+e)&mask))&mask
    key = coefficients[4]
    return (d*(((a*x+b)&mask)^key)+e)&mask


def normalize_nested(coefficients, input_key, output_key, width):
    """Preserve P(x XOR I) XOR O while enforcing all nested-query cuts."""
    mask, high = (1 << width)-1, 1 << (width-1)
    a,b,d,e,key = coefficients
    if width < 2 or not (a & 1 and d & 1):
        raise ValueError('odd affine multipliers and width >= 2 required')
    ik,ok = input_key,output_key
    if ik & 1:
        ik ^= mask
        a,b = -a,b-a
    if ik & high:
        ik ^= high
        b += high
    if ok & 1:
        ok ^= mask
        d,e = -d,-e-1
    if ok & high:
        ok ^= high
        e += high
    if key & 1:
        key ^= mask
        d,e = -d,e-d
    if key & high:
        key ^= high
        e -= high
    if b & high:
        b ^= high
        e -= high
    return [v & mask for v in (a,b,d,e,key)],ik & mask,ok & mask


def search(family='parallel', width=11, seconds=25, check_ms=22000, control=False):
    if family not in ('parallel','nested') or width not in (*range(2,13),32):
        raise ValueError('parallel/nested, width 2..12 or 32 required')
    if seconds <= 0 or check_ms <= 0:
        raise ValueError('positive time limits required')
    mask = (1 << width)-1
    a,b,d,e,key,ik,ok = z3.BitVecs('afx_a afx_b afx_d afx_e afx_key afx_ik afx_ok',width)
    solver = z3.SolverFor('QF_BV')
    solver.add(z3.Extract(width-1,width-1,ik) == 0)

    def reference(x):
        if not control:
            return middle(x)
        if family == 'parallel':
            return (21*x+19)^(128*x+34)
        return 5*((21*x+19)^42)+23

    if family == 'parallel':
        solver.add(a & 1 == 1,d & 1 == 0,b & 1 == 0)
        # x=0 eliminates the arbitrary output XOR key exactly.
        outkey = (b ^ e) ^ reference(ik)
        coefficients = [a,b,d,e]
        expr = lambda x: (a*x+b)^(d*x+e)
    else:
        solver.add(a & 1 == 1,d & 1 == 1,
            z3.Extract(width-1,width-1,key) == 0,
            z3.Extract(width-1,width-1,ok) == 0,
            z3.Extract(width-1,width-1,b) == 0,
            # Complementing the input, output, or inner key is absorbed
            # by the adjacent arbitrary affine map. Highest-bit offsets
            # restore the existing top-bit normalizations independently.
            ik & 1 == 0, ok & 1 == 0, key & 1 == 0)
        outkey = ok
        offset = (reference(ik)^ok)-d*(b ^ key)
        coefficients = [a,b,d,offset,key]
        expr = lambda x: d*((a*x+b)^key)+offset

    def constrain(p):
        x = z3.BitVecVal(p,width)
        solver.add(expr(x) == (reference(x ^ ik)^outkey))

    rng = random.Random(885)
    points = list(dict.fromkeys(p & mask for p in
        [*range(32),256,264,288,296,511,512,1023,1024,mask]
        +[rng.getrandbits(width) for _ in range(20)]))
    for p in points:
        constrain(p)
    checks = list(range(1 << width)) if width < 32 else list(dict.fromkeys(
        points+[1 << i for i in range(32)]+[rng.getrandbits(32) for _ in range(256)]))
    start = time.monotonic()
    deadline = start+seconds
    report = dict(family=family,width=width,seconds_limit=seconds,check_ms=check_ms,
        input_and_output_keys='arbitrary modulo highest-bit normalizations',
        normalizations=(['A odd','D even','B even','I highest bit zero'] if family == 'parallel'
            else ['A odd','D odd','I highest and lowest bits zero',
                  'O highest and lowest bits zero','K highest and lowest bits zero',
                  'B highest bit zero']),
        control=control,initial_samples=points[:],events=[],result='deadline',new_executable=False)
    while time.monotonic() < deadline:
        solver.set(timeout=min(check_ms,max(1,int(1000*(deadline-time.monotonic())))))
        status = solver.check()
        event = dict(samples=len(points),synthesis=str(status))
        report['events'].append(event)
        if status != z3.sat:
            report['result'] = 'projected_family_unsat' if status == z3.unsat else 'unknown'
            if status == z3.unknown:
                report['reason'] = solver.reason_unknown()
            break
        model = solver.model()
        cv = [model.eval(k,model_completion=True).as_long() for k in coefficients]
        iv,ov = [model.eval(k,model_completion=True).as_long() for k in (ik,outkey)]
        report.update(coefficients=cv,input_key=iv,output_key=ov)
        bad = [x for x in checks if evaluate(x,cv,family,mask) != ((reference(x ^ iv)^ov)&mask)][:8]
        if bad:
            event['counterexamples'] = bad
            for p in bad:
                assert p not in points
                points.append(p)
                constrain(p)
            continue
        if width < 32:
            report.update(result='equivalent_on_reduced_domain',verified_inputs=len(checks))
            break
        x = z3.BitVec('afx_full32_verify',32)
        av,bv,dv,ev = cv[:4]
        fixed = ((av*x+bv)^(dv*x+ev) if family == 'parallel' else
                 dv*((av*x+bv)^cv[4])+ev)
        verify = z3.SolverFor('QF_BV')
        verify.set(timeout=min(check_ms,max(1,int(1000*(deadline-time.monotonic())))))
        verify.add(fixed != (reference(x ^ iv)^ov))
        checked = verify.check()
        event['full32_verification'] = str(checked)
        if checked == z3.unsat:
            report['result'] = 'proved_equivalent_full32'
            break
        if checked == z3.unknown:
            report.update(result='verification_unknown',reason=verify.reason_unknown())
            break
        p = verify.model().eval(x,model_completion=True).as_long()
        event['symbolic_counterexample'] = p
        assert p not in points
        points.append(p)
        constrain(p)
    report['seconds'] = round(time.monotonic()-start,3)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--family',choices=('parallel','nested'),required=True)
    ap.add_argument('--width',type=int,default=11)
    ap.add_argument('--seconds',type=float,default=25)
    ap.add_argument('--check-ms',type=int,default=22000)
    ap.add_argument('--control',action='store_true')
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    report = search(args.family,args.width,args.seconds,args.check_ms,args.control)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report.get(k) for k in ('family','width','result','seconds','input_key','output_key')}))


if __name__ == '__main__':
    main()
