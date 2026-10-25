"""Keep four middle instructions but absorb the later restore XOR.

Unlike the earlier fixed-input-key search, the input key is arbitrary and
can be folded into the preceding C1 XOR. Low-width SAT needs lifting; only
a full32 proof could justify constructing and scheduling a new kernel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import z3

from codex.search_cross_encoding863 import PRE_C5, middle, program, necessary_inputs, normalize


def normalize_free_input(coefficients, input_key, width):
    """Preserve P(x XOR I), with I's lowest/highest bits zero and normalized arms."""
    mask, high = (1 << width)-1, 1 << (width-1)
    a,b,d,e,g,h = coefficients
    ik = input_key
    if ik & 1:
        ik ^= mask
        a,b,d,e = -a,b-a,-d,e-d
    if ik & high:
        ik ^= high
        b,e = b+a*high,e+d*high
    return normalize([a,b,d,e,g,h],width), ik & mask


def search(width=11, seconds=30, fixed_multiplier=None, output_key=PRE_C5,
           control=False):
    if width not in (*range(2,17),32) or seconds <= 0:
        raise ValueError('width 2..16 or 32 and positive seconds required')
    mask, high = (1 << width)-1, 1 << (width-1)
    a,b,d,e,g,ik = z3.BitVecs('restore_a restore_b restore_d restore_e restore_g restore_ik',width)
    solver = z3.SolverFor('QF_BV')
    # XOR arms can be swapped/complemented. Complementing x is absorbed
    # independently in both affine arms, as is an input highest-bit change.
    solver.add(a & 1 == 1, d & 1 == 0, g & 1 == 1, b & 1 == 0,
               ik & 1 == 0, ik & high == 0)
    if fixed_multiplier is not None:
        solver.add(g == (fixed_multiplier & mask))
    if control:
        # Hidden input key 42 is discoverable, not pinned. Output key is
        # deliberately present in the reference so the same target cancels it.
        reference = lambda x: middle(x ^ 42) ^ output_key
    else:
        reference = middle
    target = lambda x: reference(x ^ ik) ^ output_key
    h = target(z3.BitVecVal(0,width)) - g*(b ^ e)
    expression = lambda x: g*((a*x+b)^(d*x+e))+h
    points = necessary_inputs(width)
    for x in points:
        solver.add(expression(x) == target(z3.BitVecVal(x,width)))
    started = time.monotonic()
    result = dict(width=width,seconds_limit=seconds,output_key=output_key,
        arbitrary_input_key=True,fixed_multiplier=fixed_multiplier,control=control,
        family='G*((A*x+B) XOR (D*x+E))+H, six arbitrary coefficients unless G pinned',
        normalizations=['A odd','D even','G odd','B even','I highest and lowest bits zero'],
        initial_samples=points[:],events=[],result='deadline',new_executable=False)
    rng = random.Random(8642)
    checks = list(range(1 << width)) if width < 32 else list(dict.fromkeys(
        points+[1 << i for i in range(32)]+[rng.getrandbits(32) for _ in range(512)]))
    while time.monotonic()-started < seconds:
        solver.set(timeout=max(1,int(1000*(seconds-(time.monotonic()-started)))))
        status = solver.check()
        event = dict(samples=len(points),synthesis=str(status))
        result['events'].append(event)
        if status != z3.sat:
            result['result'] = 'projected_family_unsat' if status == z3.unsat else 'unknown'
            if status == z3.unknown:
                result['reason'] = solver.reason_unknown()
            break
        model = solver.model()
        cv = [model.eval(v,model_completion=True).as_long() for v in (a,b,d,e,g,h)]
        iv = model.eval(ik,model_completion=True).as_long()
        result.update(coefficients=cv,input_key=iv)
        bad = [x for x in checks if program(x,cv,width) != ((reference(x ^ iv)^output_key)&mask)][:8]
        if bad:
            event['counterexamples'] = bad
            for x in bad:
                assert x not in points
                points.append(x)
                solver.add(expression(x) == target(z3.BitVecVal(x,width)))
            continue
        if width < 32:
            result.update(result='equivalent_on_reduced_domain',verified_inputs=len(checks))
            break
        x = z3.BitVec('restore_full32_counterexample',32)
        aa,bb,dd,ee,gg,hh = cv
        fixed = gg*((aa*x+bb)^(dd*x+ee))+hh
        check = z3.SolverFor('QF_BV')
        check.set(timeout=max(1,int(1000*(seconds-(time.monotonic()-started)))))
        check.add(fixed != (reference(x ^ iv)^output_key))
        checked = check.check()
        event['full32_verification'] = str(checked)
        if checked == z3.unsat:
            result['result'] = 'proved_equivalent_full32'
            break
        if checked == z3.unknown:
            result.update(result='verification_unknown',reason=check.reason_unknown())
            break
        p = check.model().eval(x,model_completion=True).as_long()
        assert p not in points
        points.append(p)
        solver.add(expression(p) == target(z3.BitVecVal(p,width)))
        event['symbolic_counterexample'] = p
    result['seconds'] = round(time.monotonic()-started,3)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--width',type=int,default=11)
    ap.add_argument('--seconds',type=float,default=30)
    ap.add_argument('--fixed-multiplier',type=lambda x:int(x,0))
    ap.add_argument('--output-key',type=lambda x:int(x,0),default=PRE_C5)
    ap.add_argument('--control',action='store_true')
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    result = search(args.width,args.seconds,args.fixed_multiplier,args.output_key,args.control)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result.get(k) for k in ('width','result','seconds','input_key','coefficients')}))


if __name__ == '__main__':
    main()
