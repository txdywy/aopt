"""Concrete-prefix collision and inverse scans for remaining joint DIV rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import z3

from codex.run_middle_residual_enum863 import run
from codex.screen_middle_division_residual863 import pool
from codex.search_early_boundary863 import legal_point
from codex.search_focused_joint_boundary863 import edge_points
from codex.search_joint_live_boundary863 import inputs, wanted, NODE, MASK
from codex.audit_one_high_tail863 import information_fibre
from codex.refine_joint_quotient_range863 import target_quotient_maximum


def bypass(name, v):
    if name == 'u_xor_mix':
        return v[1]^v[7]
    if name == 'u_b_plus_s19':
        return (v[1]*v[3]+v[6]) & MASK
    if name == 'z_or_z_square_plus_b':
        return v[0] | ((v[0]*v[0]+v[3]) & MASK)
    raise ValueError(name)


def dataset():
    points = list(dict.fromkeys([*pool(), *edge_points()]))
    cases = []
    for rank, name in enumerate(('u_xor_mix','u_b_plus_s19','z_or_z_square_plus_b')):
        witnesses = {}
        for b in points:
            v = inputs(b)
            witnesses.setdefault((wanted(v, NODE)-bypass(name, v)) & MASK, b)
        rs = list(witnesses)
        random.Random(8642945+rank).shuffle(rs)
        cases.append(dict(id=rank, width=32, source_name=name, residuals=rs,
                          primary_witnesses=[witnesses[r] for r in rs]))
    return dict(node=NODE, input_names=['z','u','v','b','prior','a','a_shr19','a_xor_c1'],
                points=points, cases=cases,
                definition='residual=target-computed_bypass modulo2^32; separate definitions in audit_joint_quotient_bypass863.py')


def row_families():
    out = []
    for op in ('//','%','cdiv'):
        out += [dict(order=['mad','^',op,'mad'], wiring=[[8,8,0],[1,7],[8],[10,11,9]], family='u_xor_mix'),
                dict(order=['mad','mad',op,'mad'], wiring=[[4,4,5],[1,3,6],[8],[10,11,9]], family='u_b_plus_s19'),
                dict(order=['mad',op,'mad','^'], wiring=[[3,6,3],[8],[9,10,11],[0,10]], family='numerator_collision')]
    out.append(dict(order=['mad','|','cdiv','mad'], wiring=[[0,0,3],[0,8],[8],[9,11,10]], family='scaled_z_or_z_square_plus_b'))
    out += [dict(order=['mad',op,'mad','^'],wiring=[[2,8,3],[8],[8,8,9],[10,11]],
                 family='square_tail_zero_v') for op in ('//','%','cdiv')]
    out += [dict(order=['mad',op,'&','mad'],wiring=[[3,8,6],[8],[4,10],[9,11,10]],
                 family='masked_prior') for op in ('//','%','cdiv')]
    return out


def collision_certificate():
    # SMT found this pair; the certificate replays only concrete arithmetic
    # and the unmodified legal-input/frozen-reference construction.
    points = [36240888, 1657121264]
    vv = [inputs(b) for b in points]
    nums = [(v[3]*(v[6]+1)) & MASK for v in vv]
    residuals = [wanted(v,NODE)^v[0] for v in vv]
    assert nums[0] == nums[1] and residuals[0] != residuals[1]
    return dict(primary_b=points, numerators=nums, residuals=residuals,
                numerator='b*(a_shr19+1) modulo2^32', residual='Y XOR z',
                proof='Equal numerators give equal q for every prepared divisor and opcode. '
                'The required affine image G*q+H would then be equal, contradicting the two residuals.',
                legal_points=[legal_point(b) for b in points])


def rectangle_certificate():
    points = [3956866163,3956931699,3957128307,3957193843]
    vv = [inputs(b) for b in points]
    p = [(v[0]*v[0]+v[3]) & MASK for v in vv]
    h = [v[0]|t for v,t in zip(vv,p)]
    yy = [wanted(v,NODE) for v in vv]
    delta = lambda xs:(xs[3]+xs[0]-xs[1]-xs[2]) & MASK
    assert delta(p) == delta(h) == 0
    d = delta(yy);distance = min(d,(-d)&MASK)
    small = []
    for divisor in range(1,5):
        residual = (divisor*d)&MASK
        dist = min(residual,(-residual)&MASK)
        assert dist > 2*(divisor-1)
        small.append(dict(divisor=divisor,distance=dist,residual_bound=2*(divisor-1)))
    bound = (MASK+4)//5
    assert distance > 2*bound
    rem = information_fibre()['prepared_remainder_divisor_max']-1
    assert distance > 2*rem
    return dict(primary_b=points,numerators=p,scaled_bypasses=h,targets=yy,
                target_mixed=d,target_distance=distance,small_divisors=small,
                quotient_bound_divisor_at_least5=bound,remainder_bound=rem,
                proof='Y=G*h+q, with Delta(h)=Delta(P)=0. For floor/ceil, D*Delta(Y) '
                'is a signed mixed difference of remainders0..D-1, excluding D1..4. '
                'For D>=5, Delta(Y)=Delta(q) has distance<=2*ceil(MASK/5), a contradiction. '
                'REM uses the independent divisor information bound. G is arbitrary.',
                legal_points=[legal_point(b) for b in points])


def square_tail_certificate():
    base = ((-0xACCF6200//512)*pow(33,-1,1 << 23)) % (1 << 23)
    points = [base+(i << 23) for i in range(512)]
    k = z3.BitVec('square_zero_v_output_key',32)
    solver = z3.SolverFor('QF_BV');solver.set(timeout=5000)
    bound = target_quotient_maximum('cdiv')
    assert bound == 0x55555555
    targets = []
    for b in points:
        v = inputs(b);assert v[2] == 0
        y = wanted(v,NODE);targets.append(y)
        q = (z3.BitVecVal(y,32)^k)-((b*b)&MASK)
        solver.add(z3.ULE(q,bound))
    status = solver.check()
    assert status == z3.unsat, status
    return dict(primary_b=points,targets=targets,quotient_maximum=bound,status='unsat',
                proof='Here v=0, so t=v*A+b=b for every A. Y=(t*t+q) XOR K forces '
                'q=(Y XOR K)-b*b modulo2^32. Even with an independent q per sample in '
                '0..0x55555555, no common output K exists. DIV/CDIV divisor2 is separately '
                'excluded by bit0 dependency; divisor>=3 obeys this bound. REM is smaller.',
                legal_points=[legal_point(b) for b in points])


def mask_rectangle_certificate():
    points = [2558187825,3095059761,410700081,947572017]
    vv = [inputs(b) for b in points]
    yy = [wanted(v,NODE) for v in vv]
    delta = lambda xs:(xs[3]+xs[0]-xs[1]-xs[2]) & MASK
    priors = [v[4] for v in vv]
    h,k = priors[0]^priors[1],priors[0]^priors[2]
    assert not h&k and priors[0]^priors[1]^priors[2]^priors[3] == 0
    assert delta(points) == delta([v[6] for v in vv]) == 0
    assert delta(yy) == 1 << 31
    for bit in range(32):
        assert delta([p & (1 << bit) for p in priors]) == 0
    # This is a symbolic check over ALL masks, separate from the numeric
    # bit-coordinate verification above.
    mask = z3.BitVec('masked_prior_rectangle_mask',32)
    s = z3.SolverFor('QF_BV')
    s.add(((priors[3]&mask)+(priors[0]&mask)-(priors[1]&mask)-(priors[2]&mask)) != 0)
    assert s.check() == z3.unsat
    witness = [inputs(b) for b in (0,65536)]
    assert (witness[0][4]^witness[1][4]) & 1 == 0
    assert (wanted(witness[0],NODE)^wanted(witness[1],NODE)) & 1 == 1
    qmax = (MASK+4)//5
    assert 2*qmax < 1 << 31
    rmax = information_fibre()['prepared_remainder_divisor_max']-1
    assert 2*rmax < 1 << 31
    return dict(primary_b=points,prior_values=priors,targets=yy,target_mixed=1 << 31,
                prior_axis_masks=[h,k],numerator_coordinate_mixed=[0,0],
                mask_mixed_symbolically_zero=True,prepared_G_must_be_odd=True,
                divisor_at_least5_quotient_maximum=qmax,remainder_maximum=rmax,
                scope='Y=G*op(P,D)+(prior & L), P=A*b+a_shr19 modulo2^32; '
                'A/G/L/D root-only arbitrary uint32, D nonzero, op floor/ceil/REM.',
                proof='Even G fails the b^65536 bit0 witness. For odd G put B=G^-1. '
                'The rectangle has Delta(prior & L)=0 for EVERY L and Delta(P)=0 for EVERY A. '
                'Thus Delta(q)=B*Delta(Y)=2^31. For floor/ceil D>=5 the quotient range '
                'bounds the circular distance by2*ceil(MASK/5)<2^31. For D1/3, '
                'D*Delta(q)=2^31 cannot be a mixed remainder difference bounded by2*(D-1). '
                'D2/4 cannot transmit b16 to bit0: numerator low16 and all initial low16 '
                'are fixed, and quotient bit0 depends only on numerator low2/low3. '
                'REM is bounded by the478-state divisor limit and has the same range contradiction.',
                legal_points=[legal_point(b) for b in points])


def audit(enum_path):
    raw = enum_path.read_bytes()
    scan = json.loads(raw)
    data = dataset()
    assert scan['dataset'] == data and scan['node'] == NODE
    assert scan['scanner_source_sha256'] == hashlib.sha256(Path('codex/enumerate_middle_residual863.cpp').read_bytes()).hexdigest()
    assert scan['complete'] and {r['id'] for r in scan['results']} == {0,1,2}
    names = set()
    for row in scan['results']:
        assert row['complete'] and row['width'] == 32 and row['evaluated_odd_parameters'] == 1 << 31
        assert not row['models'] and not row['bounded_models']
        names.add(row['source_name'])
    assert names == {'u_xor_mix','u_b_plus_s19','z_or_z_square_plus_b'}
    parity = []
    for name in sorted(names):
        points = [0, 65536]
        vv = [inputs(b) for b in points]
        assert (bypass(name,vv[0])^bypass(name,vv[1])) & 1 == 0
        assert (wanted(vv[0],NODE)^wanted(vv[1],NODE)) & 1 == 1
        parity.append(dict(family=name, primary_b=points, prepared_G_must_be_odd=True))
    collision = collision_certificate()
    rectangle = rectangle_certificate()
    square = square_tail_certificate()
    masked = mask_rectangle_certificate()
    rows = row_families()
    return dict(node=NODE, scope='Exactly the sixteen recorded rows and the separately stated masked-prior family. Additionally, arbitrary one-high q '
                'in Y=G*q+h for the three listed fixed computed bypasses h. G is root-only. '
                'No other root, general mask-dependent bypass, general four-op grammar or executable lower bound is claimed.',
                closed_rows=rows, counts=dict(inverse_parameter_rows=6,numerator_collision_rows=3,
                                             fixed_bypass_rectangle_rows=1,zero_factor_rows=3,masked_prior_rows=3),
                inverse_scan=dict(path=str(enum_path), sha256=hashlib.sha256(raw).hexdigest(),
                                  evaluated_odd_parameters=3*(1 << 31), all_one_high_excluded=True),
                odd_coefficient_witnesses=parity, numerator_collision=collision,
                scaled_bypass_rectangle=rectangle,square_tail_zero_factor=square,
                masked_prior_rectangle=masked,
                legal_scan_points=[legal_point(b) for b in data['points']],
                proof='If G is even the fixed bypass bit0 cannot reproduce target variation under b^65536. '
                'For odd G, q=G^-1*(Y-h). The scanner exhausts every odd inverse and finds two distinct '
                'q>2^31 on the recorded distinct residuals. This excludes any arbitrary one-high q, '
                'including the prepared divisions after the established T-only/REM information exclusions.',
                new_executable=False, best_executable_cycles=864)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary', type=Path)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--enum-output', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.binary:
        run(args.binary, dataset(), args.enum_output, args.workers)
    result = audit(args.enum_output)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Computed bypass and collision certificates:', len(result['closed_rows']), 'rows', flush=True)


if __name__ == '__main__':
    main()
