"""Five-op candidates jointly absorbing the C1 and C5 XORs.

The fixed target is F(x XOR C1) XOR PRE_C5. Unlike the four-op search,
the input key is not free: the old C1 instruction is really removed.
Ten explicit five-instruction families are searched, together with an
optional polynomial relaxation of the parallel family. This is a scout,
not enumeration of every five-instruction graph. Low-width UNSAT applies
only to the named family; reduced SAT is never a new kernel score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import z3

from codex.cegis_c1_absorb863 import C1, middle
from codex.search_cross_encoding863 import PRE_C5, necessary_inputs


def body(x, c, family, mask):
    if family in ('input_mixed', 'masked_input'):
        u = x ^ c['K'] if family == 'input_mixed' else x & c['K']
        p, q = (c['A']*x+c['B']) & mask, (c['C']*u+c['D']) & mask
        return (c['G']*(p ^ q)+c['H']) & mask
    if family in ('input_feedback', 'input_product'):
        u = x ^ c['K']
        p, q = (c['A']*u+c['B']) & mask, (c['C']*u+c['D']) & mask
        return ((c['G']*(p ^ q)+x) if family == 'input_feedback' else x*(p ^ q)+c['H']) & mask
    if family in ('parallel', 'masked_parallel'):
        p = (c['A']*x+c['B']) & mask
        q = (c['C']*x+c['D']+c.get('Q', 0)*x*(x-1)) & mask
        v = p ^ q ^ c['K'] if family == 'parallel' else p ^ (q & c['K'])
        return (c['G']*v+c['H']) & mask
    if family in ('serial', 'masked_serial'):
        t = (c['A']*x+c['B']) & mask
        u = t ^ c['K'] if family == 'serial' else t
        v = u if family == 'serial' else t & c['K']
        return (c['G']*(u ^ ((c['C']*v+c['D']) & mask))+c['H']) & mask
    p = (c['A']*x+c['B']) & mask
    q = (c['C']*x+c['D']) & mask
    u = q ^ c['K'] if family == 'branched' else q & c['K']
    return p ^ ((c['G']*u+c['H']) & mask)


def normalize(coefficients, family, width):
    """Normalize constants by identities, without altering the fixed input."""
    c = dict(coefficients)
    mask, high = (1 << width)-1, 1 << (width-1)
    if family in ('input_mixed', 'masked_input', 'input_feedback', 'input_product'):
        if family != 'masked_input':
            arms = ('C',) if family == 'input_mixed' else ('A','C')
            if c['K'] & 1:
                c['K'] ^= mask
                for a in arms:
                    b = 'B' if a == 'A' else 'D'
                    c[b] -= c[a]
                    c[a] = -c[a]
            if c['K'] & high:
                c['K'] ^= high
                for a in arms:
                    c['B' if a == 'A' else 'D'] -= c[a]*high
        if c['B'] & 1:
            c['A'], c['B'] = -c['A'], -c['B']-1
            c['C'], c['D'] = -c['C'], -c['D']-1
        if c['B'] & high:
            c['B'] ^= high
            c['D'] ^= high
        return {k: v & mask for k,v in c.items()}
    if family == 'parallel':
        if c['B'] & 1:
            c['A'], c['B'] = -c['A'], -c['B']-1
            c['C'], c['D'] = -c['C'], -c['D']-1
            c['Q'] = -c.get('Q', 0)
        if c['B'] & high:
            c['B'] += high
            c['D'] += high
        if c['K'] & 1:
            c['K'] ^= mask
            c['H'] -= c['G']
            c['G'] = -c['G']
        if c['K'] & high:
            c['K'] ^= high
            c['H'] -= c['G']*high
        # x*(x-1) is even, so its coefficient is defined modulo high.
        c['Q'] = c.get('Q', 0) & (high-1)
    elif family == 'serial':
        # Complementing u complements both XOR arms after D -> C-D-1.
        for n in ('B', 'K'):
            if c[n] & 1:
                if n == 'B':
                    c['A'], c['B'] = -c['A'], -c['B']-1
                else:
                    c['K'] ^= mask
                c['D'] = c['C']-c['D']-1
            if c[n] & high:
                c[n] ^= high
                c['H'] -= c['G']*high
    elif family == 'branched':
        # Normalize the inner XOR and its affine argument independently.
        for n in ('D', 'K'):
            if c[n] & 1:
                if n == 'D':
                    c['C'], c['D'] = -c['C'], -c['D']-1
                else:
                    c['K'] ^= mask
                c['H'] -= c['G']
                c['G'] = -c['G']
            if c[n] & high:
                c[n] ^= high
                c['H'] -= c['G']*high
        # A common complement/high flip of the final XOR arms cancels.
        if c['B'] & 1:
            c['A'], c['B'] = -c['A'], -c['B']-1
            c['G'], c['H'] = -c['G'], -c['H']-1
        if c['B'] & high:
            c['B'] ^= high
            c['H'] ^= high
    elif family == 'masked_parallel':
        # Complement only the unmasked XOR arm, then negate the outer MAD.
        if c['B'] & 1:
            c['A'], c['B'] = -c['A'], -c['B']-1
            c['H'] -= c['G']
            c['G'] = -c['G']
        if c['B'] & high:
            c['B'] ^= high
            c['H'] -= c['G']*high
    elif family == 'masked_serial':
        # (~t)&K = K-(t&K); complement both final XOR arms.
        if c['B'] & 1:
            c['A'], c['B'] = -c['A'], -c['B']-1
            c['D'] = -c['D']-1-c['C']*c['K']
        if c['B'] & high:
            c['B'] ^= high
            c['H'] -= c['G']*high
    else:
        if c['D'] & 1:
            c['C'], c['D'] = -c['C'], -c['D']-1
            c['H'] += c['G']*c['K']
            c['G'] = -c['G']
        if c['D'] & high:
            c['D'] ^= high
            if c['K'] & high:
                c['H'] -= c['G']*high
        if c['B'] & 1:
            c['A'], c['B'] = -c['A'], -c['B']-1
            c['G'], c['H'] = -c['G'], -c['H']-1
        if c['B'] & high:
            c['B'] ^= high
            c['H'] ^= high
    return {k: v & mask for k, v in c.items()}


def normalization_controls():
    rng = random.Random(8642805)
    counts = {}
    for family in FAMILIES:
        for _ in range(100):
            c = {n: rng.randrange(64) for n in ('A','B','C','D','G','H','K','Q')}
            if family in ('parallel', 'serial', 'masked_parallel', 'masked_serial'):
                c['G'] |= 1
            if family == 'parallel':
                c['C'] = (c['C'] & ~1) | ((c['A'] ^ 1) & 1)
                c['Q'] &= ~1
            elif family == 'serial':
                c['A'] |= 1
                c['C'] &= ~1
            elif family in ('branched', 'input_mixed'):
                c['A'] = (c['A'] & ~1) | ((c['G']*c['C'] ^ 1) & 1)
                if family == 'input_mixed':
                    c['G'] |= 1
                    c['A'] = (c['A'] & ~1) | ((c['C'] ^ 1) & 1)
            elif family == 'input_feedback':
                c['G'] |= 1
                c['C'] = (c['C'] & ~1) | (c['A'] & 1)
                c['H'] = 0
            elif family == 'input_product':
                pass
            else:
                c['A'] |= 1
                c['C'] &= ~1
            if family == 'masked_parallel':
                c['Q'] = 0
            n = normalize(c, family, 6)
            assert not n['B'] & 33
            if not family.startswith('masked_'):
                assert not n['K'] & 33
            if family in ('branched', 'masked_branched'):
                assert not n['D'] & 33
            assert all(body(x, c, family, 63) == body(x, n, family, 63) for x in range(64)), (family,c,n)
        counts[family] = 100
    return dict(programs_per_family=counts, inputs_per_program=64,
                no_input_transform=True, exact_exposed_function_preserved=True)


FAMILIES = ('parallel', 'serial', 'branched', 'masked_parallel', 'masked_serial', 'masked_branched',
            'input_mixed', 'masked_input', 'input_feedback', 'input_product')


def macro_for(c, family):
    if family in ('input_mixed', 'masked_input', 'input_feedback', 'input_product'):
        order = ['&' if family == 'masked_input' else '^', 'mad','mad','^','mad']
        wiring = [[0,1],[0,2,3],[1,3,4],[2,3],[4,5,6]]
        constants = [[c['K'],0],[c['A'],c['B']],[c['C'],c['D']],[0,0],[c['G'],c['H']]]
        if family in ('input_feedback', 'input_product'):
            wiring[1] = [1,2,3]
            if family == 'input_feedback':
                wiring[4] = [4,5,0]
            else:
                wiring[4] = [0,4,5]
                constants[4] = [c['H'],0]
        return dict(input_key=C1,order=order,wiring=wiring,constants=constants)
    if family in ('parallel', 'masked_parallel'):
        assert c.get('Q', 0) == 0
        if family == 'parallel':
            order = ['mad','mad','^','^','mad']
            wiring = [[0,1,2],[0,2,3],[1,2],[3,4],[4,5,6]]
            constants = [[c['A'],c['B']],[c['C'],c['D']],[0,0],[c['K'],0],[c['G'],c['H']]]
        else:
            order = ['mad','mad','&','^','mad']
            wiring = [[0,1,2],[0,2,3],[2,3],[1,3],[4,5,6]]
            constants = [[c['A'],c['B']],[c['C'],c['D']],[c['K'],0],[0,0],[c['G'],c['H']]]
    elif family in ('serial', 'masked_serial'):
        order = ['mad','^','mad','^','mad']
        wiring = [[0,1,2],[1,2],[2,3,4],[2,3],[4,5,6]]
        constants = [[c['A'],c['B']],[c['K'],0],[c['C'],c['D']],[0,0],[c['G'],c['H']]]
        if family == 'masked_serial':
            order[1] = '&'
            wiring[3] = [1,3]
    else:
        order = ['mad','mad','^','mad','^']
        wiring = [[0,1,2],[0,2,3],[2,3],[3,4,5],[1,4]]
        constants = [[c['A'],c['B']],[c['C'],c['D']],[c['K'],0],[c['G'],c['H']],[0,0]]
        if family == 'masked_branched':
            order[2] = '&'
    return dict(input_key=C1, order=order, wiring=wiring, constants=constants)


def solve(family, width, seconds, control=False, quadratic=False, scale=None):
    mask, high = (1 << width)-1, 1 << (width-1)
    names = ['A','B','C','D','G','H','K']+(['Q'] if quadratic else [])
    c = dict(zip(names, z3.BitVecs(' '.join(names), width)))
    s = z3.SolverFor('QF_BV')
    s.add(c['B'] & (high | 1) == 0)
    if not family.startswith('masked_'):
        s.add(c['K'] & (high | 1) == 0)
    if family == 'parallel':
        s.add(c['G'] & 1 == 1, (c['A']+c['C']) & 1 == 1)
        if quadratic:
            s.add(c['Q'] & 1 == 0, z3.ULT(c['Q'], high))
    elif family == 'serial':
        s.add(c['A'] & 1 == 1, c['C'] & 1 == 0, c['G'] & 1 == 1)
    elif family == 'branched':
        s.add(c['D'] & (high | 1) == 0, (c['A']+c['G']*c['C']) & 1 == 1)
    elif family in ('masked_parallel', 'masked_serial'):
        # For a nondegenerate mask, triangular bijectivity forces C even.
        # C odd can be a permutation only for the all-zero/all-one mask;
        # those are separately excluded by the earlier <=4-op searches.
        s.add(c['A'] & 1 == 1, c['C'] & 1 == 0, c['G'] & 1 == 1)
    elif family == 'masked_branched':
        s.add(c['B'] & (high | 1) == 0, c['D'] & (high | 1) == 0,
              c['A'] & 1 == 1, (c['G']*c['C']) & 1 == 0)
    elif family == 'input_mixed':
        s.add(c['G'] & 1 == 1, (c['A']+c['C']) & 1 == 1)
    elif family == 'masked_input':
        s.add(c['G'] & 1 == 1, c['A'] & 1 == 1, c['C'] & 1 == 0)
    elif family == 'input_feedback':
        # The real target at zero is odd. With no output constant, G is odd;
        # triangular bijectivity then requires A+C even.
        s.add(c['G'] & 1 == 1, (c['A']+c['C']) & 1 == 0, c['H'] == 0)
    if scale is not None:
        s.add(c['G'] == (scale & mask))
    witness = dict(A=3, B=24, C=40, D=7, G=9, H=29, K=42, Q=2 if quadratic else 0)
    if family == 'input_feedback':
        witness.update(C=41,H=0)
    normalized = normalize(witness, family, width)
    if control:
        s.add(*(c[n] == normalized[n] for n in names))

    def target(x):
        return body(x, witness, family, mask) if control else middle(x ^ (C1 & mask), mask) ^ (PRE_C5 & mask)

    points = list(dict.fromkeys(list(range(32)) + necessary_inputs(width)))
    def add(p):
        xv = z3.BitVecVal(p, width)
        s.add(body(xv, c, family, mask) == target(xv))
    for p in points:
        add(p)
    if width <= 16:
        xs = np.arange(1 << width, dtype=np.uint64)
    else:
        rng = random.Random(8642806)
        xs = np.array(points+[rng.getrandbits(width) for _ in range(4096)], dtype=np.uint64)
    start = time.monotonic()
    row = dict(family=family, width=width, control=control, quadratic_relaxation=quadratic,
               fixed_scale=scale, input_key=C1, output_key=PRE_C5,
               status='unknown', events=[], new_executable=False)
    while time.monotonic()-start < seconds:
        s.set(timeout=max(1, int((seconds-time.monotonic()+start)*1000)))
        status = s.check()
        row['events'].append(dict(samples=len(points), status=str(status)))
        if status != z3.sat:
            row['status'] = str(status)
            if status == z3.unknown:
                row['reason'] = s.reason_unknown()
            break
        m = s.model()
        cv = {n: m.eval(v, model_completion=True).as_long() for n,v in c.items()}
        got, want = body(xs, cv, family, mask), target(xs)
        bad = np.flatnonzero(got != want)
        if not len(bad):
            if width < 32:
                row.update(status='relaxed_reduced_sat' if quadratic else 'reduced_sat',
                           parameters=cv, verified_inputs=len(xs))
                break
            x = z3.BitVec('joint_restore_verify', 32)
            checker = z3.SolverFor('QF_BV')
            checker.set(timeout=max(1, int((seconds-time.monotonic()+start)*1000)))
            checker.add(body(x, cv, family, mask) != target(x))
            checked = checker.check()
            if checked == z3.unsat:
                row.update(status='relaxed_proved_full32' if quadratic else 'proved_full32', parameters=cv)
                if not quadratic and not control:
                    row['macro'] = macro_for(cv, family)
                break
            if checked == z3.unknown:
                row.update(status='verification_unknown', reason=checker.reason_unknown())
                break
            bad_points = [checker.model().eval(x, model_completion=True).as_long()]
        else:
            bad_points = [int(xs[i]) for i in bad[:8]]
        for p in bad_points:
            assert p not in points
            points.append(p)
            add(p)
    row['seconds'] = round(time.monotonic()-start, 3)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--family', choices=FAMILIES, required=True)
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--widths', default='11,16,32')
    ap.add_argument('--quadratic', action='store_true')
    ap.add_argument('--scale', type=lambda s:int(s,0))
    ap.add_argument('--control', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert not args.quadratic or args.family == 'parallel'
    result = dict(normalization_controls=normalization_controls(), queries=[])
    for width in map(int, args.widths.split(',')):
        row = solve(args.family, width, args.seconds, args.control, args.quadratic, args.scale)
        result['queries'].append(row)
        print(json.dumps(row), flush=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        if row['status'] not in ('reduced_sat','relaxed_reduced_sat'):
            break


if __name__ == '__main__':
    main()
