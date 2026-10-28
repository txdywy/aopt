"""Finite scalar branches for P+G*q and G*P+q, q=DIV/REM/CDIV(n,P).

On positive small n, P>n gives q=0/1/n for floor/ceil/REM. A one-MAD
default prefix would put the target in a quadratic polynomial span.
One-bitop defaults are checked by exact bit lifting of the scalar with
an arbitrary same-position Boolean-function relaxation. If all defaults
fail, a small denominator pins G to finitely many word congruences.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
from math import gcd
import json
from pathlib import Path
import time

import numpy as np

from codex.audit_division_first_t863 import source_inverse
from codex.audit_runtime_shift_core863 import base_fields, monomials
from codex.audit_computed_shift_amount863 import zero_points
from codex.screen_xor_mad_polynomial import consistent
from codex.screen_division_xor_affine863 import quotient
from codex.search_early_boundary863 import MASK, NAMES, NODE, early_inputs, wanted


def default_q(n, op):
    return 0 if op == '//' or n == 0 else 1 if op == 'cdiv' else n


def scalar_roots(coefficient, rhs, odd=False):
    assert 0 < coefficient <= MASK
    common = gcd(coefficient, 1 << 32); rhs &= MASK
    if rhs % common:
        return []
    modulus = (1 << 32)//common
    base = ((rhs//common)*pow(coefficient//common, -1, modulus)) % modulus
    return [base+k*modulus for k in range(common) if not odd or (base+k*modulus) & 1]


def partial_bit_fits(values, targets, widths):
    """Exact one-bitop interpolation under per-sample low-bit constraints."""
    assert len(values) == len(targets) == len(widths)
    masks = [(1 << w)-1 for w in widths]
    fits = []
    for family in ('constant', '^K', '&K', '|K'):
        for source in ((None,) if family == 'constant' else range(9)):
            on = off = 0; valid = True
            for v, y, mask in zip(values, targets, masks):
                y &= mask; x = v[source] & mask if source is not None else 0
                if family == 'constant': need_on, need_off = y, (~y) & mask
                elif family == '^K':
                    key = x ^ y; need_on, need_off = key, (~key) & mask
                elif family == '&K':
                    valid &= y & ~x == 0; need_on, need_off = y, x & ~y & mask
                else:
                    valid &= x & ~y == 0; need_on, need_off = y & ~x & mask, (~y) & mask
                on |= need_on; off |= need_off
            if valid and on & off == 0:
                fits.append(dict(op=family, source=source, key=on))
    for i in range(9):
        for j in range(i, 9):
            for op in ('^', '&', '|'):
                valid = True
                for v, y, mask in zip(values, targets, masks):
                    p = v[i]^v[j] if op == '^' else v[i]&v[j] if op == '&' else v[i]|v[j]
                    if (p & mask) != (y & mask): valid = False; break
                if valid: fits.append(dict(op=op, operands=[i, j]))
    return fits


@lru_cache(maxsize=9)
def data(source):
    if source == 3:
        base = zero_points()[NAMES[source]] & ((1 << 23)-1)
        bs = [base | (i << 23) for i in range(512)]
    elif source == 7:
        base = zero_points()[NAMES[source]]; bs = [base, base ^ 65536]
    else:
        bs = [source_inverse(source, n) for n in range(129)]
    vv = [early_inputs(b) for b in bs]; yy = [wanted(v) for v in vv]
    nn = [v[source] for v in vv]
    matrix = [monomials(base_fields(b, v)) for b, v in zip(bs, vv)]
    return bs, vv, yy, nn, matrix


def signature_pairs(values):
    result = []
    for bit in range(32):
        seen = {}; pairs = []
        for index, v in enumerate(values):
            key = sum(((x >> bit) & 1) << j for j, x in enumerate(v))
            if key in seen: pairs.append((seen[key], index))
            else: seen[key] = index
        result.append(pairs)
    return result


def lift_scalar(values, targets, nn, op, family, cap=65536):
    """Exact low-bit lifting for a necessary arbitrary-bitwise prefix."""
    pairs = signature_pairs(values)
    aa = np.array([default_q(n, op) for n in nn], dtype=np.uint64)
    yy = np.array(targets, dtype=np.uint64)
    branches = []
    if family == 'plus':
        settings = [(0, 32, None)]  # G is odd, as required by target b16 parity.
    else:
        rr = [(y-default_q(n, op)) & MASK for y, n in zip(targets, nn)]
        nonzero = [r for r in rr if r]
        if not nonzero:
            return dict(status='necessary_survivor', zero_scalar_allowed=True, branches=[])
        valuation = min((r & -r).bit_length()-1 for r in nonzero)
        settings = [(v, 32-v, np.array([r >> v for r in rr], dtype=np.uint64))
                    for v in range(valuation+1)]
    for valuation, width, rhs in settings:
        candidates = np.array([1], dtype=np.uint64); history = []
        for bit in range(width):
            if bit:
                candidates = np.concatenate([candidates, candidates | np.uint64(1 << bit)])
            for i, j in pairs[bit]:
                with np.errstate(over='ignore'):
                    left = yy[i]-candidates*aa[i] if family == 'plus' else candidates*rhs[i]
                    right = yy[j]-candidates*aa[j] if family == 'plus' else candidates*rhs[j]
                keep = ((left ^ right) & np.uint64(1 << bit)) == 0
                candidates = candidates[keep]
                if not len(candidates): break
            history.append(dict(bit=bit, remaining=len(candidates)))
            if not len(candidates) or len(candidates) > cap: break
        finished = not len(candidates) or len(history) == width
        survivors = []
        if finished and len(candidates):
            for coefficient in candidates:
                scalar = int(coefficient) if family == 'plus' else (1 << valuation)*pow(int(coefficient), -1, 1 << width)
                if family == 'plus':
                    pp = [(y-scalar*default_q(n, op)) & MASK for y, n in zip(targets, nn)]
                    widths = [32]*len(pp)
                else:
                    pp = [(int(coefficient)*int(r)) & ((1 << width)-1) for r in rhs]
                    widths = [width]*len(pp)
                fits = partial_bit_fits(values, pp, widths)
                if fits: survivors.append(dict(scalar=scalar, fits=fits))
        branches.append(dict(valuation=valuation, width=width, history=history, complete=finished,
                             relaxed_survivors=len(candidates), survivors=survivors))
    excluded = all(r['complete'] and not r['survivors'] for r in branches)
    return dict(status='excluded' if excluded else 'open', branches=branches,
                necessary_arbitrary_bitwise_relaxation=True)


def branch_scalars(nn, yy, op, family):
    exceptions = defaultdict(lambda:defaultdict(set))
    for index, (n, y) in enumerate(zip(nn, yy)):
        assert n > 0 and y > n
        for p in range(1, n+1):
            q = quotient(n, p, op)
            coefficient, rhs = (q, y-p) if family == 'plus' else (p, y-q)
            if not coefficient:
                assert rhs & MASK  # q=0 would require y=p<=n, disallowed by selection.
                continue
            for scalar in scalar_roots(coefficient, rhs, odd=family == 'plus'):
                exceptions[scalar][index].add(p)
    return exceptions


def constrain_default(rhs, coefficient):
    """Solve coefficient*P=rhs modulo2^32, returning forced low bits."""
    common = gcd(coefficient, 1 << 32)
    if rhs % common: return None
    width = 32-(common.bit_length()-1)
    if not width: return 0, 0
    value = ((rhs//common)*pow(coefficient//common, -1, 1 << width)) & ((1 << width)-1)
    return value, width


def check_case(source, op, family, cap=65536):
    started = time.monotonic(); bs, vv, yy, nn, matrix = data(source)
    if source in (3, 7):
        assert not any(nn)
        poly = consistent(matrix, yy, 32)
        bit = (dict(status='excluded') if not partial_bit_fits(vv, yy, [32]*len(yy)) else dict(status='open')) if family == 'plus' else lift_scalar(vv, yy, nn, op, family, cap)
        return dict(source=source, source_name=NAMES[source], op=op, family=family,
                    zero_subspace=True, primary_b=bs, polynomial_default_consistent=poly,
                    bit_default=bit, mad_excluded=not poly, bit_excluded=bit['status'] == 'excluded',
                    seconds=round(time.monotonic()-started, 6))
    selected = [i for i, n in enumerate(nn) if n > 0 and yy[i] > n]
    pos_n = [nn[i] for i in selected]; pos_y = [yy[i] for i in selected]
    assert selected and pos_n[0] == 1
    default_poly = consistent([matrix[i] for i in selected], pos_y, 32)
    assert not default_poly  # Every default correction is a constant or initial numerator.
    # Include n=0 in the bitwise necessary systems; the ceil default there
    # is0, while its polynomial default argument uses positive n only.
    initial = [0, *selected]
    bit_default = lift_scalar([vv[i] for i in initial], [yy[i] for i in initial],
                              [nn[i] for i in initial], op, family, cap)
    candidates = branch_scalars(pos_n, pos_y, op, family)
    poly_cache = {}; mad_open = []; bit_open = []; direct_dead = 0
    pairs = signature_pairs([vv[i] for i in initial]); bit_filter_dead = 0
    # Each scalar can only use its recorded small-P matches. At all other
    # inputs the default relation is mandatory, enabling independent checks.
    for scalar, small in candidates.items():
        default_indices = [i for i in range(len(selected)) if i not in small]
        key = tuple(sorted(small))
        if key not in poly_cache:
            poly_cache[key] = consistent([matrix[selected[i]] for i in default_indices],
                                         [pos_y[i] for i in default_indices], 32)
        if poly_cache[key]: mad_open.append(scalar)
        values = []; pp = []; widths = []; dead = False
        for local in [None, *default_indices]:
            index = 0 if local is None else selected[local]
            n, y = nn[index], yy[index]; a = default_q(n, op)
            if family == 'plus':
                value, width = (y-scalar*a) & MASK, 32
                if local is not None and value <= n: dead = True; break
            else:
                solved = constrain_default((y-a) & MASK, scalar)
                if solved is None: dead = True; break
                value, width = solved
                if width == 32 and local is not None and value <= n: dead = True; break
            values.append(vv[index]); pp.append(value); widths.append(width)
        if dead:
            direct_dead += 1
            if scalar in mad_open: mad_open.remove(scalar)
            continue
        # A same-position Boolean-function test is stronger as a relaxation
        # and much cheaper than enumerating full first-bitop forms here.
        possible = True
        for bit, group in enumerate(signature_pairs(values)):
            if any(bit < widths[i] and bit < widths[j] and ((pp[i] ^ pp[j]) >> bit) & 1
                   for i, j in group): possible = False; break
        if not possible:
            bit_filter_dead += 1; continue
        fits = partial_bit_fits(values, pp, widths)
        if fits: bit_open.append(dict(scalar=scalar, necessary_fits=fits))
    return dict(source=source, source_name=NAMES[source], op=op, family=family,
                selected_numerators=pos_n, primary_b=[bs[i] for i in initial],
                polynomial_default_consistent=default_poly, bit_default=bit_default,
                finite_scalars=len(candidates), distinct_default_subsets=len(poly_cache),
                direct_dead=direct_dead, arbitrary_bitwise_filter_dead=bit_filter_dead,
                mad_branch_survivors=mad_open, bit_branch_survivors=bit_open,
                mad_excluded=not mad_open,
                bit_excluded=bit_default['status'] == 'excluded' and not bit_open,
                seconds=round(time.monotonic()-started, 6),
                survivors_are_necessary_not_executable=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--family', choices=('plus', 'times', 'both'), default='both')
    ap.add_argument('--cap', type=int, default=65536)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args(); rows = []
    families = ('plus', 'times') if args.family == 'both' else (args.family,)
    for source in range(9):
        for op in ('//', 'cdiv', '%'):
            for family in families:
                row = check_case(source, op, family, args.cap); rows.append(row)
                print(NAMES[source], op, family, 'G', row.get('finite_scalars', 0),
                      'MAD', row['mad_excluded'], 'BIT', row['bit_excluded'], row['seconds'], flush=True)
                result = dict(node=NODE, results=rows, complete=False, best_executable_cycles=864, new_executable=False)
                args.output.write_text(json.dumps(result, indent=2)+'\n')
    result.update(complete=True,
                  scope='P one MAD/bitop, initial numerator n, computed nonzero denominator P, final P+G*q (plus) or G*P+q (times), arbitrary prepared scalar. MAD defaults use quadratic relaxation; bit defaults use complete scalar lifting unless a recorded cap remains. Finite small-denominator scalar branches are separate from default cases. Necessary survivors are not kernels.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
