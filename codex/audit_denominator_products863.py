"""Initial-numerator computed denominators with q-plus-product tails.

For P>n the quotient/remainder is0/1/n. Small P values are checked
exactly. At every input with no small match, the last product is forced;
weighted quadratic congruences and partial one-bitop interpolation cover
every one-MAD/bitop prefix, including nonunit multiplicative factors.
"""
from __future__ import annotations

import argparse
from itertools import combinations_with_replacement
from math import gcd
import json
from pathlib import Path

from codex.branch_computed_denominator863 import data, default_q, partial_bit_fits, constrain_default
from codex.screen_xor_mad_polynomial import consistent
from codex.screen_division_xor_affine863 import quotient
from codex.search_early_boundary863 import MASK, NAMES, NODE


def weighted_case(values, targets, numerators, matrix, factor, op):
    forced = []; ambiguous = []; coeffs = []; rhs = []; pp = []; widths = []
    for index, (v, y, n) in enumerate(zip(values, targets, numerators)):
        coefficient = v[factor]
        small = [p for p in range(1, n+1) if (coefficient*p+quotient(n, p, op)) & MASK == y]
        if small:
            ambiguous.append(dict(index=index, possible_small_prefixes=small)); continue
        residual = (y-default_q(n, op)) & MASK
        low = constrain_default(residual, coefficient)
        if low is None:
            return dict(status='excluded', certificate='no_small_match_or_large_prefix_congruence',
                        witness_index=index, numerator=n, coefficient=coefficient, target=y, residual=residual)
        p, width = low
        forced.append(index); coeffs.append(coefficient); rhs.append(residual); pp.append(p); widths.append(width)
    weighted = [[(c*x) & MASK for x in matrix[index]] for index, c in zip(forced, coeffs)]
    polynomial = consistent(weighted, rhs, 32)
    bit = partial_bit_fits([values[i] for i in forced], pp, widths)
    return dict(status='excluded' if not polynomial and not bit else 'open',
                certificate='weighted_polynomial_and_partial_bitop_constraints',
                forced_indices=forced, small_prefix_branches=ambiguous,
                polynomial_consistent=polynomial, bitop_fits=bit,
                mad_excluded=not polynomial, bit_excluded=not bit)


def square_possible(value, width=32):
    value &= (1 << width)-1
    if not value: return True
    valuation = (value & -value).bit_length()-1
    if valuation & 1: return False
    remaining = width-valuation; odd = value >> valuation
    return odd & ((1 << min(3, remaining))-1) == 1


def square_case(targets, numerators, op):
    for index, (y, n) in enumerate(zip(targets, numerators)):
        small = [p for p in range(1, n+1) if (p*p+quotient(n, p, op)) & MASK == y]
        if small: continue
        residual = (y-default_q(n, op)) & MASK
        if not square_possible(residual):
            return dict(status='excluded', witness_index=index, numerator=n, target=y,
                        square_residual=residual, no_small_prefix_match=True)
    return dict(status='open')


def linear_class(coefficient, rhs, width=32):
    coefficient &= (1 << width)-1; rhs &= (1 << width)-1
    common = gcd(coefficient, 1 << width)
    if rhs % common: return None
    modulus = (1 << width)//common
    if modulus == 1: return 0, 1
    base = ((rhs//common)*pow(coefficient//common, -1, modulus)) % modulus
    return base, modulus


def intersect(left, right):
    if left is None or right is None: return None
    a, m = left; b, n = right
    if m < n: a, b, m, n = b, a, n, m
    return (a, m) if a % n == b else None


def normalize(classes):
    result = []
    for a, m in sorted(set(classes), key=lambda x:(x[1], x[0])):
        if not any(a % n == b for b, n in result): result.append((a, m))
    return result


def scalar_initial_case(values, targets, numerators, factor):
    classes = [(0, 1)]; history = []
    for index, (v, y, n) in enumerate(zip(values, targets, numerators)):
        coefficient = v[factor] if factor < 9 else 1
        updated = []
        for q in range(n+1):
            next_class = linear_class(coefficient, y-q)
            for old in classes:
                common = intersect(old, next_class)
                if common is not None: updated.append(common)
        classes = normalize(updated)
        history.append(dict(index=index, residue_classes=len(classes)))
        if not classes: break
    return dict(status='excluded' if not classes else 'open', history=history,
                scalar_classes=[list(c) for c in classes],
                proof='For every valid uint32 DIV/REM/CDIV,0<=q<=n. Thus G*initial=target-q for some integer q in this exact finite range. Congruence classes cover every uint32 G, including even G.')


def fixed_product_case(values, targets, numerators, left, right):
    for index, (v, y, n) in enumerate(zip(values, targets, numerators)):
        q = (y-v[left]*v[right]) & MASK
        if q > n:
            return dict(status='excluded', witness_index=index, required_quotient=q, numerator=n)
    return dict(status='open')


def audit():
    rows = []; datasets = {}
    for source in range(9):
        bs, vv, yy, nn, matrix = data(source)
        datasets[str(source)] = dict(primary_b=bs, numerators=nn, targets=yy)
        weighted = []; squares = []
        for op in ('//', 'cdiv', '%'):
            for factor in range(9):
                weighted.append(dict(op=op, factor=factor, **weighted_case(vv, yy, nn, matrix, factor, op)))
            squares.append(dict(op=op, **square_case(yy, nn, op)))
        scalars = [dict(factor=factor, **scalar_initial_case(vv, yy, nn, factor)) for factor in range(10)]
        fixed = [dict(factors=[i, j], **fixed_product_case(vv, yy, nn, i, j))
                 for i, j in combinations_with_replacement(range(9), 2)]
        rows.append(dict(source=source, source_name=NAMES[source], weighted_prefix_products=weighted,
                         prefix_squares=squares, scalar_initial_products=scalars, fixed_initial_products=fixed))
    counts = {}
    for key in ('weighted_prefix_products', 'prefix_squares', 'scalar_initial_products', 'fixed_initial_products'):
        group = [x for row in rows for x in row[key]]
        counts[key] = dict(cases=len(group), excluded=sum(x['status'] == 'excluded' for x in group))
    return dict(node=NODE, input_names=NAMES, datasets=datasets, results=rows, counts=counts,
                weighted_scope='P one MAD/bitop; initial numerator; q=DIV/REM/CDIV(n,P); final initial*P+q. Nonunit factors use exact low-bit congruences.',
                square_scope='Same initial numerator and arbitrary nonzero denominator P, final P*P+q; a single impossible-input word-square condition suffices when reported excluded.',
                scalar_and_fixed_scope='Prepared scalar times an initial word (or a prepared constant), or a product of two initial words, plus q. The denominator is arbitrary; only0<=q<=initial_numerator is used.',
                best_executable_cycles=864, new_executable=False, no_global_optimality_claim=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['counts'], flush=True)


if __name__ == '__main__':
    main()
