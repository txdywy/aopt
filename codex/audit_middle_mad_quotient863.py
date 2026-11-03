"""Two exact finite certificates for computed-prefix division families.

On the recorded rectangle every initial word has common low16 and zero
mixed difference, so every one-MAD prefix P and polynomial tail R have
zero mixed difference. The target mixed difference is exactly2^31.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from codex.search_early_boundary863 import MASK, NAMES, NODE, early_inputs, wanted, legal_point


HALF_RECTANGLE = [3597240495, 3597174959, 3597371567, 3597306031]


def mixed(v, mask=MASK):
    return (int(v[0])-int(v[1])-int(v[2])+int(v[3])) & mask


def audit():
    vv = [early_inputs(b) for b in HALF_RECTANGLE]; yy = [wanted(v) for v in vv]
    assert all(len({v[j] & 65535 for v in vv}) == 1 and mixed([v[j] for v in vv]) == 0
               for j in range(9))
    assert mixed(yy) == 1 << 31
    # Exact remaining small-divisor bounds after the large-divisor range cut.
    rows = []
    for op, last in (('//', 3), ('cdiv', 4)):
        for d in range(1, last+1):
            if d & 1:
                assert ((d*(1 << 31)) & MASK) == 1 << 31
                assert 2*(d-1) < 1 << 31
                why = 'D*target_mixed=2^31, but the four remainder/ceil-adjustment terms have absolute mixed value<=2*(D-1). An odd tail quotient coefficient preserves2^31.'
            else:
                assert d in (2, 4)
                why = 'For D=2 or4, floor/ceil parity depends only on prefix low3. All prefix low3 and T-only tail low bits ignore b16, whereas target parity flips.'
            rows.append(dict(op=op, divisor=d, status='excluded', proof=why))
    assert 2*(MASK//4) < 1 << 31
    assert 2*((MASK+4)//5) < 1 << 31
    return dict(node=NODE, input_names=NAMES,
                rectangle=dict(primary_b=HALF_RECTANGLE, initial_values=vv, targets=yy,
                               target_mixed=1 << 31, legal_points=[legal_point(b) for b in HALF_RECTANGLE]),
                prefix_lemma='If all word increments are multiples of2^16 and each mixed difference is0 modulo2^32, cross-products of increments vanish. Therefore every integer polynomial in these words, including a first MAD and any polynomial tail R, has zero mixed difference.',
                families=[
                    'P=one arbitrary MAD(initials/K);q=DIV/REM/CDIV(P,D);out=R(initials,P,K)+q, for any integer polynomial R.',
                    'P=one arbitrary MAD(initials/K);q=DIV/REM/CDIV(P,D);out=P+G*q, G an arbitrary root-only constant.',
                ],
                odd_g_necessity='In the second family, target parity flips under b16, while P parity does not. Thus G is odd and G*x=2^31 implies x=2^31 modulo2^32.',
                large_divisor_cuts=dict(floor_D_at_least4='excluded: q<=floor(MASK/D), so |mixed(q)|<2^31',
                    ceil_D_at_least5='excluded: q<=ceil(MASK/D), so |mixed(q)|<2^31'),
                small_divisor_cases=rows,
                rem_proof='An even D preserves prefix parity and cannot transfer b16. For odd D, |mixed(q)|<=2*(D-1) forces D>2^30. Then k=floor(P/D)<=3, and q=P-D*k gives target_mixed=-D*mixed(k), up to the odd constant G in the second family. Odd multiplication preserves the unique element2^31; |mixed(k)|<=6 cannot equal it.',
                remaining_in_stated_families=0, best_executable_cycles=864, new_executable=False,
                scope='One arbitrary MAD prefix, constant nonzero uint32 divisor, and the stated polynomial-plus-quotient / prefix-plus-constant-times-quotient output. This covers every final MAD with q as its addend, and the final P+G*q form. Does not cover a computed denominator or every q-multiplicative tail.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args(); result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Exact2^31 mixed rectangle closes stated computed-numerator MAD tails for all divisors', flush=True)


if __name__ == '__main__':
    main()
