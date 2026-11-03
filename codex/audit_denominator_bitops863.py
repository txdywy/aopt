"""Bitwise tails for an initial numerator and one-T computed denominator.

Small numerator values force the denominator for XOR/floor and XOR/ceil.
All other reported exclusions use a zero, range, or permutation argument.
This is a local grammar certificate, never a full-kernel lower bound.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from codex.branch_computed_denominator863 import data, partial_bit_fits
from codex.screen_xor_mad_polynomial import consistent
from codex.search_early_boundary863 import MASK, NAMES, NODE, legal_point


def forced_xor(targets, numerators, op):
    """Necessary P for Y=P XOR op(n,P), including the n=0 case."""
    result = []
    for index, (y, n) in enumerate(zip(targets, numerators)):
        # With P<=n, both operands of XOR are <=n. Their XOR has at most
        # n.bit_length() bits. n=0 instead always has q=0 for valid P.
        if not n or y >= 1 << n.bit_length():
            p = y ^ int(op == 'cdiv' and n > 0)
            result.append(dict(index=index, numerator=n, target=y, forced_prefix=p,
                               valid_large_prefix=p > n))
    return result


def audit():
    rows = []
    for source, name in enumerate(NAMES):
        bs, vv, yy, nn, matrix = data(source)
        zero = nn.index(0)
        assert yy[zero] and all(x != yy[zero] for x in vv[zero])
        bit_initial = dict(all_ops_excluded=True, witness=legal_point(bs[zero]),
                           reason='At n=0, q=0. XOR/OR with an initial word returns that word; AND returns0. None equals the target.')
        one = nn.index(1) if 1 in nn else zero
        assert yy[one] > nn[one]
        and_prefix = dict(all_ops_excluded=True, witness_index=one,
                          reason='q AND P <= q <= n, but this target exceeds n.')
        xor_prefix = []
        for op in ('//', 'cdiv'):
            forced = forced_xor(yy, nn, op)
            assert forced
            invalid = [r for r in forced if not r['valid_large_prefix']]
            polynomial = consistent([matrix[r['index']] for r in forced],
                                    [r['forced_prefix'] for r in forced], 32)
            bit = partial_bit_fits([vv[r['index']] for r in forced],
                                  [r['forced_prefix'] for r in forced], [32]*len(forced))
            xor_prefix.append(dict(op=op, status='excluded' if invalid or not polynomial and not bit else 'open',
                                   forced=forced, primary_b=[bs[r['index']] for r in forced],
                                   invalid_forced_prefixes=invalid, polynomial_consistent=polynomial,
                                   bitop_fits=bit))
        rows.append(dict(source=source, name=name, bitop_initial=bit_initial,
                         and_prefix=and_prefix, xor_prefix=xor_prefix))
    return dict(node=NODE, input_names=NAMES, results=rows,
                xor_prefix_rem_excluded='For every valid nonzero P, n%P<P, so P XOR(n%P) is never0. The target is a full32 permutation and includes0.',
                or_prefix_all_ops_excluded='P OR q cannot be0 when the divisor P is nonzero.',
                prepared_mask_or_affine_tail_excluded='The target is a full32 permutation. A bitwise prepared-mask or affine prepared-coefficient tail therefore requires q to be a full32 permutation. For a bijective initial numerator n with0<=q<=n, equality of sums forces q=n pointwise. Such a tail cannot change parity on b16. The two nonbijective initial numerators have maximum below MASK and cannot give a permutation q.',
                all_stated_cases_excluded=all(r['status'] == 'excluded' for x in rows for r in x['xor_prefix']),
                best_executable_cycles=864, new_executable=False,
                scope='One MAD/XOR/AND/OR prefix P, DIV/REM/CDIV(initial_numerator,P), final XOR/AND/OR with P, an initial word, or a prepared constant. Includes arbitrary root-only masks. No claim about other source grammars.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('All stated bitwise-tail cases excluded:', result['all_stated_cases_excluded'], flush=True)


if __name__ == '__main__':
    main()
