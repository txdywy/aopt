"""Initial floor/ceil quotient plus a comparison and a final MAD/bitop.

The comparison is relaxed to an arbitrary Boolean at each input. Exact
quotient-constant divisor intervals first use a vectorized three-bit
screen, then the six full16 Boolean/MAD forms. REM is not covered here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from codex.audit_comparison_core863 import fibre, fitting_forms
from codex.screen_live_division_tail863 import divisor_boundaries
from codex.search_early_boundary863 import NAMES, NODE, MASK


def small_mask(q, y, width=3):
    """Keep the union of all six MAD forms and XOR/OR with a Boolean."""
    modulus = 1 << width; mask = modulus-1
    q = np.asarray(q, dtype=np.uint64) & mask
    y = (np.asarray(y, dtype=np.uint64) & mask)[:, None]
    assert q.ndim == 2 and q.shape[0] == len(y)
    keep = np.zeros(q.shape[1], dtype=bool)
    with np.errstate(over='ignore'):
        residual = (y-q) & mask
        for a in range(modulus):
            keep |= np.all(((y-a*q) & mask) <= 1, axis=0)
            keep |= np.all((residual == 0) | (residual == a), axis=0)
            diff = (y-a) & mask
            keep |= np.all((diff == 0) | (diff == q), axis=0)
        keep |= np.all((y == q) | (y == ((2*q) & mask)), axis=0)
        keep |= np.all((y == 0) | (y == ((q+1) & mask)), axis=0)
        keep |= np.all(((y-q*q) & mask) <= 1, axis=0)
    keep |= np.all((y >> 1) == (q >> 1), axis=0)
    return keep


def screen(source, ceil=False):
    start = time.monotonic(); ff = fibre()
    nums = np.array([v[source] for v in ff['initial_values']], dtype=np.int64)
    assert np.all(nums > 0)
    nn = nums-int(ceil); edges = divisor_boundaries(nn, MASK)
    starts, ends = edges[:-1], edges[1:]-1
    for n in nn:
        assert np.array_equal(n//starts, n//ends)
    full_intervals = len(starts)
    qq = (nn[:, None]//starts[None, :]+int(ceil)).astype(np.uint64)
    keep = small_mask(qq, ff['target']); starts, ends, qq = starts[keep], ends[keep], qq[:, keep]
    remaining = []; yy = [y & 65535 for y in ff['target']]
    for index, (lo, hi) in enumerate(zip(starts, ends)):
        q = [int(v) & 65535 for v in qq[:, index]]
        forms = fitting_forms(q, yy)
        bit = all((a >> 1) == (b >> 1) for a, b in zip(q, yy))
        if forms or bit:
            remaining.append(dict(divisor_interval=[int(lo), int(hi)], mad_forms=forms, bitwise=bit))
    return dict(source=source, source_name=NAMES[source], op='cdiv' if ceil else '//',
                whole_divisor_domain=[1, MASK], quotient_intervals=full_intervals,
                boundary_sha256=hashlib.sha256(edges.astype('<i8').tobytes()).hexdigest(),
                endpoint_constancy_checked=True, low3_intervals_retained=len(starts),
                full16_survivor_intervals=remaining, status='excluded' if not remaining else 'open',
                seconds=round(time.monotonic()-start, 6))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); rows = []
    for source in range(9):
        for ceil in (False, True):
            row = screen(source, ceil); rows.append(row)
            print(row['source_name'], row['op'], row['status'], row['low3_intervals_retained'], flush=True)
    result = dict(node=NODE, input_names=NAMES, eight_state_fibre=fibre(), results=rows,
                  excluded=all(r['status'] == 'excluded' for r in rows),
                  all_divisor_intervals=sum(r['quotient_intervals'] for r in rows),
                  bool_relaxed_independently_per_input=True,
                  dead_operations='If the final op does not read both q and the Boolean, either its image has at most two low16 states, or the comparison is dead and the prior division-first two-T closure applies.',
                  comparison_before_division='The same relaxed Boolean covers C,D,T when the quotient uses an initial numerator and prepared divisor. A Boolean numerator has at most two outputs; a Boolean denominator must be1 everywhere for valid execution, reducing to an initial word or0.',
                  best_executable_cycles=864, new_executable=False,
                  scope='A first initial-word floor/ceil quotient by an arbitrary nonzero prepared divisor, one comparison, one MAD/XOR/AND/OR, in either C,D,T or D,C,T order. Comparison is arbitrary Boolean per input. No REM, computed numerator prefix or second shift/division.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
