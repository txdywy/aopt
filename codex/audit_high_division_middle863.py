"""D,T,D: a high final quotient must equal the middle ordinary word.

On each initial LSB class, an ordinary middle expression has a unary
Boolean dependence on the first quotient bit. High targets vary in both
parities, forcing an exact quotient/target parity difference relation.
All nonzero prepared divisors are retained as exact intervals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

from codex.audit_comparison_core863 import fibre
from codex.audit_division_first_t863 import refine, source_inverse
from codex.branch_computed_denominator863 import partial_bit_fits
from codex.screen_xor_mad_polynomial import consistent
from codex.search_early_boundary863 import MASK, NAMES, NODE, early_inputs, wanted, legal_point


def high_points():
    rng = random.Random(8642903)
    bs = [*range(128), *(rng.getrandbits(32) for _ in range(256))]
    bs += [source_inverse(0, (h << 16) | lo) for h in [*range(32), 32768, 65535] for lo in (0, 1, 65535)]
    bs += [source_inverse(src, n) for src in (0, 1, 2, 4, 5, 6, 8) for n in range(32)]
    return list(dict.fromkeys(x for b in bs for x in (b, b ^ 65536)))


def screen(source, ceil=False, points=None, target=wanted):
    points = high_points() if points is None else points
    rows = [(b, early_inputs(b), target(early_inputs(b))) for b in points]
    rows = [r for r in rows if r[2] > 1 << 31]
    anchors = {}; parities = {}
    for b, v, y in rows:
        cls = b & ((1 << 19) | 1)
        anchors.setdefault(cls, (b, v, y)); parities.setdefault(cls, set()).add(y & 1)
    informative = len(parities) == 4 and all(len(p) == 2 for p in parities.values())
    if not informative:
        return dict(source=source, op='cdiv' if ceil else '//', status='open',
                    survivor_intervals=[[1, MASK]], high_class_variation=False)
    intervals = [[1, MASK]]; history = []; used = []
    for b, v, y in rows:
        old = intervals
        _, base, base_y = anchors[b & ((1 << 19) | 1)]
        n, a = v[source], base[source]
        desired = (y ^ base_y ^ int(ceil and n > 0) ^ int(ceil and a > 0)) & 1
        nf, af = (max(n-1, 0), max(a-1, 0)) if ceil else (n, a)
        if nf == af:
            intervals = [] if desired else intervals
        else:
            intervals = refine(intervals, nf, af, desired)
        used.append(b)
        if old != intervals:
            history.append(dict(primary_b=b, remaining_intervals=len(intervals),
                                remaining_divisors=sum(hi-lo+1 for lo, hi in intervals)))
        if not intervals: break
    return dict(source=source, source_name=NAMES[source], op='cdiv' if ceil else '//',
                status='excluded' if not intervals else 'necessary_survivors', high_class_variation=True,
                anchors={str(k):v[0] for k, v in anchors.items()},
                tested_high_primary=used, refinement_history=history,
                survivor_intervals=intervals, exact_nonzero_divisor_domain=True,
                surviving_odd_divisors=sum((hi+1)//2-lo//2 for lo, hi in intervals))


def suffix_checks(quotients, targets, width=16):
    matrix = [[1, q, q*q] for q in quotients]
    polynomial = consistent(matrix, targets, width)
    bit = partial_bit_fits([[q]*9 for q in quotients], targets, [width]*len(targets))
    modulus = 1 << width; null = None
    if len(quotients) == 4:
        weights = []
        for i in range(4):
            x = [q for j, q in enumerate(quotients) if j != i]
            determinant = (x[1]-x[0])*(x[2]-x[0])*(x[2]-x[1])
            weights.append(((-1)**i*determinant) % modulus)
        lhs = [sum(w*q**power for w, q in zip(weights, quotients)) % modulus for power in range(3)]
        assert lhs == [0, 0, 0]
        null = dict(weights=weights, polynomial_column_residues=lhs,
                    target_residue=sum(w*y for w, y in zip(weights, targets)) % modulus)
    collision = next((dict(bit=k, pair=[i, j], quotient_bits=[(quotients[t] >> k) & 1 for t in (i, j)],
                           target_bits=[(targets[t] >> k) & 1 for t in (i, j)])
                      for k in range(width) for i in range(len(quotients)) for j in range(i)
                      if ((quotients[i] ^ quotients[j]) >> k) & 1 == 0 and
                         ((targets[i] ^ targets[j]) >> k) & 1), None)
    return dict(quadratic_consistent=polynomial, bitop_fits=bit,
                quadratic_null_certificate=null, unary_bit_collision=collision,
                status='excluded' if not polynomial and not bit else 'open')


def audit():
    points = high_points(); rows = []
    for source in range(9):
        for ceil in (False, True):
            row = screen(source, ceil, points); rows.append(row)
            print(NAMES[source], row['op'], row['survivor_intervals'][:6], flush=True)
    survivors = [(r['source'], r['op'], r['survivor_intervals']) for r in rows if r['survivor_intervals']]
    # Preserve necessary survivors explicitly; never relabel an unfinished
    # interval domain as excluded. This target has exactly one reduction.
    expected = [(0, '//', [[65536, 65536]])]
    reduced = survivors == expected
    ff = fibre(); indices = [i for i, y in enumerate(ff['target']) if y > 1 << 31]
    vv = [ff['initial_values'][i] for i in indices]; yy = [ff['target'][i] & 65535 for i in indices]
    assert len(indices) == 4 and all(len({v[j] & 65535 for v in vv}) == 1 for j in range(9))
    q = [v[0] >> 16 for v in vv]; suffix = suffix_checks(q, yy)
    assert suffix['quadratic_null_certificate']['target_residue'] != 0
    assert suffix['unary_bit_collision'] is not None
    facts_path = Path('codex/division_first_t863_audit.json'); raw = facts_path.read_bytes(); facts = json.loads(raw)
    assert facts['node'] == NODE and facts['full32_class_facts']['target_b16_flip_full32'] == 'unsat'
    assert all(r['status'] == 'unsat' for r in facts['full32_class_facts']['initial_lsb_depends_only_on_b0_b19'])
    legal = sorted({b for r in rows for b in r.get('tested_high_primary', [])} |
                   {ff['primary_b'][i] for i in indices})
    return dict(node=NODE, input_names=NAMES, results=rows, remaining_first_quotient_families=survivors,
                parity_evidence=dict(path=str(facts_path), sha256=hashlib.sha256(raw).hexdigest()),
                high_primary_pool=points,
                quotient_class_proof='All initial LSBs depend only on b0,b19. For each such class, target values>2^31 include both parities. Any ordinary middle T bit0 is a Boolean function only of first-q bit0 and this fixed class, so it must be q_bit XOR a class constant. Pairwise high-target and quotient parity differences must match.',
                rem_reduction='Even D preserves numerator parity and fails high-class target variation. For odd D, remainder parity is numerator parity XOR floor-quotient parity; numerator parity is fixed inside a class. The exact floor intervals have no odd survivor.',
                remaining_floor_suffix=dict(high_fibre_indices=indices, primary_b=[ff['primary_b'][i] for i in indices],
                    quotients=q, targets_low16=yy, **suffix,
                    proof='The only surviving first quotient is e//65536. On the four high-target fibre points, every initial low16 is constant. Any single middle MAD is therefore contained in the3-coefficient polynomial span[1,q,q^2] modulo2^16; any single bitop is a constant or q OP a prepared mask. Both necessary systems fail.'),
                last_division_partition='For final DIV/CDIV output>2^31 the denominator is1 and the numerator equals the output. Final SHR output>2^31 likewise requires count0 and equals its value operand, including data-dependent counts. A middle-T numerator/value is covered by this high-prefix certificate. An initial/first-q/constant value follows the prior conservative ancestor certificate without using the mixing T. Final REM can never attain MASK, regardless of its prefix.',
                legal_points=[legal_point(b) for b in legal],
                complete_in_stated_scope=reduced and suffix['status'] == 'excluded' and
                    all(not r.get('surviving_odd_divisors', 1) for r in rows if r['op'] == '//'),
                best_executable_cycles=864, new_executable=False,
                scope='DIV/REM/CDIV, one MAD/XOR/AND/OR, final DIV/REM/CDIV or SHR on the nine initial words and root-only prepared constants. All valid first divisors are prepared; later operands/denominators/shift counts may be computed. No larger window or additional free words.',
                no_global_optimality_claim=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('D,T,D closed:', result['complete_in_stated_scope'], flush=True)


if __name__ == '__main__':
    main()
