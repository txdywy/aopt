"""Complete floor-XOR-prefix profiles and conservative wordcut tails.

For Y=P XOR floor(P/D), permutation Y forces permutation P. Odd D has
the collision P=D-1,D. For D>=2, output and P share their leading bit.
Thus Y<128 forces P<128: all even D<128 and one D>=128 profile cover
every uint32 divisor. Every profile is checked against all one-T P.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from codex.audit_division_first_t863 import source_inverse
from codex.audit_runtime_shift_core863 import base_fields, monomials
from codex.audit_small_numerator863 import bit_fits
from codex.close_computed_denominator863 import edge_target
from codex.screen_xor_mad_polynomial import consistent
from codex.search_early_boundary863 import MASK, NAMES, NODE, early_inputs, wanted, legal_point


def inverse_profile(divisor, bits=7):
    size = 1 << bits
    inverse = {p ^ (p//divisor):p for p in range(size)}
    return [inverse[y] for y in range(size)] if len(inverse) == size else None


def audit():
    bs = [edge_target(y) for y in range(128)]; vv = [early_inputs(b) for b in bs]
    matrix = [monomials(base_fields(b, v)) for b, v in zip(bs, vv)]
    profiles = []
    for d in [*range(2, 128, 2), 128]:
        pp = inverse_profile(d); assert pp is not None
        polynomial = consistent(matrix, pp, 32); bit = bit_fits(vv, pp)
        assert not polynomial and not bit
        profiles.append(dict(divisor=d, represents_all_D_at_least128=d == 128,
                             required_prefix_values=pp, quadratic_consistent=polynomial,
                             bitop_fits=bit, status='excluded'))
    zero_b = bs[0]; zero_values = vv[0]
    assert all(zero_values) and len(set(zero_values)) == 9
    only_p = []
    # When the final output is only a function f(P), permutation Y forces
    # P to be a permutation. For the listed f, f(0)=0. A first bitop then
    # reduces to I XOR K with a permutation initial I and K=I(Y=0).
    for source in (0, 1, 2, 4, 5, 6, 8):
        key = zero_values[source]
        primary = [source_inverse(source, key ^ p) for p in (1, 2, 3)]
        values = [early_inputs(b) for b in primary]; yy = [wanted(v) for v in values]
        assert all(y > 2 for y in yy) and yy[0] & 1
        scalar_cases = []
        for q_at_one in (0, 1):
            g = (yy[0]-q_at_one) & MASK
            bad = next((dict(prefix=p, required_quotient=(y-g*p) & MASK,
                             target=y, primary_b=b)
                        for p, y, b in zip((1, 2, 3), yy, primary) if (y-g*p) & MASK > p), None)
            assert bad is not None
            scalar_cases.append(dict(quotient_at_prefix_one=q_at_one, forced_scalar=g,
                                     impossible_conservative_quotient=bad))
        only_p.append(dict(source=source, source_name=NAMES[source], pinned_xor_key=key,
                           primary_b=primary, targets=yy,
                           scalar_times_P_plus_q=dict(status='excluded', scalar_cases=scalar_cases),
                           P_plus_scalar_times_q=dict(status='excluded',
                               proof='G must be odd by b16 parity. At P=1, q is0 or1. q=0 would give Y=1; q=1 pins G=Y-1, which is even.'),
                           P_square_plus_q=dict(status='excluded',
                               proof='At P=1, q<=1, so P^2+q is1 or2, below the target.'),
                           bitop_P_q=dict(status='excluded',
                               proof='At P=1, q<=1, so any binary bitop of P/q is at most1.')))
    return dict(node=NODE, input_names=NAMES, small_target_values=list(range(128)), primary_b=bs,
                floor_xor_profiles=profiles, profile_count=len(profiles),
                odd_divisor_proof='For odd D<=MASK, g(D-1)=D-1 and g(D)=D XOR1=D-1. D=1 is also noninjective. Permutation P must visit both values, contradicting permutation target.',
                leading_bit_proof='For D>=2 and2^j<=P<2^(j+1), floor(P/D)<2^j, so Y=P XOR floor(P/D) has the same leading1. Y<128 therefore implies P<128. D>=128 gives q=0 on every such P and has a single shared profile.',
                polynomial_scope='Every one-MAD prefix is contained in the21-coefficient quadratic span in the five exact base fields. A failed congruence system excludes all root-only MAD constants. bit_fits separately covers every first XOR/AND/OR and arbitrary prepared masks.',
                ceil_rem_xor_prefix='For D>=2, P=0 and1 both give P XOR ceil(P/D)=0 and P XOR(P%D)=0; permutation P makes the target noninjective. D=1 floor/ceil gives0 and REM gives the T-only prefix.',
                bitop_only_prefix_reduction=dict(zero_primary_b=zero_b, zero_initial_values=zero_values,
                    proof='Every listed f(P,q) has f(0,0)=0 and q is a conservative function of P. Runtime XOR prefixes are nonzero at the target-zero witness because its nine initial values differ. Runtime OR prefixes are nonzero there. An AND prefix containing a permutation initial I is<=I; if both P and I are permutations, sums force P=I, which is nonzero there. The AND of the only two nonpermutation initials cannot reach MASK. Prepared AND/OR masks require MASK/0 for a permutation prefix and again give an initial word. Prepared XOR retains only the seven permutation initials, with its key pinned at target0.'),
                bitop_only_prefix_tails=only_p,
                initial_bitop_prefix_sole_output_families_closed=True,
                legal_points=[legal_point(b) for b in sorted(set(bs + [b for r in only_p for b in r['primary_b']]))],
                best_executable_cycles=864, new_executable=False, no_global_optimality_claim=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Floor-XOR profiles:', result['profile_count'], 'all excluded', flush=True)
    print('First-bitop sole-prefix tail sources:', len(result['bitop_only_prefix_tails']), 'all excluded', flush=True)


if __name__ == '__main__':
    main()
