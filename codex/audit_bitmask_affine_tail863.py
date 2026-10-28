"""Exclude AND/OR with a live word before an affine boundary output.

The other operand q is completely unconstrained. This covers the AND/OR
versions of the new division-first family and, without further solving,
two extensions of the previous joint MAD/SHR16/XOR/MAD subfamily.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import z3

from codex.search_early_boundary863 import NAMES, NODE, early_inputs, wanted, legal_point
from codex.search_joint_live_boundary863 import affine_tail_manifest


def audit():
    b = z3.BitVec('bitmask_tail_b', 32)
    vv, ww = early_inputs(b), early_inputs(b ^ (1 << 31))
    s = z3.SolverFor('QF_BV'); s.set(timeout=30000)
    s.add(z3.Extract(15, 0, wanted(vv) ^ wanted(ww)) != 0x8000)
    assert s.check() == z3.unsat
    a, c, t = z3.BitVecs('bitmask_affine_a bitmask_affine_c bitmask_affine_t', 16)
    s = z3.SolverFor('QF_BV'); s.set(timeout=30000)
    s.add(a & 1 == 1, ((a*t+c) ^ (a*(t ^ 0x8000)+c)) != 0x8000)
    assert s.check() == z3.unsat
    rng = random.Random(8642851)
    bs = [*range(256), *(rng.getrandbits(32) for _ in range(2048))]
    rows = []
    for j, name in enumerate(NAMES):
        bit = 15 if j < 5 else 12
        expected = 0 if j < 5 else 1
        s = z3.SolverFor('QF_BV'); s.set(timeout=30000)
        s.add(z3.Extract(bit, bit, vv[j] ^ ww[j]) != expected)
        assert s.check() == z3.unsat, name
        for op in ('&', '|'):
            if j < 5:
                state = int(op == '|')
                point = next(x for x in bs if ((early_inputs(x)[j] >> 15) & 1) == state)
                pair = [point, point ^ (1 << 31)]
                assert all(((early_inputs(x)[j] >> 15) & 1) == state for x in pair)
                proof = ('The live mask bit15 is0 on both pair members, fixing AND bit15 to0; '
                         'the inverse affine target must flip bit15.' if op == '&' else
                         'The live mask bit15 is1 on both pair members, fixing OR bit15 to1; '
                         'the inverse affine target must flip bit15.')
                evidence = dict(primary_pair=pair, legal_points=[legal_point(x) for x in pair])
            else:
                # Every b belongs to a pair where W12 differs. F12 must be
                # equal within the pair, so AND forces it0 globally / OR1.
                proof = ('The live mask bit12 flips on b31 pairs but the inverse affine target bit12 does not. '
                         f'Bit12 of the entire output is therefore fixed to{int(op == "|")}, '
                         'contradicting target permutation and odd affine bijectivity.')
                evidence = dict(bit12_flip_full32='unsat', fixed_output_bit12=int(op == '|'))
            rows.append(dict(source=j, source_name=name, op=op, status='excluded', proof=proof, **evidence))
    t = z3.BitVec('bitmask_self_fold_t', 32); q = z3.LShR(t, 16)
    and_solver = z3.SolverFor('QF_BV'); and_solver.add(z3.UGE(t & q, 1 << 16))
    or_solver = z3.SolverFor('QF_BV'); or_solver.add((t | q) == 1 << 31)
    assert and_solver.check() == z3.unsat and or_solver.check() == z3.unsat
    manifest, _ = affine_tail_manifest(); assert len(manifest) == 4200
    prior_path = Path('codex/joint_live_boundary863_closed_audit.json')
    prior_bytes = prior_path.read_bytes(); prior = json.loads(prior_bytes)
    assert prior['node'] == NODE and prior['complete_affine_tail']['remaining'] == 0
    return dict(node=NODE, input_names=NAMES, live_operand_masks=rows,
                target_permutation_proof='u=33*b+K1 has odd multiplier; v has factor512 so each z_j=u_j XOR v_j depends on b_j with coefficient1 and only lower b bits otherwise. Thus z, e=9*z+C4, e XOR(e>>16) XOR N, and the final odd affine are all32-bit permutations.',
                inverse_affine_b31_xor16='unsat',
                prepared_constant_masks='AND requires constant MASK and OR requires0 for bijectivity; then q alone remains. q=t>>16 has at most65536 values.',
                self_prefix_masks=dict(and_image_below_65536_full32='unsat',
                                       or_missing_output_0x80000000_full32='unsat'),
                joint_xor_evidence=dict(path=str(prior_path), sha256=hashlib.sha256(prior_bytes).hexdigest()),
                joint_affine_tail_cases=dict(xor=4200, and_=4200, or_=4200, total=12600,
                                             all_status='excluded', free_e=False),
                best_executable_cycles=864, new_executable=False,
                scope='For the fixed legal root, out=G*(q AND/OR h)+H with G/H arbitrary root-only constants, q arbitrary and h any nine live words. Also complete joint t=MAD(eight inputs/K);q=t>>16;f=q BIT h;out=G*f+H, BIT in XOR/AND/OR, h initial/t/K. Does not cover arbitrary four-op graphs or all division topologies.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('18 arbitrary-prefix live-mask exclusions; joint affine-tail XOR/AND/OR family12600 excluded', flush=True)


if __name__ == '__main__':
    main()
