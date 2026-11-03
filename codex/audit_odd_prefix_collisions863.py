"""Find parameter-universal collisions of every always-odd one-op prefix.

For out=P*q+W, a unique downward-transfer q forces P odd everywhere.
If q depends only on P/root constants, target-W must be constant on P
collisions. Prefix-parameter parity is enumerated exactly. MAD equality
on six Newton-grid evaluations proves equality for all same-parity keys;
bitop equality on all bitwise key corners is exact too.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import time

import numpy as np

from codex.audit_division_first_t863 import CLASSES, source_inverse
from codex.cegis_c1_absorb863 import _instr_options
from codex.search_early_boundary863 import MASK, NAMES, NODE, early_inputs, wanted, legal_point


def evaluate(op, wiring, values, keys):
    pp = list(values)+list(keys); a, b = pp[wiring[0]], pp[wiring[1]]
    return ((a*b+pp[wiring[2]]) & MASK) if op == 'mad' else a^b if op == '^' else a&b if op == '&' else a|b


def key_grid(op, parity):
    p, q = parity
    if op == 'mad':
        return [(p, q), (p+2, q), (p, q+2), (p+4, q), (p+2, q+2), (p, q+4)]
    return list(product((p, (MASK & ~1) | p), (q, (MASK & ~1) | q)))


def manifest():
    rows = []
    initial = [early_inputs(b) for b in CLASSES]
    for op in ('mad', '^', '&', '|'):
        for wiring in _instr_options(8, op):
            used = sorted(set(wiring) & {9, 10}); admissible = []
            for bits in product((0, 1), repeat=len(used)):
                parity = [0, 0]
                for key, bit in zip(used, bits):
                    parity[key-9] = bit
                if all(evaluate(op, wiring, v, parity) & 1 for v in initial):
                    admissible.append(parity)
            rows.append(dict(op=op, wiring=list(wiring), odd_key_parities=admissible))
    return rows


def prove_case(row, parity, bs, values, targets, preferred):
    grid = key_grid(row['op'], parity)
    with np.errstate(over='ignore'):
        cols = [np.broadcast_to(np.asarray(evaluate(row['op'], row['wiring'], values,
                                                   [np.uint64(k) for k in keys]), dtype=np.uint64), bs.shape)
                for keys in grid]
    # Preferred deterministic pairs often close whole parameter families.
    pairs = list(preferred)
    varying = [col for col in cols if np.any(col != col[0])]
    if len(varying) >= 2:
        # First-MAD grid values are affine combinations of at most two
        # input-dependent coefficients. Use exact structured equality;
        # never hash away extra columns or infer equality from a digest.
        records = np.rec.fromarrays([col.astype(np.uint32) for col in varying])
        order = np.argsort(records, kind='stable')
        same = np.ones(len(bs)-1, dtype=bool)
        for col in cols:
            same &= col[order[:-1]] == col[order[1:]]
        adjacent = np.flatnonzero(same)
    elif varying:
        order = np.argsort(varying[0], kind='stable')
        adjacent = np.flatnonzero(varying[0][order[:-1]] == varying[0][order[1:]])
    else:
        order = np.arange(len(bs)); adjacent = np.arange(min(64, len(bs)-1))
    pairs.extend((int(order[i]), int(order[i+1])) for i in adjacent)
    remaining = set(range(9)); witnesses = []
    for i, j in pairs:
        if not all(col[i] == col[j] for col in cols):
            continue
        mismatch = [w for w in sorted(remaining)
                    if ((int(targets[i])-int(values[w][i])) & MASK) !=
                       ((int(targets[j])-int(values[w][j])) & MASK)]
        if mismatch:
            witnesses.append(dict(primary_pair=[int(bs[i]), int(bs[j])], addends=mismatch,
                                  common_grid_values=[int(col[i]) for col in cols]))
            remaining.difference_update(mismatch)
            if not remaining:
                break
    return dict(op=row['op'], wiring=row['wiring'], key_parity=parity,
                parameter_grid=grid, witnesses=witnesses, remaining_addends=sorted(remaining),
                status='excluded' if not remaining else 'open')


def audit(samples=131072):
    started = time.monotonic(); preferred_bs = []
    for source in (0, 1, 2, 4, 5, 6, 8):
        for end in (1, 1 << 31):
            preferred_bs.append([source_inverse(source, 0), source_inverse(source, end)])
    preferred_bs += [[b, b ^ shift] for b in CLASSES for shift in (65536, 1 << 31)]
    # P=e^2+z=81*z^2+(18*C4+1)*z+C4^2 has a key-independent
    # quadratic reflection z'=-z-(18*C4+1)/81. Birthday discovery may
    # miss this two-to-one prefix; the algebraic pair is deterministic.
    reflected_z = (-(18*0xFD7046C5+1)*pow(81, -1, 1 << 32)) & MASK
    preferred_bs.append([source_inverse(1, 0), source_inverse(1, reflected_z)])
    flat = [b for pair in preferred_bs for b in pair]
    rng = np.random.default_rng(8642866)
    bs = np.unique(np.concatenate([np.array(flat, dtype=np.uint64), rng.integers(0, 1 << 32, samples, dtype=np.uint64)]))
    indices = {int(b):i for i, b in enumerate(bs)}
    preferred = [(indices[a], indices[b]) for a, b in preferred_bs]
    with np.errstate(over='ignore'):
        values = early_inputs(bs); targets = wanted(values)
    rows = manifest(); cases = []
    for index, row in enumerate(rows):
        for parity in row['odd_key_parities']:
            case = prove_case(row, parity, bs, values, targets, preferred); case['prefix_index'] = index
            cases.append(case)
    legal = sorted({b for case in cases for witness in case['witnesses'] for b in witness['primary_pair']})
    facts_path = Path('codex/division_first_t863_audit.json')
    facts_raw = facts_path.read_bytes(); facts = json.loads(facts_raw)
    assert facts['node'] == NODE and facts['full32_class_facts']['target_b16_flip_full32'] == 'unsat'
    return dict(node=NODE, input_names=NAMES, canonical_prefixes=len(rows), manifest=rows,
                parity_cases=len(cases), excluded=sum(r['status'] == 'excluded' for r in cases), cases=cases,
                samples=len(bs), pool_sha256=hashlib.sha256(bs.astype('<u4').tobytes()).hexdigest(),
                legal_points=[legal_point(b) for b in legal], seconds=round(time.monotonic()-started, 6),
                initial_parity_evidence=dict(path=str(facts_path), sha256=hashlib.sha256(facts_raw).hexdigest()),
                necessary_odd_proof='P and every initial addend have equal LSB at b and b^65536. Target LSB flips. If P were even on either parity class, P*q+W would not flip. Initial LSBs depend only on b0,b19, so the four-class key-parity screen is exact.',
                mad_grid_proof='Substitute K_i=parity_i+2*t_i. A first MAD is an integer polynomial of degree<=2 in t0,t1. Its six Newton forward-difference coefficients are determined by (0,0),(1,0),(0,1),(2,0),(1,1),(0,2). Equal values there imply equal polynomials as functions modulo2^32 for every integer t, without dividing by a nonunit.',
                bit_grid_proof='Each output bit reads only the corresponding key bits. The fixed parity bits and all four high-bit key corners exhaust every bitwise truth assignment.',
                constant_prefix='A root-only constant P and root-only quotient parameters produce a constant residual target-W, contradicted by the b16 parity pair.',
                best_executable_cycles=864, new_executable=False,
                scope='out=P*q+W, W any nine initial words; P one MAD/XOR/AND/OR and q any function only of P/root constants. This includes DIV/REM/CDIV(P,D) and DIV/REM/CDIV(C,P), whenever defined. Does not include a dividend from another initial word and a computed denominator P, or a different final tail.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--samples', type=int, default=131072)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--refine-from', type=Path)
    args = ap.parse_args(); result = audit(args.samples)
    if args.refine_from:
        raw = args.refine_from.read_bytes(); previous = json.loads(raw)
        assert previous['canonical_prefixes'] == result['canonical_prefixes']
        result['refined_from'] = dict(path=str(args.refine_from), sha256=hashlib.sha256(raw).hexdigest(),
                                      excluded=previous['excluded'], parity_cases=previous['parity_cases'])
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('prefixes', result['canonical_prefixes'], 'odd parity cases', result['parity_cases'],
          'excluded', result['excluded'], 'seconds', result['seconds'], flush=True)
    for row in result['cases']:
        if row['remaining_addends']:
            print('OPEN', row['op'], row['wiring'], row['key_parity'], row['remaining_addends'], flush=True)


if __name__ == '__main__':
    main()
