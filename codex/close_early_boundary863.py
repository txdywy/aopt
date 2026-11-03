"""Audit the three-op core boundary grammar with nine real live words.

Allowed: MAD/XOR/AND/OR and literal right shifts, arbitrary prepared
constants. ADD/SUB/MUL/fixed SHL normalize to MAD. This is a local macro
exclusion, not a lower bound on all kernels. Data-dependent shifts,
division/comparison, FLOW and larger windows are outside the grammar.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import gzip
import hashlib
from itertools import combinations_with_replacement, product
import json
from pathlib import Path
import random

import numpy as np
import z3

from codex.cegis_c1_absorb863 import _instr_options
from codex.search_early_boundary863 import (
    NAMES, NODE, MASK, early_inputs, wanted, legal_point, make_manifest,
)
from codex.screen_xor_mad_polynomial import consistent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row):
    return tuple(row['order']), tuple(tuple(w) for w in row['wiring'])


@lru_cache(maxsize=1)
def initial_low_tables():
    points = [early_inputs(p) for p in (0, 1, 1 << 19, (1 << 19) | 1)]
    return [sum((v[j] & 1) << (2*i+q) for i, v in enumerate(points) for q in (0, 1))
            for j in range(len(NAMES))]


def low_models(row):
    """Exact 8-bit truth-table screen; each bit represents (b0,b19,q0).

    Constants have independent parity, the unique shift bit is abstract.
    A necessary condition is output dependence on q0 in all four initial
    parity classes. No T-function assumption about the early words.
    """
    n = len(NAMES)
    initial = initial_low_tables()
    used = [(i, j-n-i) for i, w in enumerate(row['wiring']) for j in set(w) if j >= n+i]
    found = []
    for bits in product((0, 1), repeat=len(used)):
        kk = [[0, 0] for _ in row['order']]
        for (i, j), bit in zip(used, bits):
            kk[i][j] = 255*bit
        vals = list(initial)
        for op, w, cc in zip(row['order'], row['wiring'], kk):
            pool = vals+cc
            if op == 'shr16':
                value = 0xAA
            else:
                a, b = pool[w[0]], pool[w[1]]
                value = ((a & b) ^ pool[w[2]] if op == 'mad' else
                         a ^ b if op == '^' else a & b if op == '&' else a | b)
            vals.append(value)
        if (vals[-1] ^ (vals[-1] >> 1)) & 0x55 == 0x55:
            found.append((kk, vals[n]))
    return found


def literal_shift_witnesses():
    rng = random.Random(8642826)
    bs = [*range(128), *(rng.getrandbits(32) for _ in range(4096))]
    vv = [early_inputs(b) for b in bs]
    initial, bitwise_prefix = [], []
    legal = set()
    for source, name in enumerate(NAMES):
        for shift in range(33):
            seen = {}; witness = None
            for b, v in zip(bs, vv):
                signature = tuple([x & 1 for x in v]+[(v[source] >> shift) & 1])
                output = wanted(v) & 1
                if signature in seen and seen[signature][1] != output:
                    witness = [seen[signature][0], b]; break
                seen[signature] = (b, output)
            if witness is None:
                assert (source, shift) == (0, 16)
            else:
                initial.append(dict(source=source, source_name=name, shift=shift,
                                    primary_pair=witness, signature=list(signature)))
                legal.update(witness)
    # Before its unique shift, a bitwise-only prefix reads only the same
    # bit position of the original words, regardless of its constants.
    for shift in range(33):
        seen = {}; witness = None
        for b, v in zip(bs, vv):
            signature = tuple([x & 1 for x in v]+[(x >> shift) & 1 for x in v])
            output = wanted(v) & 1
            if signature in seen and seen[signature][1] != output:
                witness = [seen[signature][0], b]; break
            seen[signature] = (b, output)
        if witness is None:
            assert shift == 16
        else:
            bitwise_prefix.append(dict(shift=shift, primary_pair=witness, signature=list(signature)))
            legal.update(witness)
    assert len(initial) == 296 and len(bitwise_prefix) == 32
    return dict(initial_source_cases=initial, bitwise_prefix_cases=bitwise_prefix,
                surviving_initial_source=[0, 16], surviving_bitwise_prefix_shift=16,
                shift32_represents_all_larger_counts=True,
                legal_points=[legal_point(b) for b in sorted(legal)])


def small_shift_facts():
    b = z3.BitVec('small_shift_dependency_b', 32)
    vv, ww = early_inputs(b), early_inputs(b ^ (1 << 16))
    facts = []
    for i, name in enumerate(NAMES):
        s = z3.SolverFor('QF_BV'); s.set(timeout=10000)
        s.add(z3.Extract(15, 0, vv[i]) != z3.Extract(15, 0, ww[i]))
        assert s.check() == z3.unsat, name
        facts.append(dict(name=name+'_low16_ignores_b16', status='unsat'))
    return dict(full32_facts=facts, excluded_unique_literal_shifts=list(range(16)),
                proof='T operations before the only right shift preserve low16 indistinguishability. A shift<=15 cannot bring b16 into bit0; the target bit0 flips on b16.')


def runtime_amount_probe():
    bs = [i << 16 for i in range(8)]
    vv = [early_inputs(b) for b in bs]
    low = [v & 65535 for v in vv[0]]
    assert all([v & 65535 for v in values] == low for values in vv)
    ys = [wanted(v) for v in vv]
    assert len({y & 7 for y in ys}) == 8
    counts = [sum(v[j] >= 32 for v in vv) for j in range(9)]
    assert min(counts) >= 2
    return dict(primary_b=bs, initial_values=vv, common_low16=low, target=ys,
                large_amount_counts=dict(zip(NAMES, counts)), allowed_large_amounts_at_most=1,
                legal_points=[legal_point(b) for b in bs],
                proof='All other operations are T functions of the initial words and shift result. If two amounts>=32, both shift results are zero; the outputs have equal low16, contradicting eight distinct target low3 values.',
                scope='Exactly one data-dependent SHR, its amount reads an initial word, and every other operation is MAD/arithmetic/bitwise/literal-SHL. A new first-operation amount and multiple downward shifts remain open.')


def rectangle():
    bs = [3752425706, 3752360170, 3752556778, 3752491242]
    vv = [early_inputs(b) for b in bs]
    coords = [v+[v[3] >> 9] for v in vv]
    mixed = lambda a: (a[3]-a[1]-a[2]+a[0]) & MASK
    assert all(mixed([v[j] for v in coords]) == 0 for j in range(10))
    assert all((v[j]-coords[0][j]) % 65536 == 0 for v in coords for j in range(10))
    ys = [wanted(v) for v in vv]; delta = mixed(ys)
    distance = min(delta, (-delta) & MASK)
    assert distance > 2*65535
    return dict(primary_b=bs, coordinates=coords, coordinate_names=[*NAMES, 'v_shr9'],
                target=ys, target_mixed=delta, distance=distance,
                maximum_mixed_bounded_residual=2*65535,
                proof='Every integer polynomial in the ten coordinates has mixed difference zero modulo 2^32; all coordinate increments are divisible by 2^16.',
                legal_points=[legal_point(b) for b in bs])


def residual_images():
    bs = np.arange(1 << 17, dtype=np.uint64)
    with np.errstate(over='ignore'):
        vv = early_inputs(bs); ys = wanted(vv)
    rows = []
    for name, h in zip((*NAMES, 'v_shr9'), vv+[vv[3] >> 9]):
        residual = (ys-h) & MASK
        count = len(np.unique(residual)); assert count > 65536
        rows.append(dict(source_name=name, distinct_residuals=count,
                         sequence_sha256=hashlib.sha256(residual.astype('<u4').tobytes()).hexdigest()))
    return dict(primary_b_range=[0, 1 << 17], maximum_function_of_one_shr16_values=65536,
                results=rows, proof='K*q or q*q has at most 65536 values when q is a literal SHR with count >=16.')


def inverse_fold_polynomials():
    rng = random.Random(8642828)
    bs = list(dict.fromkeys([*range(64), MASK, MASK-1, *(1 << j for j in range(32)),
                            *(rng.getrandbits(32) for _ in range(512))]))
    vv = [early_inputs(b) for b in bs]
    mons = [()]+[(j,) for j in range(5)]+list(combinations_with_replacement(range(5), 2))
    matrix = []
    for b, v in zip(bs, vv):
        fields = [b, v[1], v[6], v[7], v[8]]; row = []
        for mon in mons:
            value = 1
            for j in mon:
                value = value*fields[j] & MASK
            row.append(value)
        matrix.append(row)
    control = [(b*v[6]+v[0]) & MASK for b, v in zip(bs, vv)]
    assert consistent(matrix, control, 32)
    results = []
    y = z3.BitVec('fold_inverse_y', 32)
    for shift in range(1, 32):
        rhs = []
        for v in vv:
            t, value = wanted(v), 0
            for j in range(0, 32, shift):
                value ^= t >> j
            rhs.append(value)
        assert not consistent(matrix, rhs, 32)
        inv = y
        for j in range(shift, 32, shift):
            inv ^= z3.LShR(y, j)
        s = z3.SolverFor('QF_BV'); s.add(inv ^ z3.LShR(inv, shift) != y)
        assert s.check() == z3.unsat
        results.append(dict(shift=shift, status='inconsistent_full32', inverse_proved=True))
    return dict(samples=bs, monomial_indices=mons, arbitrary_coefficients=21,
                variables=['b', 'z', 'a', 'a_shr19', 'a_xor_c1'], results=results,
                single_mad_positive_control=True,
                scope='t is any one-MAD expression in the nine initial words; output t XOR (t>>s).')


def odd_product():
    bs = [2418667174, 2210462107, 1755490164, 924155396, 794392646,
          2711053979, 2682505225, 2917341350, 1225381534]
    rows = []
    for source, b in enumerate(bs):
        v = early_inputs(b); residual = (wanted(v)-v[source]) & MASK
        assert residual != 0 and residual % 65536 == 0
        rows.append(dict(source=source, name=NAMES[source], primary_b=b,
                         target=wanted(v), addend=v[source], residual=residual,
                         legal_point=legal_point(b)))
    # Remaining q*(v+1) form is ruled out at this one real input.
    v = early_inputs(0); multiplier = (v[3]+1) & MASK
    required_q = (wanted(v)*pow(multiplier, -1, 1 << 32)) & MASK
    assert multiplier & 1 and required_q >= 65536
    return dict(residual_points=rows,
                proof='LSB transfer requires t odd. Then t*q nonzero and divisible by 2^16 is impossible for 0<=q<2^16: reduce modulo 2^16 and invert odd t.',
                q_times_v_plus_one=dict(primary_b=0, multiplier=multiplier, target=wanted(v),
                                       required_q=required_q, legal_point=legal_point(0)))


def permutation_sums():
    def sums(width, shift):
        m, block = 1 << width, 1 << shift
        n = m//block
        sq = n*(n-1)//2; sq2 = n*(n-1)*(2*n-1)//6
        return (block*sq) % m, (block*sq2) % m, (block*block*sq2+block*(block-1)//2*sq) % m
    checked = 0
    for width in (4, 6, 8):
        m = 1 << width
        for shift in range(1, width):
            actual = tuple(sum(fun(t, t >> shift) for t in range(m)) % m
                           for fun in (lambda t, q:q, lambda t, q:q*q, lambda t, q:t*q))
            assert actual == sums(width, shift)
            assert actual[0] == actual[1] == m//2 and actual[2] != 0
            for t in range(1, m, 2):
                q = t >> shift; r = t*q % m
                if shift >= width//2 and r:
                    assert r % (1 << (width-shift)) != 0
                checked += 1
    rows = []
    for shift in range(16, 32):
        sq, sq2, stq = sums(32, shift)
        assert sq == sq2 == 1 << 31 and stq != 0
        rows.append(dict(shift=shift, sum_q=sq, sum_q_square=sq2, sum_tq=stq))
    return dict(results=rows, exhaustive_small_odd_products=checked,
                proof='A permutation of uint32 preserves the sum. t+K*q requires K even; t+q*q and t+t*q never preserve the sum.')


def two_mad_cases():
    n, t, q = len(NAMES), len(NAMES), len(NAMES)+1
    firsts = _instr_options(n-1, 'mad')
    lasts = [w for w in _instr_options(n+1, 'mad') if q in w]
    counts = Counter(); manifest = hashlib.sha256(); total = 0
    for first, last in product(firsts, lasts):
        row = dict(order=['mad', 'shr16', 'mad'], wiring=[list(first), [t], list(last)])
        models = low_models(row)
        a, b, c = last
        if not models:
            cert = 'lsb_dependency'
        elif q not in (a, b):
            assert c == q; cert = 'polynomial_plus_bounded_q_rectangle'
        elif a == b == q:
            cert = 'residual_image' if c < n else 'sum_q_square' if c == t else 'q_only_image'
        else:
            other = b if a == q else a
            if other >= n+2:
                if c < n:
                    cert = 'residual_image'
                elif c == t:
                    assert all(kk[2][other-(n+2)] == 255 for kk, _ in models)
                    cert = 'sum_Kq_requires_even_but_lsb_requires_odd'
                else:
                    cert = 'q_only_image'
            elif other == t:
                if c < n:
                    assert all(tmask == 255 for _, tmask in models)
                    cert = 'odd_tq_cannot_have_nonzero_16_divisible_residual'
                elif c == t:
                    cert = 'sum_tq'
                else:
                    # With q=0, every t<2^s maps to the same output.
                    cert = 'noninjective_g_of_t_wordcut'
            else:
                assert c == q and other == 3, (first, last, models)
                cert = 'q_times_v_plus_one_requires_large_q'
        counts[cert] += 1; total += 1
        manifest.update((json.dumps([first, last, cert], separators=(',', ':'))+'\n').encode())
    return dict(first_options=len(firsts), last_options=len(lasts), cases_per_shift=total,
                literal_shifts=list(range(16, 32)), counts=dict(counts), unresolved=0,
                deterministic_classified_manifest_sha256=manifest.hexdigest(),
                scope='MAD, SHR of its result, MAD; all nine inputs, all canonical operand roles and arbitrary constants. Same certificate holds for each count16..31.')


def two_shift_cases():
    rng = random.Random(8642831)
    bs = [*range(64), *(rng.getrandbits(32) for _ in range(1024))]
    vv = [early_inputs(b) for b in bs]; full = (1 << len(bs))-1
    sig = lambda f:sum(f(v) << i for i, v in enumerate(vv))
    desired = sig(lambda v:wanted(v) & 1)
    initial = [sig(lambda v, j=j:v[j] & 1) for j in range(9)]
    fields = [(src, s, sig(lambda v, j=src, s=s:(v[j] >> s) & 1)) for src in range(9) for s in range(33)]
    surviving = []; checked = 0
    for i, (s1, c1, a) in enumerate(fields):
        for s2, c2, b in fields[i:]:
            for hidx, h in enumerate(initial+[a, b, 0, full]):
                for pattern, actual in (('q1*q2+h', (a & b) ^ h),
                                        ('q1*h+q2', (a & h) ^ b),
                                        ('q2*h+q1', (b & h) ^ a)):
                    checked += 1
                    if actual != desired:
                        continue
                    assert (s1, c1) == (0, 16)
                    assert (s2, c2) in ((0, 16), (1, 0), (2, 0), (3, 9))
                    if pattern == 'q1*q2+h':
                        assert (s2, c2) == (0, 16) and hidx in (1, 2)
                        cert = 'residual_image'
                    elif pattern == 'q1*h+q2':
                        assert hidx in (9, 12)
                        cert = 'residual_image'
                    else:
                        assert hidx in (1, 2, 10, 12)
                        cert = 'polynomial_plus_bounded_q_rectangle'
                    surviving.append(dict(first=[s1, c1], second=[s2, c2], other=hidx,
                                          pattern=pattern, certificate=cert))
    assert len(surviving) == 20
    return dict(samples=bs, fields=len(fields), necessary_boolean_cases=checked,
                necessary_survivors=surviving, remaining=0,
                scope='Two initial-word literal right shifts then MAD. Consecutive shifts of one word are included by summing shift counts; amounts>=32 canonicalize to32.')


def audit_one_mad(evidence):
    def read(path):
        if not Path(path).exists():
            assert path.endswith('.json') and Path(path+'.gz').exists(), path
            path += '.gz'
        evidence[path] = digest(path)
        data = Path(path).read_bytes()
        return json.loads(gzip.decompress(data) if path.endswith('.gz') else data)
    first = read('codex/early_boundary863_first_shift.json')
    complete = read('codex/early_boundary863_one_mad_complete.json')
    expected, _ = make_manifest(True)
    assert complete['node'] == NODE and complete['input_names'] == list(NAMES)
    assert complete['completed'] == len(complete['manifest']) == len(complete['results']) == len(expected) == 233640
    assert set(map(key, complete['manifest'])) == set(map(key, complete['results'])) == set(map(key, expected))
    first_path = Path('codex/early_boundary863_first_shift.json')
    first_encoded = (first_path if first_path.exists() else Path(str(first_path)+'.gz')).read_bytes()
    source_encoded = first_encoded if complete['reused_first_shift'].endswith('.gz') else (
        first_encoded if first_path.exists() else gzip.decompress(first_encoded))
    assert complete['reused_first_shift_sha256'] == hashlib.sha256(source_encoded).hexdigest()
    assert first['completed'] == len(first['manifest']) == len(first['results']) == 74100
    first_results = {key(r):r for r in first['results']}
    assert len(first_results) == 74100
    counts = Counter()
    for r in complete['results']:
        if r['status'] == 'prior_excluded':
            assert key(r) in first_results
            r = first_results[key(r)]
        assert r['status'] in ('unsat', 'lsb_dependency_excluded')
        if r['status'] == 'lsb_dependency_excluded':
            assert not low_models(r)
        counts[r['status']] += 1
    old = read('codex/live_boundary863_arms_closed_audit.json')
    assert old['complete_manifest'] == 113265 and old['remaining'] == 0
    assert old['node'] == NODE and old['mandatory_mad'] and old['shift_count'] == 16
    for path, h in old['evidence_sha256'].items():
        assert digest(path) == h, path
    return dict(nine_word_new_early_manifest=233640, original_first_shift_manifest=74100,
                counts=dict(counts), old_five_word_grammar_reused=113265,
                all_low_truth_tables_independently_rechecked=True, remaining=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); evidence = {}
    dep = literal_shift_witnesses(); small = small_shift_facts(); runtime = runtime_amount_probe()
    print('296 initial shifts and32 bitwise-prefix shifts excluded', flush=True)
    rect = rectangle(); images = residual_images(); inverse = inverse_fold_polynomials()
    odd = odd_product(); sums = permutation_sums()
    tm = two_mad_cases(); print('Two-MAD cases:', tm['cases_per_shift'], tm['counts'], flush=True)
    ts = two_shift_cases(); print('Two-SHR LSB cases:', ts['necessary_boolean_cases'], '20 remaining closed', flush=True)
    om = audit_one_mad(evidence); print('One-MAD complete:', om['counts'], flush=True)
    proof_path = 'codex/early_boundary863_proofs.json'; proof = json.loads(Path(proof_path).read_bytes())
    evidence[proof_path] = digest(proof_path)
    assert proof['node'] == NODE and proof['polynomial']['all_sources_excluded']
    assert len(proof['polynomial']['results']) == 9
    result = dict(node=NODE, input_names=NAMES, literal_shift_dependencies=dep, small_shift_facts=small,
                  runtime_initial_amount_probe=runtime,
                  rectangle=rect, residual_images=images, inverse_fold_polynomial=inverse,
                  odd_product=odd, permutation_sums=sums, two_mad=tm, two_shr=ts,
                  one_mad=om, evidence_sha256=evidence, remaining_in_stated_core_grammar=0,
                  best_executable_cycles=864, new_executable=False,
                  scope='At most three straight-line MAD/XOR/AND/OR/literal-SHR operations, nine stated live inputs and arbitrary prepared root-dependent constants. ADD/SUB/MUL/literal-SHL normalize to MAD. Does not cover data-dependent shifts, division, comparison, FLOW or larger windows; no global kernel lower bound.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Stated three-op core grammar remaining:0; best executable864', flush=True)


if __name__ == '__main__':
    main()
