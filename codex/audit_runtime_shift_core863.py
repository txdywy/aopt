"""Full32 local three-op core audit allowing data-dependent right shifts.

T means MAD/XOR/AND/OR (ADD/SUB/MUL/literal SHL normalize to MAD), S means
unsigned SHR with arbitrary initial/earlier/root-constant operands. The
eight T/S orders are covered by finite necessary systems and structural
lemmas. This is not an all-ISA or whole-kernel lower bound.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import combinations_with_replacement, product
import json
from pathlib import Path
import random

import z3

from codex.close_early_boundary863 import rectangle
from codex.search_early_boundary863 import early_inputs, wanted, NAMES, NODE, MASK, legal_point
from codex.screen_xor_mad_polynomial import consistent


def monomials(fields):
    return [1, *fields, *((a*b) & MASK for i, a in enumerate(fields) for b in fields[i:])]


def base_fields(b, v):
    return [b, v[1], v[6], v[7], v[8]]


def unary_bitwise_fit(qs, ys):
    for bit in range(16):
        q = sum(((v >> bit) & 1) << i for i, v in enumerate(qs))
        y = sum(((v >> bit) & 1) << i for i, v in enumerate(ys))
        if y not in (0, 15, q, 15 ^ q):
            return False
    return True


def high_samples():
    rng = random.Random(8642843)
    bs = [rng.getrandbits(32) for _ in range(2048)]
    points = [(b, early_inputs(b)) for b in bs]
    return [(b, v, wanted(v)) for b, v in points if wanted(v) >= 1 << 31 and min(v) >= 32][:512]


def high_output_lemmas(samples):
    value, count = z3.BitVecs('right_source right_count', 32)
    s = z3.SolverFor('QF_BV')
    s.add(z3.UGE(z3.LShR(value, count), 1 << 31), count != 0)
    assert s.check() == z3.unsat
    pair = None
    for b, v, y in samples:
        other_b = b ^ 65536; w = early_inputs(other_b); other_y = wanted(w)
        if other_y >= 1 << 31 and min(w) >= 32:
            assert [x & 1 for x in v] == [x & 1 for x in w] and (y ^ other_y) & 1
            pair = [b, other_b]; break
    assert pair is not None
    return dict(full32_high_output_requires_zero_shift='unsat', primary_pair=pair,
                legal_points=[legal_point(b) for b in pair],
                proof='For a high output, every right shift on its value-operand chain must have count0. A T-only ancestor has equal LSB on this pair, while both target outputs are high and have different LSBs.')


def rectangle_checks():
    rect = rectangle(); vv = [early_inputs(b) for b in rect['primary_b']]
    yy = [wanted(v) & 65535 for v in vv]
    assert len(set(yy)) == 4 and min(min(v) for v in vv) >= 32
    rows = []
    for source, name in enumerate(NAMES):
        for shift in range(33):
            qs = [(v[source] >> shift) & 65535 for v in vv]
            matrix = [[1, q, q*q & 65535] for q in qs]
            assert not consistent(matrix, yy, 16), (source, shift)
            assert consistent(matrix, [(q*q+5*q+7) & 65535 for q in qs], 16)
            assert not unary_bitwise_fit(qs, yy)
            assert unary_bitwise_fit(qs, [q ^ 0xA55A for q in qs])
            rows.append(dict(source=source, name=name, shift=shift,
                             quadratic_status='inconsistent_mod16', unary_bitwise_status='inconsistent_mod16',
                             positive_controls=True))
    return dict(rectangle=rect, target_low16=yy, cases=rows,
                MAD_amount_proof='A first MAD has common low16 and mixed difference0 on the rectangle. At most1 amount can be>=32, otherwise zero shift results collide. At least3 amounts thus equal the same r<32; mixed difference0 forces the fourth to equal r as well. A last MAD is at most quadratic in the sole shifted low word; a last bitop is unary bitwise. All297 possibilities fail.',
                initial_amount_proof='Every initial word is>=32 on all4 states. An initial-word shift amount makes that shift result0. The other shift is then literal or has count0; the same297 unary/quadratic tests apply.')


def high_middle_checks(samples):
    fields = [base_fields(b, v) for b, v, _ in samples]
    rhs = [y for _, _, y in samples]
    base = [monomials(f) for f in fields]
    assert not consistent(base, rhs, 32)
    assert consistent(base, [(b*v[6]+v[1]) & MASK for b, v, _ in samples], 32)
    rows = []; legal = set()
    for source, name in enumerate(NAMES):
        for shift in range(33):
            qq = [v[source] >> shift for _, v, _ in samples]
            matrix = [monomials(f+[q]) for f, q in zip(fields, qq)]
            assert not consistent(matrix, rhs, 32), (source, shift)
            assert consistent(matrix, [(b*q+v[1]) & MASK for (b, v, _), q in zip(samples, qq)], 32)
            witness = None
            for bit in range(32):
                seen = {}
                for (b, v, y), q in zip(samples, qq):
                    signature = tuple([(x >> bit) & 1 for x in v]+[(q >> bit) & 1])
                    out = y >> bit & 1
                    if signature in seen and seen[signature][1] != out:
                        witness = dict(bit=bit, primary_pair=[seen[signature][0], b], signature=list(signature)); break
                    seen[signature] = (b, out)
                if witness is not None:
                    break
            assert witness is not None, (source, shift)
            legal.update(witness['primary_pair'])
            rows.append(dict(source=source, name=name, shift=shift,
                             quadratic_status='inconsistent_full32_high_outputs', positive_control=True,
                             bitwise_high_output_witness=witness))
    return dict(primary_samples=[b for b, _, _ in samples], base_coefficients=21,
                shifted_coefficients=28, base_status='inconsistent_full32_high_outputs',
                cases=rows, legal_bitwise_witness_points=[legal_point(b) for b in sorted(legal)],
                proof='A final SHR producing a high output has count0, so its middle MAD/bitwise data word must already equal the target on these samples. The MAD relaxation is quadratic in five base words and q; a bitop preserves same-position indistinguishability.')


def dependent_second_shift_checks():
    rng = random.Random(8642844)
    bs = [rng.getrandbits(32) for _ in range(16384)]
    vv = [early_inputs(b) for b in bs]; ys = [wanted(v) for v in vv]
    rows = []; legal = set()
    for source, name in enumerate(NAMES):
        for shift in range(33):
            large = ((8191 if source == 7 else MASK) >> shift) >= 32
            indices = [i for i, v in enumerate(vv)
                       if ((v[source] >> shift) >= 32 if large else (v[source] >> shift) == 0)][:512]
            assert len(indices) >= 64
            matrix = [monomials(base_fields(bs[i], vv[i])+([vv[i][source] >> shift] if large else []))
                      for i in indices]
            rhs = [ys[i] for i in indices]
            assert not consistent(matrix, rhs, 32), (source, shift)
            control = [((bs[i]*(vv[i][source] >> shift) if large else bs[i]*vv[i][6])+vv[i][1]) & MASK
                       for i in indices]
            assert consistent(matrix, control, 32)
            seen = {}; witness = None
            for i in indices:
                signature = tuple(x >> 31 for x in vv[i]); bit = ys[i] >> 31
                if signature in seen and seen[signature][1] != bit:
                    witness = dict(primary_pair=[bs[seen[signature][0]], bs[i]],
                                   top_signature=list(signature)); break
                seen[signature] = (i, bit)
            assert witness is not None, (source, shift)
            legal.update(witness['primary_pair'])
            rows.append(dict(source=source, name=name, shift=shift,
                             condition='first_q>=32' if large else 'first_q=0', samples=len(indices),
                             selected_indices_sha256=hashlib.sha256(b''.join(i.to_bytes(4, 'little') for i in indices)).hexdigest(),
                             quadratic_status='inconsistent_full32', positive_control=True,
                             bitwise_top_witness=witness))
    return dict(primary_pool=bs, cases=rows,
                minimum_selected_samples=min(r['samples'] for r in rows),
                legal_bitwise_witness_points=[legal_point(b) for b in sorted(legal)],
                proof='Second count=first_q. When first_q>=32 the second result is0; when first_q=0 it is an initial word or root constant. A last MAD is quadratic in base words plus first_q (or base words alone). A last bitop cannot distinguish the recorded identical top-bit signatures.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); evidence = {}
    def read(path):
        raw = Path(path).read_bytes(); evidence[path] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    fixed = read('codex/early_boundary863_closed_audit.json')
    amount = read('codex/computed_shift_amount863_audit.json')
    assert fixed['remaining_in_stated_core_grammar'] == 0 and fixed['node'] == NODE
    assert amount['first_bitwise_amount_remaining'] == 0 and amount['node'] == NODE
    samples = high_samples(); high = high_output_lemmas(samples); rect = rectangle_checks()
    print('High-output ancestor and297 rectangle conditions closed', flush=True)
    middle = high_middle_checks(samples)
    print('Middle MAD/bitop on high outputs:297 conditions closed', flush=True)
    second = dependent_second_shift_checks()
    print('Data-dependent second right shift:297 conditional systems closed', flush=True)
    partitions = {
        'TTT':'No downward bit transfer: target LSB flips on b16 while all initial LSBs stay fixed.',
        'STT':'Initial-word amount makes shift0 on the rectangle; prepared-root amounts use the fixed-shift core audit.',
        'TST':'Initial/root amounts as above; a BIT-computed amount uses the computed-amount audit; a MAD-computed amount uses the rectangle. Constant data sources and t>>t are separately closed.',
        'TTS':'High-output ancestor pair: final right shift must have count0, and its T-only data ancestor has equal LSB on the pair.',
        'TSS':'High-output ancestor pair along every value-operand shift chain.',
        'STS':'For initial/root constant value ancestors use the high pair. For middle T value ancestors, high-output quadratic/bitwise tests cover every first fixed shift; initial-word first amounts make q=0 on all selected high samples.',
        'SST':'Both prepared-root amounts use fixed-shift core. An initial-word amount uses the rectangle. Second amount=first_q uses297 conditional quadratic/bitwise tests. A constant first source reduces to prepared-root amounts.',
        'SSS':'High-output ancestor pair along the value-operand shift chain.',
    }
    assert set(partitions) == {''.join(p) for p in product('TS', repeat=3)}
    result = dict(node=NODE, input_names=NAMES, high_output=high, rectangle=rect,
                  high_middle=middle, dependent_second=second, T_S_order_partition=partitions,
                  T_operations=['MAD', 'XOR', 'AND', 'OR'],
                  S_operation='uint32 right shift with initial/earlier/root-constant value and amount',
                  literal_left_shift_normalizes_to_MAD=True, remaining_in_stated_three_op_core=0,
                  evidence_sha256=evidence, best_executable_cycles=864, new_executable=False,
                  scope='At most3 straight-line MAD/XOR/AND/OR/SHR operations over nine stated live words, arbitrary root-prepared constants and arbitrary data-dependent SHR operands. No DIV/REM/CDIV/comparison/data-dependent SHL/FLOW/load/store or larger windows; no global kernel optimality claim.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Stated three-op core remaining0; best executable864', flush=True)


if __name__ == '__main__':
    main()
