"""Join scoped division/comparison evidence without claiming all-ISA closure."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from codex.search_early_boundary863 import NODE


def audit():
    evidence = {}; decoded_sources = {}
    def read(path):
        actual = Path(path) if Path(path).exists() else Path(path+'.gz')
        raw = actual.read_bytes()
        decoded = gzip.decompress(raw) if actual.suffix == '.gz' else raw
        value = json.loads(decoded)
        evidence[str(actual)] = hashlib.sha256(raw).hexdigest()
        decoded_sources[path] = hashlib.sha256(decoded).hexdigest()
        assert value['node'] == NODE
        return value
    comparisons = read('codex/comparison_core863_audit.json')
    division_first = read('codex/division_first_t863_audit.json')
    division_last = read('codex/final_division863_audit.json')
    xor_tail = read('codex/division_xor_affine863_screen.json')
    div_bool = read('codex/division_boolean863_screen.json')
    masks = read('codex/bitmask_affine_tail863_audit.json')
    high = read('codex/joint_live_boundary863_affine_high.json')
    e_high = read('codex/early_e_affine_high863.json')
    assert comparisons['remaining_in_stated_three_op_core'] == 0
    assert len(comparisons['literal_right_cases']) == 297
    assert division_first['complete'] and division_first['final_family_status'] == 'excluded'
    assert len(division_first['results']) == 18
    assert division_last['remaining_in_stated_family'] == 0
    assert xor_tail['complete'] and xor_tail['family_cases'] == xor_tail['excluded'] == 243
    assert div_bool['excluded'] and len(div_bool['results']) == 18
    assert len(masks['live_operand_masks']) == 18
    assert masks['joint_affine_tail_cases']['total'] == 12600
    assert len(high['results']) == 8 and all(r['status'] == 'unsat' for r in high['results'])
    assert e_high['source_name'] == 'e' and e_high['status'] == 'unsat'
    return dict(node=NODE, evidence_sha256=evidence, decoded_source_sha256=decoded_sources,
                closed_scopes=[
                    'At most3 MAD/XOR/AND/OR/SHL/SHR/< /== operations, arbitrary operands on nine words/root-only constants; no division.',
                    'A first DIV/REM/CDIV of an initial word/constant followed by at most two MAD/XOR/AND/OR; initial denominators are invalid at a legal zero witness, nonzero constant denominators all covered.',
                    'A final single DIV/CDIV after at most two non-division core/comparison ops, arbitrary computed denominator; any-prefix final REM also impossible.',
                    'An initial-word DIV/CDIV with a comparison and a final MAD/bitop, either D,C,T or C,D,T. Comparison relaxed to arbitrary per-input Boolean; all constant divisors covered.',
                    '243 constant-divisor initial DIV/REM/CDIV, live XOR, affine-tail cases; AND/OR live-mask counterparts impossible for arbitrary q.',
                    'Complete12600 joint MAD,SHR16,BIT,affine-MAD canonical graphs, BIT=XOR/AND/OR, eight inputs excluding e; only this four-op topology.',
                    'out=G*(h XOR q)+H for any of nine initial h and arbitrary q<65536, using inverse affine high-half necessary constraints.',
                ],
                still_open=[
                    'Computed DIV/REM/CDIV in the middle, with an initial-word or computed bypass.',
                    'Division first mixed with another division or variable shifts; REM combined with a comparison and mixing final op.',
                    'Other four-op joint topologies, extra live inputs, state encodings and larger windows.',
                    'Other kernel algorithms, memory/FLOW/cross-lane mechanisms, and actual scheduling of any future candidate.',
                ],
                bounded_q_corollaries=dict(floor_constant_divisor_at_least=65536,
                    ceil_constant_divisor_at_least=65537, rem_constant_divisor_at_most=65536,
                    proof='Each bound guarantees q<65536 for any uint32 numerator. The arbitrary-prefix high-half necessary constraints apply when the remaining structure is G*(h XOR q)+H.'),
                no_global_optimality_claim=True, best_executable_cycles=864, new_executable=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args(); result = audit()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Seven scoped evidence groups joined; open division placements retained; best executable864', flush=True)


if __name__ == '__main__':
    main()
