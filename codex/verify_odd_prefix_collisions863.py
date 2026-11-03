"""Independently replay each recorded prefix collision with full32 keys."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import z3

from codex.audit_odd_prefix_collisions863 import evaluate, key_grid, manifest
from codex.search_early_boundary863 import MASK, NODE, early_inputs, wanted


def verify(path):
    raw = path.read_bytes(); source = json.loads(raw)
    assert source['node'] == NODE and source['manifest'] == manifest()
    required = {(i, tuple(p)) for i, row in enumerate(source['manifest']) for p in row['odd_key_parities']}
    found = set(); calls = []
    for case in source['cases']:
        identity = case['prefix_index'], tuple(case['key_parity'])
        assert identity in required and identity not in found
        found.add(identity)
        assert case['parameter_grid'] == [list(k) for k in key_grid(case['op'], case['key_parity'])]
        covered = set()
        for witness in case['witnesses']:
            i, j = witness['primary_pair']; v, w = early_inputs(i), early_inputs(j)
            root = z3.BitVecs('odd_prefix_k0 odd_prefix_k1', 32)
            solver = z3.SolverFor('QF_BV'); solver.set(timeout=10000)
            solver.add(*(k & 1 == p for k, p in zip(root, case['key_parity'])))
            left = evaluate(case['op'], case['wiring'], v, root)
            right = evaluate(case['op'], case['wiring'], w, root)
            solver.add(left != right)
            status = solver.check(); assert status == z3.unsat, (case, status)
            for keys, common in zip(case['parameter_grid'], witness['common_grid_values']):
                assert evaluate(case['op'], case['wiring'], v, keys) == common
                assert evaluate(case['op'], case['wiring'], w, keys) == common
            for addend in witness['addends']:
                assert ((wanted(v)-v[addend]) & MASK) != ((wanted(w)-w[addend]) & MASK)
                covered.add(addend)
            calls.append(dict(prefix_index=case['prefix_index'], key_parity=case['key_parity'],
                              primary_pair=[i, j], all_same_parity_full32_keys='unsat'))
        assert covered == set(range(9))-set(case['remaining_addends'])
    assert found == required
    assert source['excluded'] == sum(not c['remaining_addends'] for c in source['cases'])
    return dict(node=NODE, source=str(path), source_sha256=hashlib.sha256(raw).hexdigest(),
                canonical_prefixes=len(source['manifest']), parity_cases=len(required), calls=calls,
                remaining=sum(bool(c['remaining_addends']) for c in source['cases']),
                full32_parameter_queries=len(calls), best_executable_cycles=864, new_executable=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args(); result = verify(args.source)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Full32 universal-key collision checks', result['full32_parameter_queries'], 'remaining', result['remaining'], flush=True)


if __name__ == '__main__':
    main()
