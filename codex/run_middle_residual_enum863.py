"""Run the exact odd-parameter scanner and preserve its finite dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np

from codex.screen_middle_division_residual863 import pool
from codex.search_early_boundary863 import MASK, NAMES, NODE, early_inputs, wanted


def dataset():
    points = pool(); values = [early_inputs(b) for b in points]
    targets = [wanted(v) for v in values]; cases = []
    for source in range(9):
        witness = {}
        for b, v, y in zip(points, values, targets):
            witness.setdefault((y-v[source]) & MASK, b)
        rs = list(witness); random.Random(8642863+source).shuffle(rs)
        cases.append(dict(id=source, width=32, source_name=NAMES[source], residuals=rs,
                          primary_witnesses=[witness[r] for r in rs]))
    return dict(node=NODE, input_names=NAMES, points=points, cases=cases,
                definition='residual=target-initial_word modulo2^32',
                all_points_reachable_via_prior_legal_point_construction=True)


def encode(cases):
    parts = [str(len(cases))]
    for case in cases:
        rs = case['residuals']
        parts.append(f"{case['id']} {case['width']} {len(rs)}")
        parts.append(' '.join(map(str, rs)))
    return '\n'.join(parts)+'\n'


def run(binary, data, output, workers=4):
    started = time.monotonic(); results = []; source = Path('codex/enumerate_middle_residual863.cpp')
    report = dict(node=NODE, dataset=data, scanner_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), results=results,
                  complete=False, best_executable_cycles=864, new_executable=False)
    output.write_text(json.dumps(report, indent=2)+'\n')
    proc = subprocess.Popen([str(binary.resolve()), str(workers)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    proc.stdin.write(encode(data['cases'])); proc.stdin.close()
    for line in proc.stdout:
        row = json.loads(line); case = next(c for c in data['cases'] if c['id'] == row['id'])
        assert row['width'] == case['width']
        if row['complete']:
            assert row['evaluated_odd_parameters'] == 1 << (row['width']-1)
        models = row['models']; aa = np.array([a for a, _ in models], dtype=np.uint64)
        assert len(set(aa.tolist())) == len(aa) and np.all(aa & 1)
        counts = np.zeros(len(aa), dtype=np.uint32)
        mask = (1 << row['width'])-1; half = 1 << (row['width']-1)
        with np.errstate(over='ignore'):
            for r in case['residuals']:
                counts += ((aa*np.uint64(r)) & mask) > half
        assert counts.tolist() == [h for _, h in models] and np.all(counts <= 1)
        row.update(source_name=case['source_name'], bounded_models=[a for a, h in models if h == 0])
        results.append(row)
        print(row['source_name'], 'complete', row['complete'], 'models', len(models),
              'bounded', len(row['bounded_models']), 'seconds', row['seconds'], flush=True)
        output.write_text(json.dumps(report, indent=2)+'\n')
    code = proc.wait(); stderr = proc.stderr.read()
    assert code == 0, (code, stderr)
    assert len(results) == len(data['cases'])
    report.update(complete=all(r['complete'] for r in results), seconds=round(time.monotonic()-started, 6),
                  all_one_high_excluded=all(r['complete'] and not r['models'] for r in results),
                  all_bounded_excluded=all(r['complete'] and not r['bounded_models'] for r in results),
                  all_sat_models_require_full32_function_verification=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args(); run(args.binary, dataset(), args.output, args.workers)


if __name__ == '__main__':
    main()
