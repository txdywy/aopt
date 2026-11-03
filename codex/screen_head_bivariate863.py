"""Exact degree-four polynomial relaxation of t=A*x+B; u=t XOR x; MAD; MAD.

The accelerator performs integer elimination, no timeout/SMT heuristics.
This screen can exclude this family but cannot return an executable kernel.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from codex.screen_affine_head_keys863 import decode
from codex.search_cross_encoding863 import necessary_inputs

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'screen_head_bivariate863.cpp'


def compile_accelerator(destination=Path('/tmp/aopt-head-bivariate863')):
    result = subprocess.run(['clang++', '-std=c++17', '-O3', '-Wall', '-Wextra',
        '-o', str(destination), str(SOURCE)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return destination


def run(artifact, executable, degree=4, positive=False, pairs=None):
    previous = json.loads(artifact.read_text())
    assert previous['width'] == 11 and not previous['control']
    all_pairs = decode(previous['surviving_pairs'])
    canonical = all_pairs[np.all(all_pairs < 1024, axis=1)]
    original_pairs = {tuple(map(int, row)) for row in all_pairs}
    normalized = {(a & 1023, b & 1023) for a, b in original_pairs}
    assert normalized == {tuple(map(int, row)) for row in canonical}
    assert len(original_pairs) == 4*len(normalized)
    for a, b in normalized:
        assert all((a+1024*i, b+1024*j) in original_pairs for i in range(2) for j in range(2))
    if pairs is None:
        pairs = [(6, 19)] if positive else sorted(normalized)
    samples = list(dict.fromkeys(necessary_inputs(11)+[32, 64, 96]))
    data = str(len(samples))+'\n'+' '.join(map(str, samples))+'\n'
    data += ''.join(f'{a} {b}\n' for a, b in pairs)
    start = time.monotonic()
    done = subprocess.run([str(executable), str(degree), 'control' if positive else 'target'],
        input=data, text=True, capture_output=True)
    if done.returncode:
        raise RuntimeError(done.stderr)
    survivors = []
    for line in done.stdout.splitlines():
        a, b, count, *keys = map(int, line.split())
        assert len(keys) == count and set(keys).issubset(range(0, 1024, 2))
        survivors.append(dict(a=a, b=b, output_keys=keys))
    assert f'completed tested={len(pairs)} survivors={len(survivors)}' in done.stderr
    return dict(width=11, control=positive, degree=degree,
        input_source='u=(A*x+B) XOR x', samples=samples,
        total_unrestricted_affine_pairs=1 << 22, prior_surviving_affine_pairs=len(all_pairs),
        canonical_pairs=len(normalized), tested_pairs=len(pairs),
        all_canonical_pairs_tested=not positive and len(pairs) == len(normalized),
        normalized_output_keys_per_pair=512, survivors=survivors,
        remaining_pairs=len(survivors), remaining_pair_key_assignments=sum(len(row['output_keys']) for row in survivors),
        positive_witness_retained=any(row['a'] == 6 and row['b'] == 19 and 0 in row['output_keys']
                                     for row in survivors) if positive else None,
        prior_evidence=dict(path=artifact.name, sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()),
        accelerator_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        progress=done.stderr.splitlines(), seconds=round(time.monotonic()-start, 3),
        result='sample_module_excluded' if not survivors else 'necessary_module_memberships_survive',
        new_executable=False,
        normalizations=['A and B highest bits zero: u changes by an affine polynomial modulo 2048',
            'O highest/lowest bits zero: constant addition and polynomial complementation'],
        scope='Independent coefficients for all bivariate monomials x^i*u^j of total degree '
              f'at most {degree}. Input-x XOR branch only. Survivors are sample-domain '
              'module memberships, not four-operation or full32 candidates.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, default=ROOT/'affine_head_cubes863_joint.json')
    parser.add_argument('--degree', type=int, default=4, choices=(2, 3, 4))
    parser.add_argument('--control', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    executable = compile_accelerator()
    result = run(args.prior, executable, args.degree, args.control)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('result', 'tested_pairs', 'remaining_pairs',
        'remaining_pair_key_assignments', 'seconds', 'positive_witness_retained')}))


if __name__ == '__main__':
    main()
