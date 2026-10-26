"""Enumerate normalized constant-XOR affine prefixes using exact module tests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from codex.screen_affine_head_keys863 import decode
from codex.search_cross_encoding863 import necessary_inputs

ROOT = Path(__file__).resolve().parent


def canonical_pair(a, b):
    a, b = a & 1023, b & 1023
    if a > 512:
        a, b = (-a) & 1023, (-b-1) & 1023
    if a % 2 == 0:
        b &= ~1
    return a, b


def canonical_triple(a, b, key):
    a, b = canonical_pair(a, b)
    key &= 1023
    if key & 512:
        key ^= 1023
    if a % 2 == 0:
        key &= ~1
    return a, b, key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, default=ROOT/'affine_head_cubes863_joint.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit-pairs', type=int)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--control', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error('shard must be in 0..shards-1')
    prior = json.loads(args.prior.read_text())
    all_pairs = decode(prior['surviving_pairs'])
    normalized = sorted({canonical_pair(int(a), int(b)) for a, b in all_pairs})
    all_set = {tuple(map(int, pair)) for pair in all_pairs}
    assert set(normalized).issubset(all_set)
    pairs = normalized[args.shard::args.shards]
    if args.limit_pairs is not None:
        pairs = pairs[:args.limit_pairs]
    triples = [(a, b, key) for a, b in pairs for key in range(0, 512, 2 if a % 2 == 0 else 1)]
    if args.control:
        triples = [(6, 19, 42), canonical_triple(6, 19, 42)]
    source = ROOT/'screen_head_keyed_bivariate863.cpp'
    executable = Path(f'/tmp/aopt-head-keyed-module863-{os.getpid()}')
    compiled = subprocess.run(['clang++', '-std=c++17', '-O3', '-Wall', '-Wextra',
        '-o', str(executable), str(source)], capture_output=True, text=True)
    if compiled.returncode:
        raise RuntimeError(compiled.stderr)
    samples = list(dict.fromkeys(necessary_inputs(11)+[32, 64, 96]))
    data = str(len(samples))+'\n'+' '.join(map(str, samples))+'\n'
    data += ''.join(f'{a} {b} {key}\n' for a, b, key in triples)
    start = time.monotonic()
    done = subprocess.run([str(executable), 'control' if args.control else 'target'],
        input=data, text=True, capture_output=True)
    if done.returncode:
        raise RuntimeError(done.stderr)
    survivors = []
    for line in done.stdout.splitlines():
        a, b, key, count, *keys = map(int, line.split())
        assert len(keys) == count
        survivors.append(dict(a=a, b=b, inner_key=key, output_keys=keys))
    assert f'completed tested={len(triples)} survivors={len(survivors)}' in done.stderr
    report = dict(width=11, degree=4, control=args.control, shard=args.shard, shards=args.shards,
        total_canonical_pairs=len(normalized), tested_pairs=len({(a,b) for a,b,k in triples}), tested_triples=len(triples),
        all_shard_pairs_tested=args.limit_pairs is None and not args.control,
        samples=samples, survivors=survivors, remaining_triples=len(survivors),
        remaining_triple_key_assignments=sum(len(row['output_keys']) for row in survivors),
        positive_witness_retained=all(any((row['a'],row['b'],row['inner_key']) == triple
            and 0 in row['output_keys'] for row in survivors) for triple in triples) if args.control else None,
        source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (source, ROOT/'screen_head_bivariate863.cpp')},
        prior_evidence=dict(path=args.prior.name, sha256=hashlib.sha256(args.prior.read_bytes()).hexdigest()),
        seconds=round(time.monotonic()-start, 3), progress=done.stderr.splitlines(),
        result='sample_module_excluded' if not survivors else 'necessary_module_memberships_survive',
        new_executable=False,
        scope='Degree-four arbitrary integer polynomial in x,u with u=(A*x+B) XOR K. '
              'Only enumerated triples are covered until all shards complete; survivors are '
              'sample-domain module memberships, not executable candidates.')
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('result', 'total_canonical_pairs', 'tested_pairs',
        'tested_triples', 'remaining_triples', 'seconds', 'positive_witness_retained')}))


if __name__ == '__main__':
    main()
