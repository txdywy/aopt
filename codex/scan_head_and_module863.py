"""Necessary polynomial-module screens for AND/OR after one affine MAD.

OR needs no separate module enumeration: t OR z = t+z-(t AND z), so
polynomials of bounded total degree in x and either result are equivalent
when z is x or a constant. No XOR-specific constant normalization is reused.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from codex.scan_head_keyed_module863 import canonical_pair
from codex.screen_affine_head_keys863 import decode
from codex.search_cross_encoding863 import necessary_inputs

ROOT=Path(__file__).resolve().parent


def canonical_keyed(a,b,key):
    key &= 2047
    if key&1024:
        key ^= 2047  # t AND ~K = t-(t AND K), an affine module substitution.
    a,b=canonical_pair(a,b)
    if a%2==0:key &= ~1
    return a,b,key


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',choices=('input','key'),required=True)
    parser.add_argument('--limit-pairs',type=int)
    parser.add_argument('--only-odd',action='store_true')
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--control',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 0<=args.shard<args.shards:parser.error('shard must be in 0..shards-1')
    prior_path=ROOT/'affine_head_cubes863_joint.json'
    prior=json.loads(prior_path.read_text())
    pairs=[tuple(map(int,row)) for row in decode(prior['surviving_pairs'])]
    if args.source=='key':pairs=sorted({canonical_pair(*pair) for pair in pairs})
    total_pairs=len(pairs)
    if args.only_odd:pairs=[pair for pair in pairs if pair[0]&1]
    pairs=pairs[args.shard::args.shards]
    if args.limit_pairs is not None:pairs=pairs[:args.limit_pairs]
    triples=([(a,b,-1) for a,b in pairs] if args.source=='input' else
        [(a,b,k) for a,b in pairs for k in range(0,1024,2 if a%2==0 else 1)])
    if args.control:
        triples=([(6,19,-1)] if args.source=='input' else
                 [(6,19,42),canonical_keyed(6,19,42)])
    source=ROOT/'screen_head_and_module863.cpp'
    executable=Path(f'/tmp/aopt-head-and-module863-{os.getpid()}')
    compiled=subprocess.run(['clang++','-std=c++17','-O3','-Wall','-Wextra',
        '-o',str(executable),str(source)],capture_output=True,text=True)
    if compiled.returncode:raise RuntimeError(compiled.stderr)
    samples=list(dict.fromkeys(necessary_inputs(11)+[32,64,96]))
    data=str(len(samples))+'\n'+' '.join(map(str,samples))+'\n'
    data+=''.join(f'{a} {b} {k}\n' for a,b,k in triples)
    start=time.monotonic()
    done=subprocess.run([str(executable),args.source,'control' if args.control else 'target'],
                        input=data,text=True,capture_output=True)
    if done.returncode:raise RuntimeError(done.stderr)
    survivors=[]
    for line in done.stdout.splitlines():
        a,b,k,count,*keys=map(int,line.split())
        assert len(keys)==count
        survivors.append(dict(a=a,b=b,inner_key=k,output_keys=keys))
    assert f'completed tested={len(triples)} survivors={len(survivors)}' in done.stderr
    report=dict(width=11,degree=4,source=args.source,control=args.control,
        shard=args.shard,shards=args.shards,
        total_pairs=total_pairs,tested_pairs=len({(a,b) for a,b,k in triples}),
        tested_triples=len(triples),only_odd=args.only_odd,
        all_pairs_tested=args.shards==1 and args.limit_pairs is None and not args.only_odd and not args.control,
        all_shard_pairs_tested=args.limit_pairs is None and not args.only_odd and not args.control,
        samples=samples,survivors=survivors,remaining_triples=len(survivors),
        remaining_triple_key_assignments=sum(len(r['output_keys']) for r in survivors),
        positive_witness_retained=all(any((r['a'],r['b'],r['inner_key'])==t and
            0 in r['output_keys'] for r in survivors) for t in triples) if args.control else None,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source,ROOT/'screen_head_bivariate863.cpp')},
        prior_evidence=dict(path=prior_path.name,sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest()),
        seconds=round(time.monotonic()-start,3),progress=done.stderr.splitlines(),
        result='sample_module_excluded' if not survivors else 'necessary_module_memberships_survive',
        new_executable=False,
        scope='Arbitrary degree-four integer polynomial in x,u, u=(A*x+B) AND x/K; '
              'same module covers corresponding OR. Tested tuples only; any survivors '
              'require exact DAG and full32 verification before being candidates.')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('result','source','tested_pairs','tested_triples',
        'remaining_triples','remaining_triple_key_assignments','seconds','positive_witness_retained')}))


if __name__=='__main__':
    main()
