"""Partition the affine/constant-bitwise/live-MAD frontier into concrete roles.

Each Z3 query uses a separate subprocess. This covers this fixed graph only,
not all three-op live-input programs. Earlier raw UNKNOWN results remain.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys


def partitions(first_sources=('a','keyed')):
    # Live words: 0=a, 1=prior, 2=a XOR K, 3=first MAD, 4=XOR.
    # Constants 5 and 6 can be renamed freely within the last instruction.
    scaled=[(4,5,t)for t in (0,1,2,3,4,6)]
    products=[(*sorted((s,4)),t)for s in range(5)for t in range(6)]
    addends=[(a,b,4)for a,b in itertools.combinations_with_replacement((0,1,2,3,5),2)
             if (a,b)!=(5,5)]
    roles=scaled+products+addends
    assert len(roles)==len(set(roles))==50
    for first in first_sources:
        for role in roles:
            yield first,role


def canonical_role(role):
    a,b,c=role
    a,b=sorted((a,b))
    if a==4 and b>=5 and c>=5:
        # Independent constants also include the case D*q+D.
        return (4,5,6)
    if a>=5 and b>=5:
        # Constant product plus q is contained in D*q+E.
        assert c==4
        return (4,5,6)
    mapping={}
    next_constant=5
    out=[]
    for v in (a,b,c):
        if v>=5 and v not in mapping:
            mapping[v]=next_constant;next_constant+=1
        out.append(mapping.get(v,v))
    return tuple(out)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds',type=float,default=6)
    ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--bitwise',choices=('^','&','|'),default='^')
    ap.add_argument('--first-sources',default='a,keyed')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    first_sources=args.first_sources.split(',')
    if any(s not in ('a','keyed')for s in first_sources):raise ValueError('first sources a or keyed required')
    previous=Path(__file__).with_name('free_keys863_complete_grammar.json')
    report=dict(width=11,first_sources=first_sources,bitwise=args.bitwise,middle_source='arbitrary constant',
        first_prior_reduction='At width 11, prior=a-C0; replace B by B-A*C0 in the first affine MAD.',
        prior_single_input_evidence=dict(path=previous.name,sha256=hashlib.sha256(previous.read_bytes()).hexdigest()),
        scope='First MAD affine in the named sources (a also covers prior); middle bitwise op with a constant; last MAD with arbitrary live roles.',
        results=[],new_executable=False)
    def save():
        report['counts']=dict(Counter(r['result']for r in report['results']))
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    def run(job):
        first,role=job
        row=dict(first_source=first,last_operands=list(role))
        if first=='keyed'and not set(role).intersection((0,1)):
            return dict(row,result='covered_by_prior_single_input_proof',
                reduction='Set x=a XOR K. Then F(a XOR C1) XOR O=F(x XOR (K XOR C1)) XOR O; only x and its two computed words are read.')
        source='from codex.search_keyed_live_middle863 import search;import json;print(json.dumps(search('+repr(('mad',args.bitwise,'mad'))+',11,'+str(args.seconds)+','+str(max(1,int(args.seconds*990)))+',first_source='+repr(first)+',middle_source="key",last_operands='+repr(role)+')))'
        completed=subprocess.run([sys.executable,'-c',source],capture_output=True,text=True)
        if completed.returncode:raise RuntimeError(completed.stderr)
        return dict(row,**{'result':json.loads(completed.stdout)['result'],
                          'raw':json.loads(completed.stdout)})
    with ThreadPoolExecutor(max_workers=args.workers)as pool:
        pending=[pool.submit(run,job)for job in partitions(first_sources)]
        for future in as_completed(pending):
            report['results'].append(future.result());save()
            if len(report['results'])%10==0:print(json.dumps(dict(done=len(report['results']),counts=report['counts'])),flush=True)
    report['results'].sort(key=lambda r:(r['first_source'],r['last_operands']))
    save()


if __name__=='__main__':main()
