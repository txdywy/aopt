"""Test the 28 remaining first-MAD shapes with keyed earlier live inputs.

This is a concrete MAD/bitwise/MAD graph search. All Z3 calls execute in
separate subprocesses; the controller threads never share a Z3 context.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

from codex.scan_keyed_live_frontier863 import partitions


def first_forms():
    for source,tail in ((0,2),(1,2),(2,0),(2,1)):
        yield 'mixed',(source,3,tail)
    for a,b in itertools.combinations_with_replacement(range(3),2):
        for c in range(4):
            yield 'product',(a,b,c)


def canonical_first(role):
    """Classify any first MAD: an old affine family or one of the 28 forms.

    prior is an affine bijection of a modulo 2**w. Therefore an arbitrary
    scaled a plus prior (and the reverse) is contained in the old affine
    family. Constant aliases may be relaxed to independent constants.
    """
    a,b,c=role
    a,b=sorted((a,b))
    if b<3:
        return 'product',(a,b,min(c,3))
    if a<3 and c<3 and a!=c and {a,c}!={0,1}:
        return 'mixed',(a,3,c)
    return 'affine',None


def control_program(first,bitwise='^',source='key'):
    other={'a':0,'prior':1,'keyed':2,'key':4}[source]
    return [dict(op='mad',operands=list(first),constants=[5,7]),
            dict(op=bitwise,operands=[*sorted((3,other)),0],constants=[85,0]),
            dict(op='mad',operands=[4,5,6],constants=[9,13])]


def run_job(job,seconds,width,last_operands=None):
    kind,first,bitwise,source=job
    options=dict(first_operands=first,middle_source=source)
    if last_operands is not None:options['last_operands']=last_operands
    source_code=('from codex.search_keyed_live_middle863 import search;import json;'
        f'print(json.dumps(search({("mad",bitwise,"mad")!r},{width!r},{seconds!r},'
        f'{max(1,int(seconds*990))!r},**{options!r})))')
    done=subprocess.run([sys.executable,'-c',source_code],capture_output=True,text=True)
    if done.returncode:raise RuntimeError(done.stderr)
    raw=json.loads(done.stdout)
    return dict(kind=kind,first_operands=list(first),bitwise=bitwise,middle_source=source,
                last_operands=last_operands,result=raw['result'],raw=raw)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds',type=float,default=4)
    ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--width',type=int,default=11)
    ap.add_argument('--refine',type=Path,help='partition final MAD roles for the saved UNKNOWN cases')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    jobs=[(kind,first,op,source)for kind,first in first_forms()
          for op in ('^','&','|')for source in ('a','prior','keyed','key')]
    refinement=None
    if args.refine:
        source=json.loads(args.refine.read_text())
        if source['width']!=args.width:raise ValueError('refinement width must match the source')
        jobs=[(row['kind'],tuple(row['first_operands']),row['bitwise'],row['middle_source'])
              for row in source['results']if row['result']=='unknown']
        refinement=dict(path=args.refine.name,sha256=hashlib.sha256(args.refine.read_bytes()).hexdigest(),
                        coarse_cases=len(jobs))
    work=[(job,role)for job in jobs for role in
          ([role for _,role in partitions(('a',))]if args.refine else [None])]
    report=dict(width=args.width,seconds_limit=args.seconds,total_jobs=len(work),refinement=refinement,
        scope='28 first-MAD shapes, middle bitwise of that result and one initial live word '
              'or arbitrary constant, arbitrary final MAD operands. UNKNOWN remains open. '
              'Reduced SAT is not full32 equivalence or a kernel score.',
        results=[],new_executable=False)
    def save():
        report['counts']=dict(Counter(r['result']for r in report['results']))
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    with ThreadPoolExecutor(max_workers=args.workers)as pool:
        pending=[pool.submit(run_job,job,args.seconds,args.width,role)for job,role in work]
        for future in as_completed(pending):
            report['results'].append(future.result());save()
            if len(report['results'])%24==0:
                print(json.dumps(dict(done=len(report['results']),counts=report['counts'])),flush=True)
    report['results'].sort(key=lambda row:(row['kind'],row['first_operands'],row['bitwise'],
                                         row['middle_source'],row['last_operands']or []))
    save()


if __name__=='__main__':main()
