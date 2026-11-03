"""Search four-op head absorption with arbitrary live products and output key.

Keep the complete DAG operand selectors after one fixed first-MAD family.
This extends the earlier eight affine graph families; UNKNOWN stays open.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
import argparse
from itertools import combinations_with_replacement,permutations
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from codex.search_cross_encoding863 import C1


def orders():
    return sorted({p for source in(('mad','mad','mad','^'),('mad','mad','^','^'))
                   for p in permutations(source)if p[0]=='mad'})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds',type=float,default=6)
    ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--refine',type=Path,help='fix every second-instruction role in saved UNKNOWN cases')
    args=ap.parse_args()
    def run(job):
        order,first,roles=job
        fixed={}if roles is None else{1:roles}
        code=('from codex.search_xor_equivalent_middle863 import search;import json;'
              f'print(json.dumps(search({order!r},11,{args.seconds!r},{int(args.seconds*990)!r},'
              f'first_form={first!r},fixed_input_key={C1!r},fixed_operands={fixed!r})))')
        done=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        if done.returncode:raise RuntimeError(done.stderr)
        return json.loads(done.stdout)
    cases=[(order,first,None)for order in orders()for first in('affine','square_const','square_x')]
    refinement=None
    if args.refine:
        prior=json.loads(args.refine.read_text())
        assert prior['width']==11 and prior['fixed_input_key']==C1
        cases=[]
        for row in prior['results']:
            if row['result']!='unknown':continue
            order,first=tuple(row['sequence']),row['first_form']
            for a,b in combinations_with_replacement(range(4),2):
                for c in(range(4)if order[1]=='mad'else (0,)):
                    cases.append((order,first,(a,b,c)))
        refinement=dict(path=args.refine.name,sha256=hashlib.sha256(args.refine.read_bytes()).hexdigest())
    report=dict(width=11,fixed_input_key=C1,free_output_key=True,new_executable=False,
        refinement=refinement,total_queries=len(cases),
        scope='Six four-op orders starting with MAD; first affine, square+constant or square+input; '
              'all remaining operands and constants unrestricted. No global four-op or kernel claim.',results=[])
    with ThreadPoolExecutor(max_workers=args.workers)as pool:
        pending=[pool.submit(run,case)for case in cases]
        for future in as_completed(pending):
            row=future.result();report['results'].append(row)
            report['counts']=dict(Counter(r['result']for r in report['results']))
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            if not args.refine or len(report['results'])%20==0:
                print(json.dumps(dict(done=len(report['results']),counts=report['counts'])),flush=True)


if __name__=='__main__':main()
