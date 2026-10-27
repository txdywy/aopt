"""Refine MAD/MAD/AND-or-OR/MAD by exact bitwise and final operand roles."""
import argparse
import hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import subprocess
import sys

from codex.scan_head_final_product863 import XOR_ROLES as SOURCES,last_roles
from codex.scan_head_final_quadratic863 import quadratic_roles,AFFINE
from codex.search_cross_encoding863 import C1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=3)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--refine',type=Path)
    args=parser.parse_args()
    jobs=[(op,source,last,None) for op in ('&','|') for source in SOURCES for last in last_roles()]
    prior=None
    if args.refine:
        raw=json.loads(args.refine.read_text())
        assert len(raw['results'])==raw['total_queries']==306
        templates={(row['operator'],row['source'],tuple(min(v,4) for v in row['last_operands']))
            for row in raw['results'] if row['result']=='unknown'}
        jobs=[(op,source,last,second) for op,source,last in sorted(templates)
            for second in quadratic_roles()+([AFFINE] if source=='first' else [])]
        prior=dict(path=args.refine.name,sha256=hashlib.sha256(args.refine.read_bytes()).hexdigest(),
            affine_input_and_key_defer_to_module=True)
    report=dict(width=11,fixed_input_key=C1,free_output_key=True,total_queries=len(jobs),
        results=[],new_executable=False,refinement=prior,
        scope='Affine first MAD, unrestricted second MAD, AND/OR of second result '
              'with x/first-result/constant, arbitrary final MAD reading bitwise result. '
              'Queries enforce live intermediates; dead programs need the separate three-op proof.')
    def run(job):
        op,source,last,second=job
        fixed={0:(0,1,2),2:SOURCES[source],3:last}
        if second is not None:fixed[1]=second
        order=('mad','mad',op,'mad')
        code=('from codex.search_xor_equivalent_middle863 import search;import json;'
            f'print(json.dumps(search({order!r},11,{args.seconds!r},{int(args.seconds*990)!r},'
            f'first_form="affine",fixed_input_key={C1!r},fixed_operands={fixed!r})))')
        done=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        if done.returncode:raise RuntimeError(done.stderr)
        raw=json.loads(done.stdout)
        return dict(operator=op,source=source,last_operands=last,second_operands=second,
                    result=raw['result'],raw=raw)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run,job) for job in jobs]):
            report['results'].append(future.result())
            report['counts']=dict(Counter(row['result'] for row in report['results']))
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            if len(report['results'])%30==0 or len(report['results'])==len(jobs):
                print(json.dumps(dict(done=len(report['results']),counts=report['counts'])),flush=True)


if __name__=='__main__':main()
