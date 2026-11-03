"""Explore two MADs plus two XOR/AND/OR operations, excluding all-XOR orders."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from itertools import permutations,product
import json
from pathlib import Path
import subprocess
import sys

from codex.search_cross_encoding863 import C1


def cases():
    orders={order for left,right in product(('^','&','|'),repeat=2)
            if (left,right)!=('^','^')
            for order in permutations(('mad','mad',left,right))}
    return [(order,first) for order in sorted(orders) for first in
            (('affine','square_const','square_x') if order[0]=='mad' else (None,))]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=5)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    jobs=cases()
    report=dict(width=11,fixed_input_key=C1,free_output_key=True,total_queries=len(jobs),
        results=[],new_executable=False,
        scope='Exactly two MAD and two bitwise XOR/AND/OR with at least one AND/OR; '
              'unrestricted operands; first MAD partitions when present. '
              'UNKNOWN is preserved; any reduced SAT requires full32 verification.')
    def run(job):
        order,first=job
        code=('from codex.search_xor_equivalent_middle863 import search;import json;'
            f'print(json.dumps(search({order!r},11,{args.seconds!r},{int(args.seconds*990)!r},'
            f'first_form={first!r},fixed_input_key={C1!r})))')
        done=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        if done.returncode:raise RuntimeError(done.stderr)
        return json.loads(done.stdout)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run,job) for job in jobs]):
            report['results'].append(future.result())
            report['counts']=dict(Counter(row['result'] for row in report['results']))
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            if len(report['results'])%12==0 or len(report['results'])==len(jobs):
                print(json.dumps(dict(done=len(report['results']),counts=report['counts'])),flush=True)


if __name__=='__main__':main()
