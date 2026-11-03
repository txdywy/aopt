"""Fix the first bitwise operands in unresolved MAD/BIT/BIT/MAD orders."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
from itertools import combinations_with_replacement
import json
from pathlib import Path
import subprocess
import sys

from codex.search_cross_encoding863 import C1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior',type=Path,default=Path('codex/head_two_bitwise863_results.json'))
    parser.add_argument('--seconds',type=float,default=4)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    prior=json.loads(args.prior.read_text())
    orders={tuple(row['sequence']) for row in prior['results'] if row['result']=='unknown'
            and row['first_form']=='affine' and row['sequence'][0]=='mad'
            and row['sequence'][1]!='mad' and row['sequence'][2]!='mad' and row['sequence'][3]=='mad'}
    jobs=[(order,(*pair,0)) for order in sorted(orders)
          for pair in combinations_with_replacement(range(4),2)]
    report=dict(width=11,fixed_input_key=C1,free_output_key=True,total_queries=len(jobs),
        results=[],new_executable=False,
        prior=dict(path=args.prior.name,sha256=hashlib.sha256(args.prior.read_bytes()).hexdigest()),
        scope='Unresolved affine-first MAD/BIT/BIT/MAD; fix every first bitwise operand pair; '
              'remaining operands unrestricted. Preserve UNKNOWN and any reduced SAT.')
    def run(job):
        order,second=job
        code=('from codex.search_xor_equivalent_middle863 import search;import json;'
            f'print(json.dumps(search({order!r},11,{args.seconds!r},{int(args.seconds*990)!r},'
            f'first_form="affine",fixed_input_key={C1!r},fixed_operands={{1:{second!r}}})))')
        done=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        if done.returncode:raise RuntimeError(done.stderr)
        return json.loads(done.stdout)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run,job) for job in jobs]):
            report['results'].append(future.result())
            report['counts']=dict(Counter(row['result'] for row in report['results']))
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            if len(report['results'])%10==0 or len(report['results'])==len(jobs):
                print(json.dumps(dict(done=len(report['results']),counts=report['counts'])),flush=True)


if __name__=='__main__':main()
