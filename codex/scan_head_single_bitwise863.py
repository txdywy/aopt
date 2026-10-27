"""Explore four-op head absorption with one AND/OR and three arbitrary MADs."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

from codex.search_cross_encoding863 import C1


def cases():
    rows = []
    for op in ('&','|'):
        for position in range(4):
            order = tuple(op if i == position else 'mad' for i in range(4))
            rows.extend((order, first) for first in (
                (None,) if position == 0 else ('affine','square_const','square_x')))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=6)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    report = dict(width=11,fixed_input_key=C1,free_output_key=True,
        total_queries=len(cases()),results=[],new_executable=False,
        scope='Four operations: three MAD plus one AND/OR in any position. '
              'First MAD partitions affine, square+constant, square+input; '
              'remaining operands unrestricted. UNKNOWN remains open.')

    def run(job):
        order,first = job
        code = ('from codex.search_xor_equivalent_middle863 import search;import json;'
            f'print(json.dumps(search({order!r},11,{args.seconds!r},{int(args.seconds*990)!r},'
            f'first_form={first!r},fixed_input_key={C1!r})))')
        done = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        if done.returncode:
            raise RuntimeError(done.stderr)
        return json.loads(done.stdout)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run,job) for job in cases()]):
            report['results'].append(future.result())
            report['counts'] = dict(Counter(row['result'] for row in report['results']))
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(dict(done=len(report['results']),counts=report['counts'])),flush=True)


if __name__ == '__main__':
    main()
