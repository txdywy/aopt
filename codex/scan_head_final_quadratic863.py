"""Split the last twelve MMXM templates by the second MAD's quadratic term."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from itertools import combinations_with_replacement
import json
from pathlib import Path
import subprocess
import sys

from codex.scan_head_final_product863 import XOR_ROLES
from codex.search_cross_encoding863 import C1

ROOT = Path(__file__).resolve().parent
AFFINE = (0, 2, 3)


def quadratic_roles():
    return [(*pair,c) for pair in combinations_with_replacement(range(2),2) for c in range(3)]


def canonical_second(role):
    a,b,c = role
    if b >= 2:
        return 'affine', AFFINE
    return 'quadratic', (a,b,min(c,2))


def cases(prior):
    rows = []
    for item in prior['remaining_final_templates']:
        source,last = item['source'],tuple(item['last_operands'])
        # If p is affine and z is p XOR x/K, the first affine result t
        # becomes a polynomial in x after the bitwise prefix. The complete
        # degree-four module proofs cover any subsequent use of t or p.
        seconds = quadratic_roles()+([AFFINE] if source == 'first' else [])
        rows.extend((source,second,last) for second in seconds)
    return rows


def fixed_roles(source,second,last):
    return {0:(0,1,2),1:second,2:XOR_ROLES[source],3:last}


def first_is_live(source,second,last):
    return 1 in second or 1 in XOR_ROLES[source][:2] or 1 in last


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=4)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    path = ROOT/'head_affine_frontier863_audit.json'
    prior = json.loads(path.read_text())
    jobs = cases(prior)
    report = dict(width=11,fixed_input_key=C1,free_output_key=True,
        sequence=['mad','mad','^','mad'],total_queries=len(jobs),results=[],
        prior=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
        affine_input_and_key_covered_by_prior_modules=True,new_executable=False,
        scope='Twelve remaining affine-first MMXM final templates; nine canonical '
              'quadratic second-MAD roles, and independent affine second MAD for '
              'XOR with first MAD. UNKNOWN remains open.')

    def run(job):
        source,second,last = job
        fixed = fixed_roles(*job)
        code = ('from codex.search_xor_equivalent_middle863 import search;import json;'
            f'print(json.dumps(search(("mad","mad","^","mad"),11,{args.seconds!r},'
            f'{max(1,int(args.seconds*990))!r},first_form="affine",'
            f'fixed_input_key={C1!r},fixed_operands={fixed!r})))')
        done = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        if done.returncode:
            raise RuntimeError(done.stderr)
        raw = json.loads(done.stdout)
        return dict(source=source,second_operands=second,last_operands=last,
                    result=raw['result'],raw=raw)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run,job) for job in jobs]):
            report['results'].append(future.result())
            report['counts'] = dict(Counter(row['result'] for row in report['results']))
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            if len(report['results']) % 20 == 0 or len(report['results']) == len(jobs):
                print(json.dumps(dict(done=len(report['results']),counts=report['counts'])),flush=True)


if __name__ == '__main__':
    main()
