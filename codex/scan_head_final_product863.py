"""Fix the XOR and final MAD in the remaining MAD/MAD/XOR/MAD order."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations_with_replacement
import json
from pathlib import Path
import subprocess
import sys

from codex.search_cross_encoding863 import C1


XOR_ROLES = {'input': (0, 2, 0), 'first': (1, 2, 0), 'key': (2, 3, 0)}


def last_roles():
    return [(a, b, c) for a, b in combinations_with_replacement(range(6), 2)
            for c in range(6) if 3 in (a, b, c)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=3)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    jobs = [(source, role) for source in XOR_ROLES for role in last_roles()]
    report = dict(width=11, fixed_input_key=C1, free_output_key=True,
        sequence=['mad', 'mad', '^', 'mad'], total_queries=len(jobs), results=[],
        new_executable=False, scope='Affine first MAD; second MAD unrestricted; XOR reads '
            'second result and x, first result or arbitrary constant; every final MAD '
            'operand role reading XOR result. UNKNOWN remains unresolved.')

    def run(job):
        source, role = job
        fixed = {0: (0, 1, 2), 2: XOR_ROLES[source], 3: role}
        code = ('from codex.search_xor_equivalent_middle863 import search;import json;'
            f'print(json.dumps(search(("mad","mad","^","mad"),11,{args.seconds!r},'
            f'{max(1,int(args.seconds*990))!r},first_form="affine",'
            f'fixed_input_key={C1!r},fixed_operands={fixed!r})))')
        done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(done.stderr)
        raw = json.loads(done.stdout)
        return dict(source=source, last_operands=role, result=raw['result'], raw=raw)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(run, job) for job in jobs]
        for future in as_completed(pending):
            report['results'].append(future.result())
            report['counts'] = dict(Counter(row['result'] for row in report['results']))
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            if len(report['results']) % 20 == 0 or len(report['results']) == len(jobs):
                print(json.dumps(dict(done=len(report['results']), counts=report['counts'])), flush=True)


if __name__ == '__main__':
    main()
