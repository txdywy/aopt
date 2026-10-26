"""Partition the affine MAD/XOR/XOR/MAD frontier into four XOR pairs.

After known three-op and keyed-live reductions, the last MAD must read both
nontrivial XOR outputs u,v. This leaves 13 roles and four pair templates.
Every SMT query runs in its own process; UNKNOWN is retained unchanged.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations_with_replacement
import json
from pathlib import Path
import subprocess
import sys

from codex.search_cross_encoding863 import C1

PAIRS = {
    'xor_then_key': ((0, 1, 0), (2, 3, 0)),       # u=t XOR x, v=u XOR K
    'key_then_xor': ((1, 2, 0), (0, 2, 0)),       # u=t XOR K, v=u XOR x
    'parallel_input_key': ((0, 1, 0), (1, 3, 0)), # u=t XOR x, v=t XOR K
    'parallel_keys': ((1, 2, 0), (1, 3, 0)),      # u=t XOR K, v=t XOR L
}


def last_roles():
    return sorted({(a, min(b, 4), min(c, 4))
        for a, b in combinations_with_replacement(range(6), 2) for c in range(6)
        if {2, 3}.issubset((a, b, c))})


def fixed_roles(family, last):
    second, third = PAIRS[family]
    return {0: (0, 1, 2), 1: second, 2: third, 3: last}


def control_program(family, last):
    fixed = fixed_roles(family, last)
    return [dict(op=op, operands=list(fixed[i]), constants=constants)
        for i, (op, constants) in enumerate(zip(('mad', '^', '^', 'mad'),
            ([5, 7], [13, 19], [21, 23], [9, 29])))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=4)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    jobs = [(family, last) for family in PAIRS for last in last_roles()]
    report = dict(width=11, fixed_input_key=C1, free_output_key=True,
        total_queries=len(jobs), results=[], new_executable=False,
        scope='Affine first MAD; two XOR outputs form one of four explicit pairs; '
              'final MAD reads both outputs in one of 13 canonical roles. '
              'Other connections require separately checked structural reductions.')

    def run(job):
        family, last = job
        fixed = fixed_roles(family, last)
        code = ('from codex.search_xor_equivalent_middle863 import search;import json;'
            f'print(json.dumps(search(("mad","^","^","mad"),11,{args.seconds!r},'
            f'{max(1,int(args.seconds*990))!r},first_form="affine",'
            f'fixed_input_key={C1!r},fixed_operands={fixed!r})))')
        done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(done.stderr)
        raw = json.loads(done.stdout)
        return dict(family=family, last_operands=last, result=raw['result'], raw=raw)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(run, job) for job in jobs]
        for future in as_completed(pending):
            report['results'].append(future.result())
            report['counts'] = dict(Counter(row['result'] for row in report['results']))
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            if len(report['results']) % 8 == 0 or len(report['results']) == len(jobs):
                print(json.dumps(dict(done=len(report['results']), counts=report['counts'])), flush=True)


if __name__ == '__main__':
    main()
