"""Fix the interior MAD for two nontrivial MAD/XOR/MAD/{XOR,MAD} families."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations_with_replacement
import json
from pathlib import Path
import subprocess
import sys

from codex.search_cross_encoding863 import C1


def roles():
    # Both independent constants remain, including aliases. Final XOR is free.
    return [(a, b, c) for a, b in combinations_with_replacement(range(5), 2) for c in range(5)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=3)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--last-op', choices=('^', 'mad'), default='^')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    jobs = [(source, role) for source in ('input', 'key') for role in roles()]
    order = ('mad', '^', 'mad', args.last_op)
    report = dict(width=11, fixed_input_key=C1, free_output_key=True, sequence=order,
        total_queries=len(jobs), results=[], new_executable=False,
        scope='Affine first MAD; second is t XOR x or t XOR K; every third-MAD '
              'operand role; final instruction arbitrary. UNKNOWN remains open.')

    def run(job):
        source, role = job
        fixed = {0: (0, 1, 2), 1: (0, 1, 0) if source == 'input' else (1, 2, 0), 2: role}
        code = ('from codex.search_xor_equivalent_middle863 import search;import json;'
            f'print(json.dumps(search({order!r},11,{args.seconds!r},'
            f'{max(1,int(args.seconds*990))!r},first_form="affine",'
            f'fixed_input_key={C1!r},fixed_operands={fixed!r})))')
        done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(done.stderr)
        raw = json.loads(done.stdout)
        return dict(source=source, third_operands=role, result=raw['result'], raw=raw)

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
