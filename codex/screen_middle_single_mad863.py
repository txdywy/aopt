"""Bounded three-instruction middle-hash orders with one MAD and two bitwise ops.

Timeouts are recorded as inconclusive. Low-width equivalences are not uint32
equivalences. This runner never changes the submission kernel.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
from pathlib import Path
import subprocess
import sys


def orders():
    return sorted({tuple(list(bits[:position])+['mad']+list(bits[position:]))
                   for bits in itertools.product(('^', '&', '|'), repeat=2)
                   for position in range(3)})


def check(order, timeout_ms):
    command = [sys.executable, '-m', 'codex.synthesize_hash_cegis', '--slice', 'middle',
        '--instructions', '3', '--width', '12', '--op-sequence='+','.join(order),
        '--canonical-operands', '--consecutive-samples', '32', '--extra-samples', '32',
        '--seconds', str(max(2, timeout_ms//1000+2)), '--check-ms', str(timeout_ms)]
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=timeout_ms/1000+15)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return {'order': order, 'command': command,
            'output': [json.loads(line) for line in result.stdout.splitlines()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout-ms', type=int, default=3000)
    args = parser.parse_args()
    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for row in pool.map(lambda order: check(order, args.timeout_ms), orders()):
            rows.append(row)
            args.output.write_text(json.dumps(rows, indent=2)+'\n')
            print(json.dumps({'order': row['order'], 'result': row['output'][-1]['result']}), flush=True)


if __name__ == '__main__':
    main()
