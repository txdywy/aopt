"""Bounded structural split for first affine MAD / SHR20 / final MAD.

Prints evidence only; UNKNOWN remains open. Does not emit a kernel.
"""
import itertools
import json
import subprocess
import sys


if __name__ == '__main__':
    def target(x):
        a = (4097*x+0x7ed55d16)&0xffffffff
        return a^(a>>19)^0xc761c23c
    # Exact witness invalidating any degree<=2 polynomial over Z/(2**32).
    assert (target(3)-3*target(2)+3*target(1)-target(0))&0xffffffff == 0xffff7fe8
    counts = {'no_shift_polynomial':0,'shift_only_cardinality':0,'unsat':0,'inconclusive':0}
    for a,b,c in itertools.product(range(5),repeat=3):
        if a > b:
            continue
        shape = (a,b,c)
        if 2 not in shape:
            # x and first affine MAD are affine in x; last MAD is quadratic.
            counts['no_shift_polynomial'] += 1
            continue
        if not any(index in shape for index in (0,1)):
            counts['shift_only_cardinality'] += 1
            continue
        command = [sys.executable,'-m','codex.synthesize_hash_cegis','--slice','first',
                   '--instructions','3','--op-sequence','mad,>>,mad',
                   '--first-mad-form','affine','--right-shift-amount','20',
                   '--canonical-operands','--first-boundaries','--high-bit-pairs',
                   '--extra-samples','64','--last-mad-operands',','.join(map(str,shape)),
                   '--seconds','3','--check-ms','2000']
        result = subprocess.run(command,text=True,capture_output=True,check=True)
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        status = rows[-1]['result']
        key = 'unsat' if status == 'grammar_unsatisfiable' else status
        counts[key] = counts.get(key,0)+1
        print(json.dumps(dict(shape=shape,status=status)),flush=True)
    print(json.dumps(dict(summary=counts,total_shapes=sum(counts.values()))),flush=True)
