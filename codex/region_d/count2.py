"""Complete SHR operand space: SHR(a, b), a in {runtime values, k0}, b in {runtime values, k0}.
  a runtime, b = k0      -> constant shift (count.py, variable=False)
  a runtime, b runtime   -> variable shift (count.py, variable=True)
  a = k0,    b runtime   -> constant shifted by a runtime amount (new)
  a = k0,    b = k0      -> a constant; never needed
kind tags each SHR 'c' (constant shift), 'v' (variable), 'k' (constant >> runtime)."""
import sys
sys.path.insert(0, '/home/user/aopt')
from itertools import product
from collections import Counter
from codex.cegis_c1_absorb863 import _instr_options, pool_size
from count import orders_with_shr


def shr_all(i):
    ka = i + 1
    vals = list(range(0, i + 1))
    out = [((a, ka), 'c') for a in vals] + [((a, b), 'v') for a in vals for b in vals] + [((ka, b), 'k') for b in vals]
    return out


def wirings_tagged(order):
    opts = [shr_all(i) if op == '>>' else [(x, None) for x in _instr_options(i, op)] for i, op in enumerate(order)]
    n = len(order)
    for combo in product(*opts):
        w = tuple(x[0] for x in combo)
        ok = True
        for j in range(n - 1):
            ref = j + 1
            if not any(ref in w[i] and ref < pool_size(i) - 2 for i in range(j + 1, n)):
                ok = False
                break
        if ok:
            yield w, ''.join(x[1] for x in combo if x[1])


if __name__ == '__main__':
    c = Counter()
    for o in orders_with_shr():
        for w, kinds in wirings_tagged(o):
            c[kinds] += 1
    for k, n in sorted(c.items(), key=lambda kv: (len(kv[0]), kv[0])):
        print(f'{k:4s} {n}')
