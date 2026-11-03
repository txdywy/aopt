import sys
sys.path.insert(0, '/home/user/aopt')
from itertools import product
from collections import Counter
from codex.cegis_c1_absorb863 import _instr_options, pool_size

OPS_ANY = ('mad', '^', '&', '|', '>>')
OPS_T = ('mad', '^', '&', '|')


def shr_options(i, variable):
    """SHR(a, b): a is a non-constant pool value; b is k0 (constant shift) or a runtime value."""
    ka = i + 1
    vals = list(range(0, i + 1))          # x, t0..t(i-1)
    out = []
    for a in vals:
        if variable:
            for b in vals:
                out.append((a, b))
        else:
            out.append((a, ka))
    return out


def wirings(order, variable):
    opts = [shr_options(i, variable) if op == '>>' else _instr_options(i, op)
            for i, op in enumerate(order)]
    n = len(order)
    for combo in product(*opts):
        ok = True
        for j in range(n - 1):
            ref = j + 1
            if not any(ref in combo[i] and ref < pool_size(i) - 2 for i in range(j + 1, n)):
                ok = False
                break
        if ok:
            yield combo


def orders_with_shr():
    return [o for o in product(OPS_ANY, OPS_ANY, OPS_ANY, OPS_T) if '>>' in o[:3]]


if __name__ == '__main__':
    orders = orders_with_shr()
    print('orders with >=1 SHR in I1..I3, I4 a T-op:', len(orders))
    for variable in (False, True):
        tot = 0
        byn = Counter()
        for o in orders:
            c = sum(1 for _ in wirings(o, variable))
            tot += c
            byn[o.count('>>')] += c
        print(('variable' if variable else 'constant') + '-shift wirings:', tot, dict(byn))
