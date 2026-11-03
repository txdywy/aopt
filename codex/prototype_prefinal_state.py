"""Algorithm-only pre-final-xorshift state; not a submitted ISA kernel."""
import random
import json
from tests.frozen_problem import Tree, Input, reference_kernel

MASK = (1<<32)-1
C5 = 0xb55a4f09


def t(x):
    return x^(x>>16)


def prefinal(x):
    a = (4097*x+0x7ed55d16)&MASK
    b = a^(a>>19)^0xc761c23c
    c = ((33*b+0xe9f8cc1d)&MASK)^((16896*b+0xaccf6200)&MASK)
    return (9*c+0xfd7046c5)&MASK


def run(seed):
    rng = random.Random(seed)
    tree = Tree(10,[rng.getrandbits(32) for _ in range(2047)])
    values = [rng.getrandbits(32) for _ in range(256)]
    inp = Input([0]*256,values.copy(),16)
    reference_kernel(tree,inp)
    nodes = [t(n^C5) for n in tree.values]
    state = [t(v^C5) for v in values]
    indices = [0]*256
    for rnd in range(16):
        for lane in range(256):
            x = t(state[lane]^nodes[indices[lane]])
            state[lane] = prefinal(x)
            # This is the parity cost that moving T does NOT remove.
            parity = (state[lane]^(state[lane]>>16)^C5)&1
            idx = 2*indices[lane]+1+parity
            indices[lane] = idx if idx < 2047 else 0
    assert [t(s)^C5 for s in state] == inp.values
    assert indices == inp.indices


if __name__ == '__main__':
    for seed in range(40):
        run(seed)
    print(json.dumps(dict(verified_full32_inputs=40,all_values_and_indices=True,
                         isa_implemented=False,
                         finding='Moving final T across XOR preserves values but needs separate parity work.'),indent=2))
