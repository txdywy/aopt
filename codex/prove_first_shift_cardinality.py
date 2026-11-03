"""Exact finite witness against H(x)=x+K*(y(x)>>s), s>=20.

y may be ANY uint32 function. This is stronger than restricting it to affine.
"""
import json
import random


def first(x):
    a = (4097*x+0x7ed55d16)&0xffffffff
    return a^(a>>19)^0xc761c23c


if __name__ == '__main__':
    rng = random.Random(900)
    inputs = [rng.getrandbits(32) for _ in range(8192)]
    differences = {(first(x)-x)&0xffffffff for x in inputs}
    assert len(differences) > 4096
    # For any K, scaling at most4096 distinct shifted words cannot yield more.
    print(json.dumps(dict(inputs=len(inputs),distinct_differences=len(differences),
                         maximum_candidate_differences_at_shift20=4096,
                         excluded_literal_shifts=list(range(20,33)),
                         scope='x+K*(arbitrary_uint32_function(x)>>s), modulo2^32'),indent=2))
