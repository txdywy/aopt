"""Check interval-tiling lemma for a+K*(a>>s), modulo2**w.

Each input block of length L=2**s maps to a translated interval of length L.
If these intervals partition the circle, all starts lie in the same residue
class modulo L. Block zero starts at zero, block one at L+K, hence L divides K.
This is a necessary condition, not sufficient (the quotient stride must be odd).
"""
import json


def h(x):
    a=(4097*x+0x7ed55d16)&0xffffffff
    return a^(a>>19)^0xc761c23c


if __name__ == '__main__':
    tested=0
    permutations=0
    for width in range(2,9):
        modulus=1<<width
        for shift in range(1,width):
            length=1<<shift
            for k in range(modulus):
                outputs={(a+k*(a>>shift))%modulus for a in range(modulus)}
                actual=len(outputs)==modulus
                expected=k%length==0 and ((1+k//length)&1)==1
                assert actual==expected,(width,shift,k)
                tested+=1
                permutations+=actual
    # An affine map modulo2**s respects congruences modulo2**s.
    for shift in range(1,20):
        assert ((h(0)^h(1<<19))&((1<<shift)-1)) != 0
    # Affine maps have vanishing second difference modulo2**s.
    for shift in range(20,32):
        assert (h(2)-2*h(1)+h(0))% (1<<shift) != 0
    print(json.dumps(dict(small_word_cases=tested,permutations=permutations,
                         target_nonaffine_low_bit_witnesses=31,
                         excluded='H(x)=a+K*(a>>s), a=A*x+B, 1<=s<32'),indent=2))
