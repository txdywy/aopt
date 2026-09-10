"""Exclude +,logical-right-shift,MAD from x alone, even variable shifts.

First addition yields x+K,2*x,or constant. For constant shifts1..6, sample
x=n*2**shift (n=0..63): shifted inputs are affine in n except at most one
uint32 wrap. For shifts7..31 sample x=0..63: shifted inputs are constant
except at most one boundary. Shifts0 and>=32 are affine or zero outright.
The final MAD must be quadratic on each of at most two intervals.

For shift amount x or2*x, sample x=64..255: shift output is zero. For
amount x+K it is zero outside at most one32-input interval, including wrap.
Outside that interval the last MAD again must be quadratic. Constant shift
amounts were already covered. All operands may reuse x or earlier results.

The same consecutive-input argument covers first A*x+B at constant shift s
>=6 whenever63*abs(A)<2**s (signed representatives,modulo2**32). The shifted
affine traverses less than one shift-bin width, even across wrap; shifted x
is constant. All remaining operands are affine and the last MAD quadratic.
"""
import json

MASK = (1 << 32) - 1


def first(x):
    a = (4097*x + 0x7ED55D16) & MASK
    return a ^ (a >> 19) ^ 0xC761C23C


def quadratic(values):
    return all((d-3*c+3*b-a) & MASK == 0
               for a,b,c,d in zip(values,values[1:],values[2:],values[3:]))


def possible_cuts(values):
    return [cut for cut in range(len(values)+1)
            if quadratic(values[:cut]) and quadratic(values[cut:])]


def possible_windows(values):
    # Allow arbitrary values inside the exceptional window: a relaxation.
    return [start for start in range(-32, len(values)+1)
            if quadratic(values[:max(0, start)])
            and quadratic(values[min(len(values), max(0, start+32)):])]


if __name__ == '__main__':
    for shift in range(33):
        step = 1 << shift if 1 <= shift <= 6 else 1
        assert not possible_cuts([first(n*step) for n in range(64)])
    assert not possible_windows([first(x) for x in range(64, 256)])
    # Positive controls include nonunit coefficients and modular wrap.
    for cut in range(65):
        control = [((7*x*x-19*x+0xFFFFFFFB) if x < cut
                    else (12*x*x+37*x+11)) & MASK for x in range(64)]
        assert cut in possible_cuts(control)
    for start in range(-32, 193):
        control = [(x*x+17*x+3) & MASK if not start <= x < start+32
                   else (0xABCDEF01 ^ x) for x in range(192)]
        assert start in possible_windows(control)
    print(json.dumps({'constant_shift_cases': 33,
                      'variable_shift_windows': 225,
                      'surviving_cases': 0,
                      'positive_controls': 290,
                      'bounded_affine_multipliers': {
                          str(s): ((1 << s)-1)//63 for s in range(6, 32)},
                      'excluded': '+,>>,mad',
                      'scope': 'one input, arbitrary uint32 constants and operands'}))
