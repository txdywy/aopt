"""Four-input certificate excluding every integer-polynomial middle rewrite.

For any integer polynomial P, its mixed difference with strides h,k is
divisible by h*k. This follows on every monomial by two binomial expansions.
Choose h=256,k=8, so it must vanish modulo 2048. The target violates it.
Thus no straight-line expression using only +,-,*,MAD and constants, of
any length or polynomial degree, computes this target from x alone.
Other instructions and additional non-polynomial live inputs are not covered.
"""
import json

from codex.screen_quadratic_xor_affine863 import target


def certificate():
    width, h, k, x = 11, 256, 8, 0
    mask = (1 << width)-1
    inputs = [x, x+k, x+h, x+h+k]
    outputs = [target(value) & mask for value in inputs]
    residue = (outputs[3]-outputs[2]-outputs[1]+outputs[0]) & mask
    assert h*k == mask+1 and residue != 0
    return {'width': width, 'strides': [h, k], 'inputs': inputs,
            'outputs': outputs, 'mixed_difference': residue,
            'excluded': True,
            'scope': 'Every integer polynomial from x alone; no shifts, bitwise ops, '
                     'division, selection, memory or extra non-polynomial inputs'}


if __name__ == '__main__':
    print(json.dumps(certificate()))
