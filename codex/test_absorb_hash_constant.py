"""Independent exhaustive low-four-bit check, without SMT or symmetry rules."""
import unittest


def lifted_candidates(mode, width=11):
    xs = range(1 << width)
    ys = [((33*(x ^ 0xC761C23C)+0xE9F8CC1D) ^
           (16896*(x ^ 0xC761C23C)+0xACCF6200)) for x in xs]
    states = [(0, 0, 0)]
    for bit in range(width):
        flag = 1 << bit
        next_states = []
        for a, b, d in states:
            for choice in range(8):
                aa = a | (flag if choice & 1 else 0)
                bb = b | (flag if choice & 2 else 0)
                dd = d | (flag if choice & 4 else 0)
                if mode == 'nested':
                    f = ys[0] - dd*bb
                    valid = all(not ((dd*((aa*x+bb)^x)+f)^y) & flag for x,y in zip(xs,ys))
                else:
                    f = ys[0] ^ (aa*bb+dd)
                    valid = all(not (((aa*(x^bb)+dd)^f)^y) & flag for x,y in zip(xs,ys))
                if valid: next_states.append((aa,bb,dd))
        states = next_states
        if not states: return bit+1, states
    return width, states


def candidates(absorb):
    mask = 15
    ys = []
    for x in range(16):
        b = x ^ 0xC761C23C if absorb else x
        ys.append(((33*b + 0xE9F8CC1D) ^ (16896*b + 0xACCF6200)) & mask)
    found = []
    # x=0 forces F=B XOR target(0), covering every four-constant tuple
    # that could satisfy the target. No symmetry reduction is used here.
    for a in range(16):
        for b in range(16):
            f = b ^ ys[0]
            for d in range(16):
                if all((((a*x+b) ^ (d*x+f)) & mask) == y for x,y in enumerate(ys)):
                    found.append((a,b,d,f))
    return found


class AbsorbConstantTests(unittest.TestCase):
    def test_nested_and_input_xor_templates(self):
        for mode in ('nested', 'input-xor'):
            with self.subTest(mode=mode):
                self.assertEqual(lifted_candidates(mode), (11, []))

    def test_c1_absorption_impossible_in_two_affine_xor_template(self):
        self.assertEqual(candidates(True), [])

    def test_original_middle_positive_control(self):
        self.assertIn((33 & 15, 0xE9F8CC1D & 15, 16896 & 15, 0xACCF6200 & 15),
                      candidates(False))


if __name__ == '__main__':
    unittest.main()
