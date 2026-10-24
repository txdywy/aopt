"""Exact small-domain oracles for the full uint32 divisor interval screen."""
import itertools
import json
from pathlib import Path
import random
import unittest

from codex.audit_full_live_tail863 import coverage
from codex.screen_live_division_tail863 import affine_intervals, prove
from codex.search_full_live_shift863 import live_words, zero_witnesses


def oracle(numerators, outputs, bits, limit, ceil):
    modulus = 1 << bits
    good = set()
    for divisor in range(1, limit+1):
        qs = [(n+divisor-1)//divisor if ceil else n//divisor for n in numerators]
        for scale in range(modulus):
            offset = (outputs[0]-scale*qs[0]) % modulus
            if all((scale*q+offset-y) % modulus == 0 for q, y in zip(qs, outputs)):
                good.add(divisor)
                break
    return good


class DivisorIntervalTests(unittest.TestCase):
    def test_intervals_match_exhaustive_divisors_and_coefficients(self):
        rng = random.Random(878)
        for ceil, bits, limit in itertools.product((False, True), (2, 3, 5), (1, 7, 63)):
            for _ in range(8):
                nums = [0, 255]+[rng.randrange(256) for _ in range(3)]
                ys = [rng.randrange(1 << bits) for _ in nums]
                ys[1] = ys[0] ^ 1  # An odd difference, as in the actual certificate.
                result = affine_intervals(nums, ys, ceil=ceil, bits=bits, divisor_limit=limit)
                actual = {d for lo, hi in result['survivor_intervals'] for d in range(lo, hi+1)}
                self.assertEqual(oracle(nums, ys, bits, limit, ceil), actual,
                                 (nums, ys, bits, limit, ceil))
                self.assertEqual(limit, result['covered_divisors'])

    def test_real_affine_division_controls_are_kept(self):
        for ceil, divisor, scale in itertools.product((False, True), (1, 3, 7, 16, 63), (1, 5, 17)):
            nums = [0, 1, divisor, divisor+1, 255]
            qs = [(n+divisor-1)//divisor if ceil else n//divisor for n in nums]
            ys = [(scale*q+23)&31 for q in qs]
            result = affine_intervals(nums, ys, ceil=ceil, bits=5, divisor_limit=255)
            self.assertTrue(any(lo <= divisor <= hi for lo, hi in result['survivor_intervals']))
        result = affine_intervals([0, 0, 0], [23, 23, 23])
        self.assertEqual([[1, 0xffffffff]], result['survivor_intervals'])

    def test_final_mad_role_partition_by_complete_small_word_enumeration(self):
        # Eight existing values and two constants are fixed within each
        # congruence class; only the first instruction result q varies.
        constants = [3, 7, 8, 14, 1, 6, 12, 15, None, 11, 4]
        for a, b, c in itertools.product(range(11), repeat=3):
            ys = []
            for q in range(16):
                pool = [q if value is None else value for value in constants]
                ys.append((pool[a]*pool[b]+pool[c])&15)
            if a == b == 8:
                if c == 8:
                    self.assertTrue(all(y%2 == 0 for y in ys))
                else:
                    self.assertLessEqual(len({y%4 for y in ys}), 2)
            else:
                coefficient = (ys[1]-ys[0])&15
                self.assertTrue(all(y == (coefficient*q+ys[0])&15 for q, y in enumerate(ys)))

    def test_remainder_reduction_and_divisor_zero_witnesses(self):
        for low, divisor, scale, offset in itertools.product(range(16), range(1, 64), (1, 3, 4, 15), (0, 7)):
            nums = [low+16*j for j in range(8)]
            actual = [(scale*(n%divisor)+offset)&15 for n in nums]
            relaxed = [((-scale*divisor)*(n//divisor)+scale*low+offset)&15 for n in nums]
            self.assertEqual(actual, relaxed)
        for scope, count in (('middle', 5), ('tail', 8)):
            rows = zero_witnesses(scope)
            self.assertEqual(count, len(rows))
            for row in rows:
                self.assertEqual(0, live_words(row['prior'], scope)[row['source']])

    def test_full_range_enumeration_completes_only_the_stated_grammar(self):
        report = prove()
        self.assertTrue(report['excluded'])
        self.assertEqual(16, len(report['rows']))
        self.assertTrue(all(row['covered_divisors'] == 0xffffffff for row in report['rows']))
        self.assertTrue(all(row['positive_control']['retained'] for row in report['rows']))
        folder = Path(__file__).parent
        previous = json.loads((folder/'full_live_tail_closure863_results.json').read_text())
        names = ('full_live_tail_shift863_results.json', 'full_live_tail_shift863_refinement.json',
                 'full_live_tail_division863_results.json', 'full_live_tail_division863_refinement.json')
        searches = [json.loads((folder/name).read_text()) for name in names]
        completed = coverage(previous, searches, report)
        self.assertEqual((196, 0), (completed['closed'], completed['unresolved']))
        self.assertEqual(864, completed['best_verified_cycles'])
        self.assertFalse(completed['new_executable'])
        self.assertTrue(any(row['result'] == 'unknown' for row in searches[-1]['rows']))

    def test_reject_unsupported_or_invalid_input_domains(self):
        for kwargs in ({'bits': 17}, {'divisor_limit': 0}, {'divisor_limit': 1 << 32}):
            with self.assertRaises(ValueError):
                affine_intervals([0, 1], [0, 1], **kwargs)
        for nums, ys in (([0], [0]), ([0, 1], [0]), ([-1, 2], [0, 1]), ([0, 1], [0, 2])):
            with self.assertRaises(ValueError):
                affine_intervals(nums, ys)


if __name__ == '__main__':
    unittest.main()
