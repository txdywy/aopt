import random
import unittest

import z3

from codex.cegis_c1_absorb863 import _instr_options
from codex.search_keyed_live_middle863 import live_values, reference
from codex.search_keyed_live_restore863 import control, evaluate, manifest, parallel_manifest, solve, validate
from codex.test_keyed_live_middle863 import frozen


def legacy_rows(row, constants):
    return [dict(op=op,operands=w+[0]*(3-len(w)),constants=cc)
            for op,w,cc in zip(row['order'],row['wiring'],constants)]


class KeyedLiveRestoreTests(unittest.TestCase):
    def test_full32_random_programs_match_frozen_instructions(self):
        rng = random.Random(8642948)
        for _ in range(80):
            order = [rng.choice(('mad','^','&','|')) for _ in range(4)]
            row = dict(order=order,wiring=[list(rng.choice(_instr_options(2+i,op))) for i,op in enumerate(order)])
            cc = [[rng.getrandbits(32),rng.getrandbits(32)] for _ in order]
            key = rng.getrandbits(32)
            for a in (0,0xffffffff,0x80001,rng.getrandbits(32)):
                expected = frozen(a,key,legacy_rows(row,cc))
                self.assertEqual(expected,evaluate(row,cc,a,key,32))
                kk = [[z3.BitVecVal(v,32) for v in pair] for pair in cc]
                self.assertEqual(expected,z3.simplify(evaluate(row,kk,z3.BitVecVal(a,32),key,32)).as_long())

    def test_low_domain_preserves_the_real_input_relations(self):
        rng = random.Random(8642949)
        row = dict(order=['mad','^','mad','^'],wiring=[[0,1,3],[2,3],[1,4,5],[5,6]])
        for width in (5,11,16):
            mask = (1 << width)-1
            for _ in range(40):
                a,key = rng.randrange(1 << width),rng.getrandbits(32)
                cc = [[rng.getrandbits(32),rng.getrandbits(32)] for _ in row['order']]
                self.assertEqual([v&mask for v in live_values(a,key,32)],live_values(a,key&mask,width))
                self.assertEqual(evaluate(row,cc,a,key,32)&mask,evaluate(row,cc,a,key&mask,width))
                self.assertEqual(reference(a,32)&mask,reference(a,width)&mask)
                before,after = live_values(a,key,32),live_values(a,key^0x13579bdf,32)
                self.assertEqual(before[:2],after[:2])
                self.assertEqual(before[2]^after[2],0x13579bdf)

    def test_known_four_op_identity_is_checked_over_full32(self):
        result = control()
        self.assertEqual('proved_full32',result['status'],result)
        self.assertFalse(result['new_executable'])

    def test_automatic_control_with_prior_products_and_fixed_output_key(self):
        row = dict(order=['mad','^','mad','^'],wiring=[[0,1,3],[2,3],[1,4,5],[5,6]])
        witness = dict(row=row,constants=[[5,7],[0,0],[9,13],[17,0]],key=42)
        result = solve(row,5,10,control=witness)
        self.assertEqual('reduced_sat',result['status'],result)
        for a in range(32):
            expected = evaluate(row,witness['constants'],a,42,5) ^ (result['output_key'] & 31)
            self.assertEqual(expected,evaluate(row,result['constants'],a,result['input_key'],5))
        self.assertFalse(result['new_executable'])

    def test_manifest_is_deterministic_live_and_prior_aware(self):
        rows,stats = manifest()
        self.assertEqual((rows,stats),manifest())
        self.assertEqual(504,len(rows))
        self.assertEqual(63,len(stats))
        for row in rows:
            validate(row)
            self.assertTrue(any(1 in w for w in row['wiring']))
            for j in range(3):
                self.assertTrue(any(3+j in row['wiring'][i] for i in range(j+1,4)))
        parallel,orders = parallel_manifest()
        self.assertEqual(1395,len(parallel))
        self.assertEqual(3,len(orders))
        self.assertEqual(1395,len({(tuple(r['order']),tuple(map(tuple,r['wiring']))) for r in parallel}))
        for row in parallel:
            validate(row)
            self.assertNotIn(3,row['wiring'][1])
            self.assertEqual([[3,4],[5,6,7]],row['wiring'][2:])

    def test_rejects_noncommuting_or_invalid_grammar(self):
        for row in (dict(order=['shr16'],wiring=[[0]]),dict(order=['mad'],wiring=[[0,1,5]]),
                    dict(order=['mad'],wiring=[[0,1]])):
            with self.assertRaises(ValueError):
                validate(row)
        with self.assertRaises(ValueError):
            solve(dict(order=['mad'],wiring=[[0,3,4]]),19)


if __name__ == '__main__':
    unittest.main()
