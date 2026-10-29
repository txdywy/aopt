import random
import unittest

import numpy as np
import z3

from codex.cegis_c1_absorb863 import _instr_options
from codex.refine_joint_quotient_range863 import quotient_maximum, sketch, solve, validate_row
from codex.search_joint_live_boundary863 import inputs, NODE, C0, C4, MASK
from codex.test_live_boundary863 import frozen


class QuotientRangeTests(unittest.TestCase):
    def test_actual_division_outputs_fit_the_scoped_ranges(self):
        rng = random.Random(8642946)
        for op in ('//','%','cdiv'):
            for d in (2,3,5,65535,65536,65537,9004123):
                for n in (0,1,d-1,d,MASK,*(rng.getrandbits(32) for _ in range(20))):
                    q = n//d if op=='//' else n%d if op=='%' else (n+d-1)//d
                    self.assertLessEqual(q, quotient_maximum(op,d))
                    self.assertLessEqual(q, quotient_maximum(op))
        self.assertEqual((MASK+1)//2, quotient_maximum('cdiv'))

    def test_random_frozen_programs_embed_in_the_relaxation(self):
        rng = random.Random(8642947)
        for op in ('//','%','cdiv'):
            for _ in range(24):
                special = rng.choice((1,2))
                order = [rng.choice(('mad','^','&','|')) for _ in range(4)]
                order[special] = op
                wiring = [list((rng.randrange(8+i),)) if i==special else
                          list(rng.choice(_instr_options(7+i,code))) for i,code in enumerate(order)]
                row = dict(order=order,wiring=wiring)
                cc = [[rng.getrandbits(32),rng.getrandbits(32)] for _ in order]
                cc[special][0] = rng.choice((3,5,65535,65537,9004123))
                bs = [rng.getrandbits(32) for _ in range(8)]
                vv = inputs(np.array(bs,dtype=np.uint64))
                actual = frozen(row,cc,vv)
                for lane in range(8):
                    initial = [int(v[lane]) for v in vv]
                    q = actual[8+special][lane]
                    self.assertLessEqual(q,quotient_maximum(op,cc[special][0]))
                    self.assertEqual(actual[-1][lane],sketch(row,cc,initial,q))
                    kk = [[z3.BitVecVal(k,32) for k in pair] for pair in cc]
                    symbolic = sketch(row,kk,[z3.BitVecVal(v,32) for v in initial],z3.BitVecVal(q,32))
                    self.assertEqual(actual[-1][lane],z3.simplify(symbolic).as_long())

    def test_known_full_target_is_retained_by_the_range_sketch(self):
        row = dict(order=['mad','//','^','^','mad'],
                   wiring=[[0,8,9],[8],[8,9],[10,11],[11,12,13]])
        cc = [[9,C4],[65536,0],[0,0],[NODE,0],[4097,C0]]
        pins = [(i,j,v) for i,pair in enumerate(cc) for j,v in enumerate(pair)]
        result = solve(row,5,points=[0,1,65536,MASK],pins=pins)
        self.assertEqual('range_sat',result['status'],result)
        self.assertTrue(result['relaxed_quotients_independent'])
        self.assertFalse(result['new_executable'])

    def test_absorption_cannot_repair_the_quotient_range(self):
        row = dict(order=['mad','//','&','|'],wiring=[[2,6,0],[8],[2,9],[9,10]])
        result = solve(row,3)
        self.assertEqual('range_unsat',result['status'],result)

    def test_invalid_prepared_grammar_is_rejected(self):
        for row in (dict(order=['//'],wiring=[[0,1]]),dict(order=['//','%'],wiring=[[0],[8]]),
                    dict(order=['shr16'],wiring=[[0]]),dict(order=['mad'],wiring=[[0,1,10]])):
            with self.assertRaises(ValueError):
                validate_row(row)
        with self.assertRaises(ValueError):
            quotient_maximum('//',1)


if __name__ == '__main__':
    unittest.main()
