import random
import unittest

import numpy as np
import z3

from codex.search_joint_live_boundary863 import NODE, inputs, wanted, expression, make_manifest, solve, prefix_cut_guards
from codex.search_prepared_shift_joint863 import control, with_prepared_shift
from codex.test_live_boundary863 import frozen


class PreparedShiftJointTests(unittest.TestCase):
    def test_second_prefix_cut_closes_entire_shift_domain(self):
        row=dict(order=['^','mad','shr','mad'],wiring=[[1,8],[0,8,9],[9],[10,11,9]])
        prefix=dict(order=row['order'][:2],wiring=row['wiring'][:2])
        for k in range(4):
            for offset in range(4):
                images={expression(prefix,[[k,0],[offset,0]],inputs(b))&3 for b in range(4)}
                self.assertLessEqual(len(images),2)
        result=solve(row,NODE,2)
        self.assertEqual(result['status'],'unsat')
        self.assertEqual(result['prefix_mod4_guard_cuts'],[9])
        self.assertIn('T-only value cut',result['certificate'])

    def test_prefix_guard_preserves_bypasses_and_non_T_initials(self):
        kk=[z3.BitVecs(f'prefix_guard_{i}a prefix_guard_{i}b',32) for i in range(4)]
        bypass=dict(order=['^','mad','shr','mad'],wiring=[[1,8],[0,8,9],[9],[10,11,3]])
        self.assertEqual(prefix_cut_guards(bypass,kk),[])
        early=dict(order=['mad','shr','^','mad'],wiring=[[4,8,9],[8],[8,9],[10,11,12]])
        self.assertEqual(prefix_cut_guards(early,kk),[])

    def test_multiple_shifts_rejected_by_single_shift_search(self):
        row=dict(order=['mad','shr','shr','mad'],wiring=[[0,8,9],[8],[9],[10,11,12]])
        with self.assertRaisesRegex(AssertionError,'exactly one right shift'):
            solve(row,NODE,1)

    def test_complete_count_domain_lemma_full32(self):
        b = z3.BitVec('prepared_shift_lemma_b',32)
        a, c = inputs(b), inputs(b^65536)
        solver = z3.SolverFor('QF_BV'); solver.set(timeout=10000)
        solver.add(z3.Or(*(z3.Extract(15,0,x) != z3.Extract(15,0,y) for x,y in zip(a,c))))
        self.assertEqual(solver.check(),z3.unsat)
        solver = z3.SolverFor('QF_BV'); solver.set(timeout=10000)
        solver.add(z3.Extract(0,0,wanted(a,NODE)) == z3.Extract(0,0,wanted(c,NODE)))
        self.assertEqual(solver.check(),z3.unsat)

    def test_prepared_count_recovered_without_pinning(self):
        result=control()
        self.assertEqual(result['status'],'proved_equivalent')
        self.assertEqual(result['constants'][1][0],16)

    def test_count_semantics_match_numpy_z3_and_frozen(self):
        rows,_=make_manifest(2,8642933);rng=random.Random(8642934)
        for count in (0,1,7,15,16,17,19,23,31,32,63,0xffffffff):
            for _ in range(8):
                row=with_prepared_shift(rng.choice(rows))
                cc=[[rng.getrandbits(32),rng.getrandbits(32)] for _ in row['order']]
                cc[row['order'].index('shr')][0]=count
                bs=[rng.getrandbits(32) for _ in range(8)]
                values=inputs(np.array(bs,dtype=np.uint64))
                with np.errstate(over='ignore'):
                    expected=expression(row,cc,values).tolist()
                self.assertEqual(frozen(row,cc,values)[-1],expected)
                self.assertEqual([expression(row,cc,inputs(b)) for b in bs],expected)
                kk=[[z3.BitVecVal(k,32) for k in pair] for pair in cc]
                self.assertEqual([z3.simplify(expression(row,kk,inputs(z3.BitVecVal(b,32)))).as_long() for b in bs],expected)


if __name__=='__main__':
    unittest.main()
