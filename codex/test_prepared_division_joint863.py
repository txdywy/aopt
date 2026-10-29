import random
import unittest

import numpy as np
import z3

from codex.joint_division_relation863 import relation

from codex.search_joint_live_boundary863 import NODE,inputs,expression,make_manifest,solve
from codex.search_prepared_division_joint863 import control,division_row
from codex.test_live_boundary863 import frozen


class PreparedDivisionJointTests(unittest.TestCase):
    def test_product_relations_exact_in_both_directions(self):
        width=4;n,d,q=z3.BitVecs('relation_n relation_d relation_q',width)
        for op in ('//','%','cdiv'):
            expected=(z3.UDiv(n,d) if op=='//' else z3.URem(n,d) if op=='%' else
                      z3.Extract(width-1,0,z3.UDiv(z3.ZeroExt(width,n)+z3.ZeroExt(width,d)-1,z3.ZeroExt(width,d))))
            conditions=relation(n,q,d,op,width,tag='exact_'+op)
            s=z3.SolverFor('QF_BV');s.set(timeout=10000);s.add(*conditions,q!=expected)
            self.assertEqual(s.check(),z3.unsat)
            s=z3.SolverFor('QF_BV');s.set(timeout=10000);s.add(d!=0,q==expected,z3.Not(z3.And(*conditions)))
            if op=='%':
                # The remainder relation existentially quantifies the multiple.
                multiple=z3.BitVec('exact_%_multiple',width)
                s.add(multiple==z3.UDiv(n,d))
            self.assertEqual(s.check(),z3.unsat)

    def test_product_encoding_keeps_natural_positive(self):
        result=control('product')
        self.assertEqual(result['status'],'proved_equivalent')
        self.assertEqual(result['division_encoding'],'product')

    def test_search_rejects_unmodeled_or_runtime_denominator(self):
        comparison=dict(order=['mad','//','<','mad'],wiring=[[0,8,9],[8],[8,9],[10,11,12]])
        with self.assertRaisesRegex(AssertionError,'Unsupported operation'):solve(comparison,NODE,1)
        runtime=dict(order=['mad','//','^','mad'],wiring=[[0,8,9],[8,1],[8,9],[10,11,12]])
        with self.assertRaisesRegex(AssertionError,'private constant'):solve(runtime,NODE,1)

    def test_natural_floor_positive_program(self):
        self.assertEqual(control()['status'],'proved_equivalent')

    def test_ceil_carry_and_zero_denominator(self):
        row=dict(order=['cdiv'],wiring=[[0]])
        cases=[(0xffffffff,0xffffffff,1),(0xffffffff,2,0x80000000),(0,0xffffffff,0),(1,0xffffffff,1)]
        for n,d,q in cases:
            self.assertEqual(expression(row,[[d,0]],[n]),q)
            self.assertEqual(z3.simplify(expression(row,[[z3.BitVecVal(d,32),z3.BitVecVal(0,32)]],[n])).as_long(),q)
        with self.assertRaises(ZeroDivisionError):expression(row,[[0,0]],[1])

    def test_division_semantics_match_frozen_numpy_z3(self):
        rows,_=make_manifest(2,8642936);rng=random.Random(8642937)
        for op in ('//','%','cdiv'):
            for divisor in (1,2,3,17,65535,65536,65537,0x7fffffff,0x80000000,0xffffffff):
                for _ in range(4):
                    row=division_row(rng.choice(rows),op)
                    cc=[[rng.getrandbits(32),rng.getrandbits(32)] for _ in row['order']]
                    cc[row['order'].index(op)][0]=divisor
                    bs=[rng.getrandbits(32) for _ in range(8)];vv=inputs(np.array(bs,dtype=np.uint64))
                    with np.errstate(over='ignore'):expected=expression(row,cc,vv).tolist()
                    self.assertEqual(frozen(row,cc,vv)[-1],expected)
                    self.assertEqual([expression(row,cc,inputs(b)) for b in bs],expected)
                    kk=[[z3.BitVecVal(k,32) for k in pair] for pair in cc]
                    self.assertEqual([z3.simplify(expression(row,kk,inputs(z3.BitVecVal(b,32)))).as_long() for b in bs],expected)


if __name__=='__main__':unittest.main()
