"""Ensure broader four-op partitions admit independent known programs."""
import random
import unittest

from codex.scan_head_nonlinear863 import orders
from codex.search_xor_equivalent_middle863 import execute,search
from codex.test_free_middle_keys863 import frozen
from codex.screen_head_square863 import certificate,difference,orthogonal,POINTS


def witness(order,first):
    roles={'affine':[0,1,2],'square_const':[0,0,1],'square_x':[0,0,0]}[first]
    rows=[dict(op='mad',operands=roles,constants=[5,7])]
    for i,op in enumerate(order[1:],1):
        rows.append(dict(op=op,operands=[i,i+1,i+2]if op=='mad'else[0,i,0],constants=[9,13]))
    return rows


class HeadNonlinearTests(unittest.TestCase):
    def test_all_18_automatic_controls_with_fixed_dag_roles(self):
        for order in orders():
            for first in('affine','square_const','square_x'):
                rows=witness(order,first)
                report=search(order,5,10,9000,first_form=first,control_program=rows,
                    fixed_operands={i:r['operands']for i,r in enumerate(rows)})
                self.assertEqual('equivalent_on_reduced_domain',report['result'],report)
                self.assertEqual(32,report['verified_inputs'])
                self.assertFalse(report['new_executable'])
                for x in range(32):
                    self.assertEqual(frozen(x,rows)&31,execute(x,report['program'],5))

    def test_fixed_roles_match_frozen_full32_and_original_control(self):
        rng=random.Random(86462)
        for order in orders():
            for first in('affine','square_const','square_x'):
                rows=witness(order,first)
                for x in (0,0xffffffff,rng.getrandbits(32)):
                    self.assertEqual(frozen(x,rows),execute(x,rows,32))
        roles={0:(0,1,2),1:(0,2,3),2:(1,2,0),3:(3,4,5)}
        result=search(('mad','mad','^','mad'),32,10,9000,pin_original=True,
                      first_form='affine',fixed_operands=roles)
        self.assertEqual('proved_equivalent_full32',result['result'],result)

    def test_rejects_invalid_fixed_roles(self):
        for fixed in({4:(0,1,2)},{1:(0,4,1)},{1:(2,0,1)},{2:(1,2,1)}):
            with self.assertRaises(ValueError):
                search(('mad','mad','^','mad'),5,fixed_operands=fixed)

    def test_square_rectangle_for_bitwise_prefixes_and_polynomial_tails(self):
        report=certificate()
        self.assertEqual([1024],report['target_mixed_differences'])
        self.assertEqual(2048,report['checked_output_keys'])
        rng=random.Random(86463)
        for first in('square_const','square_x'):
            for _ in range(160):
                offset,key,other=[rng.getrandbits(32)for _ in range(3)]
                outputs=[];prefix=[[],[],[],[]]
                for x in POINTS:
                    t=x*x+(x if first=='square_x'else offset)
                    q=(t^key)&(x|other)
                    r=(q|x)^(t&other)
                    s=(q^t)|(r&x)
                    for rows,v in zip(prefix,(t,q,r,s)):rows.append(v&2047)
                    outputs.append((r*r*s+q*x*x+other*t*s+17*q+key)&2047)
                self.assertTrue(all(orthogonal(rows)for rows in prefix))
                self.assertEqual(0,difference(outputs))


if __name__=='__main__':unittest.main()
