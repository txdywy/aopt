import random
import unittest

import numpy as np
import z3

from tests.frozen_problem import Machine,DebugInfo
from codex.cegis_c1_absorb863 import _instr_options
from codex.search_live_boundary863 import inputs,expression,make_manifest,primary_path_avoiding,lsb_conditions,target,C0,C4,C5
from codex.build_live_boundary863 import CONTROL,verify


def frozen(row,constants,initial):
    setup={};cursor=0;program=[]
    def vec(data):
        nonlocal cursor
        base=cursor;cursor+=8;setup[base]=list(map(int,data));return base
    values=[vec(v) for v in initial];shift=vec([16]*8)
    for op,w,cc in zip(row['order'],row['wiring'],constants):
        pool=values+[vec([k]*8) for k in cc]
        args=[pool[j] for j in w];out=vec([0]*8)
        slot=('>>',out,args[0],shift) if op=='shr16' else ('multiply_add' if op=='mad' else op,out,*args)
        program.append({'valu':[slot]});values.append(out)
    m=Machine([0]*32,program,DebugInfo({}));m.enable_pause=False;m.enable_debug=False
    for base,data in setup.items():m.cores[0].scratch[base:base+8]=data
    m.run()
    return [m.cores[0].scratch[base:base+8] for base in values]


class LiveBoundaryTests(unittest.TestCase):
    def test_context_matches_frozen_middle(self):
        rng=random.Random(8642818)
        row=dict(order=['mad','mad','^','mad'],wiring=[[0,1,2],[0,2,3],[1,2],[3,4,5]])
        cc=[[33,0xE9F8CC1D],[16896,0xACCF6200],[0,0],[9,C4]]
        for _ in range(32):
            bs=[rng.getrandbits(32) for _ in range(8)]
            got=frozen(row,cc,[bs])
            want=inputs(np.array(bs,dtype=np.uint64),0,'arms',False)
            for actual,expected in zip((got[4],got[3],got[1],got[2],got[0]),want):
                self.assertEqual(actual,expected.tolist())

    def test_random_programs_match_frozen_numpy_integer_z3(self):
        rng=random.Random(8642819)
        for live,runtime in (('ez',False),('arms',False),('ez',True),('arms',True)):
            rows,_=make_manifest(live,runtime,16)
            for _ in range(100):
                row=rng.choice(rows);cc=[[rng.getrandbits(32),rng.getrandbits(32)] for _ in row['order']]
                ps=[rng.getrandbits(32) for _ in range(8)]
                roots=[rng.getrandbits(30) for _ in range(8)]
                nn=np.array([r^C5 for r in roots],dtype=np.uint64) if runtime else np.uint64(0xA76E1971)
                vv=inputs(np.array(ps,dtype=np.uint64),nn,live,runtime)
                actual=frozen(row,cc,vv)[-1]
                self.assertEqual(actual,[expression(row,cc,[int(v[i]) for v in vv]) for i in range(8)])
                with np.errstate(over='ignore'):
                    self.assertEqual(actual,expression(row,cc,vv).tolist())
                kk=[[z3.BitVecVal(k,32) for k in pair] for pair in cc]
                expected=[]
                for i,p in enumerate(ps):
                    node=z3.BitVecVal(roots[i]^C5 if runtime else 0xA76E1971,32)
                    values=inputs(z3.BitVecVal(p,32),node,live,runtime)
                    expected.append(z3.simplify(expression(row,kk,values)).as_long())
                self.assertEqual(actual,expected)

    def test_cut_guard_retains_every_primary_bypass(self):
        row=dict(order=['mad','shr16','mad'],wiring=[[0,2,3],[2],[3,4,5]])
        self.assertFalse(primary_path_avoiding(row,2,False,3))
        row['wiring'][-1]=[3,4,1]
        self.assertTrue(primary_path_avoiding(row,2,False,3))
        row['wiring'][-1]=[0,4,5]
        self.assertTrue(primary_path_avoiding(row,2,False,3))
        runtime=dict(order=['mad','shr16','mad'],wiring=[[2,3,0],[3],[4,5,2]])
        self.assertFalse(primary_path_avoiding(runtime,3,True,4))
        runtime['wiring'][-1][-1]=1
        self.assertTrue(primary_path_avoiding(runtime,3,True,4))

    def test_mod4_prefix_constraint_preserves_small_permutations(self):
        for w in _instr_options(1,'mad'):
            row=dict(order=['mad'],wiring=[w])
            for a in range(16):
                for b in range(16):
                    got=[expression(row,[[a,b]],inputs(p,0,'ez',False))&15 for p in range(16)]
                    if len(set(got))==16:
                        self.assertEqual(len({v&3 for v in got[:4]}),4)

    def test_fixed_root_candidate_is_not_an_executable_macro(self):
        bad=dict(CONTROL,runtime_node=False)
        with self.assertRaisesRegex(ValueError,'fixed-root'):
            verify(bad)

    def test_target_lsb_flips_on_primary_bit16(self):
        p=z3.BitVec('primary_dependency',32);r=z3.BitVec('root_dependency',30)
        n=z3.ZeroExt(2,r)^C5
        for live in ('ez','arms'):
            a=target(inputs(p,n,live,True)[0],n)
            b=target(inputs(p^(1<<16),n,live,True)[0],n)
            s=z3.SolverFor('QF_BV');s.set(timeout=5000)
            s.add(z3.Extract(0,0,a)==z3.Extract(0,0,b))
            self.assertEqual(s.check(),z3.unsat)

    def test_lsb_gate_rejects_a_runtime_multiplier_that_can_be_even(self):
        row=dict(order=['shr16','mad','mad'],wiring=[[0],[2,0,1],[3,4,5]])
        ks=[z3.BitVecs(f'gate_{i}a gate_{i}b',32) for i in range(3)]
        s=z3.SolverFor('QF_BV');s.add(*lsb_conditions(row,ks,'ez',False,0xA76E1971))
        self.assertEqual(s.check(),z3.unsat)
        cc=[[z3.BitVecVal(k,32) for k in pair] for pair in CONTROL['constants']]
        s=z3.SolverFor('QF_BV');s.add(z3.Not(z3.And(*lsb_conditions(CONTROL,cc,'ez',True,0))))
        self.assertEqual(s.check(),z3.unsat)

    def test_packed_arm_orientation_is_actually_checked(self):
        verify(CONTROL,seconds=2)
        # This computes the target XOR (canonical_u XOR actual_arm_a).
        # It is valid only in the canonical orientation, then fails on odd lanes.
        macro=dict(live='arms',runtime_node=True,
                   order=['mad','^','shr16','^','^','mad','^'],
                   wiring=[[4,6,7],[6,2],[0],[0,8],[9,5],[10,11,12],[11,7]],
                   constants=[[33,0xE9F8CC1D],[0,0],[0,0],[0,0],[0,0],[4097,C0],[0,0]])
        with self.assertRaisesRegex(ValueError,'swapped=True'):
            verify(macro,seconds=2)


if __name__=='__main__':unittest.main()
