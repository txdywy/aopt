"""A frozen-ISA check of the in-place three-uniform packing primitive."""
import random
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from frozen_problem import Machine


class TripleUniformTests(unittest.TestCase):
    def test_two_overlapping_selects_preserve_all_three_vectors(self):
        rng=random.Random(864392)
        for _ in range(128):
            a,b,c=[rng.getrandbits(32) for _ in range(3)]
            machine=Machine([],[],None);core=machine.cores[0]
            core.scratch[10:14]=[a]*4
            core.scratch[14:16]=[b]*2
            core.scratch[30:34]=[c]*4
            core.scratch[100:108]=[1]*6+[0]*2
            core.scratch[108:116]=[1]*4+[0]*4
            machine.scratch_write={}
            machine.flow(core,'vselect',14,100,10,8)
            for dest,value in machine.scratch_write.items():core.scratch[dest]=value
            self.assertEqual([a]*4+[b]*4,core.scratch[14:22])
            machine.scratch_write={}
            machine.flow(core,'vselect',22,108,18,26)
            for dest,value in machine.scratch_write.items():core.scratch[dest]=value
            self.assertEqual([a]*8,core.scratch[10:18])
            self.assertEqual([b]*8,core.scratch[18:26])
            self.assertEqual([c]*8,core.scratch[26:34])

    def test_ones_bootstrap_without_existing_ones_vector(self):
        machine=Machine([],[],None);core=machine.cores[0]
        # Select destination 73..80 overlaps both condition seeds and three
        # initial zeros. The machine reads all operands before its writes.
        core.scratch[70:73]=[1]*3
        core.scratch[76:78]=[1]*2
        machine.scratch_write={}
        machine.flow(core,'vselect',73,76,70,68)
        for dest,value in machine.scratch_write.items():core.scratch[dest]=value
        self.assertEqual([1]*8,core.scratch[70:78])
        self.assertEqual([0]*8,core.scratch[78:86])

    def test_self_masked_ones_and_two_nonzero_constants(self):
        rng=random.Random(863901)
        for _ in range(128):
            b=rng.randrange(1,1<<32)
            c=rng.randrange(1,1<<32)
            machine=Machine([],[],None);core=machine.cores[0]
            core.scratch[10:14]=[1]*4
            core.scratch[14:16]=[b]*2
            core.scratch[30:34]=[c]*4
            machine.scratch_write={}
            machine.flow(core,'vselect',14,10,10,8)
            for dest,value in machine.scratch_write.items():core.scratch[dest]=value
            machine.scratch_write={}
            machine.flow(core,'vselect',22,18,18,26)
            for dest,value in machine.scratch_write.items():core.scratch[dest]=value
            self.assertEqual([1]*8,core.scratch[10:18])
            self.assertEqual([b]*8,core.scratch[18:26])
            self.assertEqual([c]*8,core.scratch[26:34])
            self.assertEqual([0]*8,core.scratch[34:42])

    def test_self_masked_ones_and_one_nonzero_constant(self):
        rng=random.Random(863903)
        for _ in range(128):
            value=rng.randrange(1,1<<32)
            machine=Machine([],[],None);core=machine.cores[0]
            core.scratch[10:14]=[1]*4
            core.scratch[22:26]=[value]*4
            machine.scratch_write={}
            machine.flow(core,'vselect',14,10,10,18)
            for dest,word in machine.scratch_write.items():core.scratch[dest]=word
            self.assertEqual([1]*8,core.scratch[10:18])
            self.assertEqual([value]*8,core.scratch[18:26])
            self.assertEqual([0]*8,core.scratch[26:34])

    def test_four_uniform_chain_with_two_inplace_intermediate_seeds(self):
        rng=random.Random(863902)
        for _ in range(128):
            a,b,c,d=[rng.getrandbits(32) for _ in range(4)]
            machine=Machine([],[],None);core=machine.cores[0]
            core.scratch[10:14]=[a]*4
            core.scratch[14:16]=[b]*2
            core.scratch[22:24]=[c]*2
            core.scratch[38:42]=[d]*4
            core.scratch[100:108]=[1]*6+[0]*2
            core.scratch[108:116]=[1]*4+[0]*4
            for dest,mask,yes,no in ((14,100,10,8),(22,100,18,16),(30,108,26,34)):
                machine.scratch_write={}
                machine.flow(core,'vselect',dest,mask,yes,no)
                for addr,value in machine.scratch_write.items():core.scratch[addr]=value
            for start,value in ((10,a),(18,b),(26,c),(34,d)):
                self.assertEqual([value]*8,core.scratch[start:start+8])


if __name__=='__main__':unittest.main()
