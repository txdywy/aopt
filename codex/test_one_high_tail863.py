import unittest

from codex.audit_one_high_tail863 import audit,information_fibre,ordinary_tail_certificate
from codex.search_early_boundary863 import NODE


class OneHighTailTests(unittest.TestCase):
    def test_entire_tail_and_fullword_scalar_evidence(self):
        result=audit()
        self.assertTrue(result['complete_in_stated_scope'])
        self.assertEqual(len(result['tail_manifest']),913)
        self.assertEqual(result['full_odd_inverse_parameters'],19327352832)
        self.assertTrue(all(p['frozen_reference_checked'] for p in result['legal_points']))
        info=information_fibre()
        self.assertEqual(len(set(info['target_residues'])),478)
        self.assertEqual(info['prepared_floor_divisor_max'],9004124)
        self.assertEqual(info['prepared_ceil_divisor_max'],9023040)

    def test_information_bound_smallword_exhaustive_controls(self):
        mask=255;count=8
        for d in range(mask//(count-1)+1,mask+1):
            self.assertLess(len({n//d for n in range(mask+1)}),count)
            for residue in range(8):
                self.assertLess(len({n%d&7 for n in range(residue,mask+1,8)}),count)
        for d in range((mask-1)//(count-2)+1,mask+1):
            self.assertLess(len({(n+d-1)//d for n in range(mask+1)}),count)
        # A denominator inside the necessary bound can transmit all8 residues.
        self.assertEqual({n%17&7 for n in range(0,256,8)},set(range(8)))

    def test_application_keeps_computed_bypasses_and_root_scope(self):
        for op in ('shr','//','%','cdiv'):
            row=dict(order=['mad','mad',op,'mad'],wiring=[[0,8,9],[8,9,5],[9],[10,11,1]])
            proof=ordinary_tail_certificate(row,NODE)
            self.assertEqual(proof['tail_class'],'odd_scalar_q_initial_addend')
            self.assertIsNone(ordinary_tail_certificate(row,0))
            row['wiring'][-1]=[10,11,9]
            self.assertIsNone(ordinary_tail_certificate(row,NODE))
        natural=dict(order=['mad','//','^','mad'],wiring=[[0,8,9],[8],[8,9],[10,11,12]])
        self.assertIsNone(ordinary_tail_certificate(natural,NODE))


if __name__=='__main__':unittest.main()
