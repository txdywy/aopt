import json
from itertools import product
from pathlib import Path
import random
import unittest

import numpy as np
import z3

from codex.audit_runtime_shift_band863 import audit as band_audit
from codex.audit_runtime_count_tail863 import audit as tail_audit
from codex.close_runtime_count_prefix863 import affine_certificate, raw_shift_certificates
from codex.search_first_runtime_count863 import count_manifest, interpolate, solve as count_solve
from codex.search_joint_live_boundary863 import inputs, wanted, NODE, C0, C4, MASK
from codex.search_runtime_count_prefix863 import count_word, prefix, manifest, solve
from codex.search_runtime_count_tail863 import FAMILIES, fits, fibre_fits, solve as tail_solve
from codex.enumerate_runtime_count_tail863 import vector_scan, scalar_scan
from codex.test_live_boundary863 import frozen
from tests.frozen_problem import Machine, DebugInfo


def frozen_prefix(row,constants,initial):
    """Execute actual dynamic-count slots; no replacement by a literal shift."""
    pool=[8*i for i in range(8)]+[64+8*i for i in range(len(constants))]
    if row['op']=='mad':
        first=('multiply_add',88,*(pool[i] for i in row['wiring']))
    else:
        right=pool[row['other']] if 'other' in row else 64
        first=(row['op'],88,pool[row['source']],right)
    datum=88 if row['numerator']=='count' else pool[7+len(constants)] if row['numerator']=='constant' else pool[row['numerator']]
    machine=Machine([0]*32,[{'valu':[first]},{'valu':[('>>',96,datum,88)]}],DebugInfo({}))
    machine.enable_pause=False;machine.enable_debug=False
    for i,values in enumerate(initial):machine.cores[0].scratch[8*i:8*i+8]=list(map(int,values))
    for i,value in enumerate(constants):machine.cores[0].scratch[64+8*i:72+8*i]=[value]*8
    machine.run()
    return machine.cores[0].scratch[88:96],machine.cores[0].scratch[96:104]


class RuntimeCountPrefixTests(unittest.TestCase):
    def test_full32_dynamic_prefixes_match_frozen_integer_numpy_and_z3(self):
        rng=random.Random(8642967)
        rows=[r for family in ('constant','runtime','mad') for r in manifest(family)]
        cases=rng.sample(rows,72)
        cases += [dict(op='mad',wiring=[3,8,9],numerator=n)
                  for n in (0,'constant','count') for _ in range(8)]
        for rank,row in enumerate(cases):
            cc=[rng.getrandbits(32) for _ in range(3 if row['op']=='mad' else 2)]
            if rank>=72:cc[:2]=[0,(0,1,15,16,23,31,32,65536)[(rank-72)%8]]
            bs=[rng.getrandbits(32) for _ in range(8)]
            with np.errstate(over='ignore'):vv=inputs(np.array(bs,dtype=np.uint64))
            actual_count,actual=frozen_prefix(row,cc,vv)
            with np.errstate(over='ignore'):
                self.assertEqual(actual_count,np.broadcast_to(count_word(row,cc,vv),(8,)).tolist())
                self.assertEqual(actual,np.broadcast_to(prefix(row,cc,vv),(8,)).tolist())
            kk=[z3.BitVecVal(k,32) for k in cc]
            for lane,b in enumerate(bs):
                self.assertEqual(actual[lane],prefix(row,cc,inputs(b)))
                self.assertEqual(actual[lane],z3.simplify(prefix(row,kk,inputs(z3.BitVecVal(b,32)))).as_long())

    def test_clamping_positive_control_is_verified_over_all32_bits(self):
        result=solve(dict(op='&',source=3,numerator='constant'),5,True,[(0,65536),(1,1)])
        self.assertEqual('lsb_feasible_full32',result['status'],result)
        self.assertEqual([65536,1],result['constants'])
        self.assertEqual('unsat',result['events'][-1]['full32_check'])
        self.assertFalse(result['universal_count_band_used'])

    def test_mad_positive_control_preserves_bv_sorts_and_literal16(self):
        result=solve(dict(op='mad',wiring=[3,8,9],numerator=1),5,'u16',[(0,0),(1,16)])
        self.assertEqual('lsb_feasible_full32',result['status'],result)
        self.assertEqual([0]*4,result['class_flips'])

    def test_toy_targets_cannot_borrow_actual_target_band(self):
        with self.assertRaisesRegex(AssertionError,'specific'):
            solve(dict(op='&',source=3,numerator='constant'),1,True,use_count_band=True)

    def test_self_shift_is_zero_for_every_uint32(self):
        t=z3.BitVec('test_runtime_self_shift',32)
        s=z3.SolverFor('QF_BV');s.add(z3.LShR(t,t)!=0)
        self.assertEqual(z3.unsat,s.check())

    def test_canonical_prefix_and_count_manifests_are_unique(self):
        self.assertEqual([240,1080,4200],[len(manifest(f)) for f in ('constant','runtime','mad')])
        rows=count_manifest();self.assertEqual(552,len(rows))
        self.assertEqual(552,len({json.dumps(r,sort_keys=True) for r in rows}))
        for family in ('constant','runtime','mad'):
            with self.assertRaises(ValueError):manifest(family+'invalid')

    def test_every_prepared_raw_shift_has_a_concrete_same_class_collision(self):
        pairs=raw_shift_certificates();self.assertEqual(64,len(pairs))
        for row in pairs:
            a,b=row['primary_pair'];va,vb=inputs(a),inputs(b)
            self.assertEqual([x&1 for x in va],[x&1 for x in vb])
            self.assertNotEqual(((va[row['source']]>>row['count'])^wanted(va,NODE))&1,
                                ((vb[row['source']]>>row['count'])^wanted(vb,NODE))&1)

    def test_affine_source_reachability_and_nonzero_gating_certificates(self):
        result=affine_certificate()
        self.assertEqual(list(range(16,32)),[r['valuation'] for r in result['nonzero_coefficient_parity_gating']])
        for row in result['source_reachability']:
            self.assertEqual(row['value'],inputs(row['primary_b'])[row['source']])
        # Independent exhaustive small-word version of the two congruence arguments.
        for a in range(256):
            for b in range(256):
                if all(8<=((a*x+b)&255)<=11 for x in (0,1,4)):
                    self.assertEqual(0,a)
        for a in range(32,256,32):
            for b in range(256):
                parities={i&1 for i in range(256) if ((a*i+b)&255)<16}
                self.assertLessEqual(len(parities),1)

    def test_count_screen_retains_a_prepared16_count(self):
        row=dict(op='mad',wiring=[0,8,9])
        retained=count_solve(row,5,False,[(0,0),(1,16)])
        self.assertEqual('count_screen_sat',retained['status'],retained)
        excluded=count_solve(row,5,True,[(0,0),(1,16)])
        self.assertEqual('nonconstant_count_unsat',excluded['status'],excluded)

    def test_count_interpolation_quantifies_over_free_root_parameters(self):
        rows=[dict(op='mad',wiring=[0,8,9]),dict(op='mad',wiring=[2,8,9]),
              dict(op='mad',wiring=[2,2,8]),dict(op='^',source=3,other=7),
              dict(op='&',source=6),dict(op='|',source=2)]
        for row in rows:
            result=interpolate(row)
            self.assertEqual('finite_constancy_implies_full32_constant',result['status'],result)
            self.assertTrue(result['root_constants_unrestricted'])

    def test_saved_first_count_proof_covers_all552_shapes(self):
        record=json.loads(Path('codex/first_runtime_count863_closed_20261029.json').read_text())
        self.assertTrue(record['every_one_T_count_must_be_root_constant'])
        self.assertEqual({'finite_constancy_implies_full32_constant':552},record['counts'])
        self.assertEqual(552,len(record['checks']))
        self.assertEqual((16,23),(record['constant_count_minimum'],record['constant_count_maximum']))


class RuntimeShiftBandTests(unittest.TestCase):
    def test_complete_projection_and_cardinality_certificate(self):
        band=band_audit();table=band['universal_fibres']
        self.assertEqual(32768,table['complete_cases'])
        self.assertEqual(33554432,table['primary_inputs'])
        self.assertEqual((387,483),(table['minimum_distinct'],table['maximum_distinct']))
        self.assertEqual(32768,sum(table['histogram'].values()))
        self.assertEqual((16,23),(band['count_low16_minimum'],band['count_low16_maximum']))
        self.assertEqual(65535,band['global_quotient_maximum'])
        self.assertTrue(band['prepared_numerator_impossible'])

    def test_full_target_formula_and_minimum_fibre_match_frozen(self):
        row=dict(order=['mad','shr16','^','^','mad'],
                 wiring=[[0,8,9],[8],[8,9],[10,11],[11,12,13]])
        cc=[[9,C4],[0,0],[0,0],[NODE,0],[4097,C0]]
        sig=band_audit()['universal_fibres']['minimum_signature']
        for low,high in ((sig['low9'],sig['high19_through24']),(0,0),(511,63)):
            residues=set()
            for start in range(0,1024,8):
                bs=[low+(high<<19)+(j<<9) for j in range(start,start+8)]
                with np.errstate(over='ignore'):vv=inputs(np.array(bs,dtype=np.uint64))
                actual=frozen(row,cc,vv)
                self.assertTrue(all(0<=q<=65535 for q in actual[9]))
                self.assertEqual(actual[-1],[wanted(inputs(b),NODE) for b in bs])
                residues.update(y&511 for y in actual[-1])
            self.assertGreaterEqual(len(residues),387)
            if (low,high)==(sig['low9'],sig['high19_through24']):self.assertEqual(387,len(residues))

    def test_eight_fibre_information_is_independent_of_prefix_length(self):
        rng=random.Random(8642971)
        for _ in range(32):
            base=rng.getrandbits(32)&~(7<<16)
            vv=[inputs(base|(j<<16)) for j in range(8)]
            self.assertTrue(all(len({v[i]&65535 for v in vv})==1 for i in range(8)))
            self.assertEqual(8,len({wanted(v,NODE)&7 for v in vv}))
            for c in (14,15,30,31,32,65536):
                # Maximal possible low3 image sizes, including clamped0.
                fixed_low=rng.randrange(65536)
                numerator=[fixed_low|(j<<16) for j in range(8)]
                image={(n>>c)&7 for n in numerator}|{0}
                self.assertLess(len(image),8)

    def test_saved_closure_keeps_timeouts_and_separates_executable_score(self):
        record=json.loads(Path('codex/runtime_count_prefix863_closed_20261029.json').read_text())
        self.assertEqual(5520,len(record['closed_prefixes']))
        self.assertEqual(50,record['original_search_observations']['mad']['counts']['unknown'])
        self.assertEqual(0,record['remaining_in_stated_prefix_family'])
        self.assertFalse(record['new_executable'])
        self.assertEqual(864,record['best_executable_cycles'])


class RuntimeCountTailTests(unittest.TestCase):
    def test_active_mad_tail_ranges_match_frozen_isa(self):
        rng=random.Random(8642977)
        for family in FAMILIES:
            for c in range(16,24):
                h=[rng.getrandbits(32) for _ in range(8)]
                q=[rng.randrange(65536) for _ in range(8)]
                factors=(q,[c]*8,h) if family==FAMILIES[0] else ([c]*8,h,q)
                machine=Machine([0]*32,[{'valu':[('multiply_add',24,0,8,16)]}],DebugInfo({}))
                machine.enable_pause=False;machine.enable_debug=False
                for i,word in enumerate(factors):machine.cores[0].scratch[8*i:8*i+8]=word
                machine.run();actual=machine.cores[0].scratch[24:32]
                for x,y in zip(h,actual):
                    self.assertTrue(fits(family,x,y))
                    self.assertTrue(z3.is_true(z3.simplify(fits(family,z3.BitVecVal(x,32),z3.BitVecVal(y,32)))))

    def test_range_search_retains_automatic_controls(self):
        for family in FAMILIES:
            result=tail_solve(dict(op='mad',wiring=[0,8,9]),family,5,True,
                              [(0,9),(1,C4)],True)
            self.assertEqual('range_sat',result['status'],result)
            self.assertTrue(result['coupled_fibre_count'])

    def test_fibre_count_is_one_shared_choice_not_eight_independent_choices(self):
        hh=[z3.BitVecVal(1<<24,32) for _ in range(8)]
        yy=[z3.BitVecVal((16+j)*(1<<24)+17,32) for j in range(8)]
        self.assertTrue(z3.is_true(z3.simplify(fibre_fits(FAMILIES[1],hh,yy,False))))
        self.assertTrue(z3.is_false(z3.simplify(fibre_fits(FAMILIES[1],hh,yy,True))))

    def test_fibre_relaxation_allows_one_clamped_exception(self):
        hh=[z3.BitVecVal(0,32) for _ in range(8)]
        yy=[z3.BitVecVal(0,32) for _ in range(7)]+[z3.BitVecVal(MASK,32)]
        for family in FAMILIES:
            self.assertTrue(z3.is_true(z3.simplify(fibre_fits(family,hh,yy,True))))
        yy[6]=z3.BitVecVal(MASK,32)
        for family in FAMILIES:
            self.assertTrue(z3.is_false(z3.simplify(fibre_fits(family,hh,yy,True))))

    def test_full_small_word_high_assignments_match_both_product_scanners(self):
        rng=random.Random(8642989);rectangles=([0,1,2,3],[4,5,6,7])
        for _ in range(32):
            ys=[rng.randrange(16) for _ in range(8)]
            vector,models=vector_scan(ys,rectangles,2,range(4),True)
            self.assertEqual(vector,scalar_scan(ys,rectangles,2,range(4)))
            expected=[]
            for h,c in product((1,3),range(4)):
                inverse=pow(h,-1,4)
                qs=[((y-c)*inverse)&3 for y in ys]
                options=[[t for t in range(4) if (q*(h+4*t)+c)&15==y] for q,y in zip(qs,ys)]
                feasible=True
                for order in rectangles:
                    if not any((ts[3]+ts[0]-ts[1]-ts[2])%4==0
                               for ts in product(*(options[i] for i in order))):
                        feasible=False;break
                if feasible:expected.append((h,c))
            self.assertEqual(set(expected),set(models))

    def test_product_inverse_scan_retains_a_true_multiplicative_model(self):
        rectangles=([0,1,2,3],[4,5,6,7])
        h=[3+16*i for i in range(8)];q=list(range(1,9))
        ys=[(a*b+5)&255 for a,b in zip(h,q)]
        vector,models=vector_scan(ys,rectangles,4,range(4,8),True)
        self.assertIn((3,5),models)
        self.assertEqual(vector,scalar_scan(ys,rectangles,4,range(4,8)))

    def test_all_paired_rectangles_replay_free_parameters_and_complete_inverse_scan(self):
        result=tail_audit()
        self.assertEqual(1656,result['closed_connections'])
        self.assertEqual(552,len(result['checks']))
        self.assertEqual(0,result['remaining_in_stated_three_tail_families'])
        self.assertEqual(4718592,result['inverse_product_scan']['evaluated_parameters'])
        self.assertEqual(0,result['inverse_product_scan']['remaining_models'])
        self.assertEqual(194,result['original_search_counts']['unknown'])


if __name__=='__main__':unittest.main()
