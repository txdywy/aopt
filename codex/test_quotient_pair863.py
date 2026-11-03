import unittest

from codex.audit_quotient_pair863 import sum_scan,audit,applicable
from codex.search_early_boundary863 import NODE


class QuotientPairTests(unittest.TestCase):
    def test_complete_interval_scans_match_independent_smallword_sums(self):
        for width in range(2,9):
            m=1<<width;record=sum_scan(width)
            for op in ('floor','ceil'):
                sums={d:sum(n//d if op=='floor' else(n+d-1)//d for n in range(m))%m for d in range(1,m)}
                self.assertEqual(record[op]['sum_zero'],[d for d,v in sums.items() if v==0])
                self.assertEqual(record[op]['sum_half'],[d for d,v in sums.items() if v==m//2])

    def test_interval_tiling_permutation_controls_exhaustive(self):
        for width in range(2,7):
            m=1<<width;mask=m-1
            for d in range(2,m):
                for g in range(m):
                    actual=len({(g*n+n//d)&mask for n in range(m)})==m
                    expected=d&(d-1)==0 and g%(m//d)==0 and(g//(m//d))&1==1
                    self.assertEqual(actual,expected)
            ceil_p=[g for g in range(m) if len({(g*n+(n+1)//2)&mask for n in range(m)})==m]
            ceil_q=[g for g in range(1,m,2) if len({(n+g*((n+1)//2))&mask for n in range(m)})==m]
            self.assertEqual(ceil_p,[m//2-1]);self.assertEqual(ceil_q,[m//2-1])

    def test_target_predicates_have_legal_counterexamples(self):
        record=audit()
        self.assertEqual(len(record['bit_witnesses']),32)
        self.assertEqual(len(record['ceil_predicate_witnesses']),2)
        self.assertTrue(all(p['frozen_reference_checked'] for p in record['legal_points']))

    def test_guard_is_limited_to_prepared_affine_pair_and_true_cut(self):
        for op in ('shr','//','%','cdiv'):
            row=dict(order=['^','mad',op,'mad'],wiring=[[1,8],[0,8,9],[9],[9,11,10]])
            self.assertEqual(applicable(row,NODE)['family'],'G_P_plus_q')
            self.assertIsNone(applicable(row,0))
            row['wiring'][-1]=[10,11,9]
            self.assertEqual(applicable(row,NODE)['family'],'P_plus_G_q')
            row['wiring'][-1]=[10,2,9]
            self.assertIsNone(applicable(row,NODE))


if __name__=='__main__':unittest.main()
