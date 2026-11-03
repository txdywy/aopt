"""Check solver-free bounds against exhaustive tiny legal calendars."""
from itertools import product
from random import Random
from types import SimpleNamespace
import unittest

from codex.bound_boundary_no_solver863 import prefix, suffix
from codex.bound_chain_menu863 import report
from codex.bound_bilinear_grid863 import compute,contributions


def graph(engines, edges):
    deps=[{} for _ in engines]
    succ=[[] for _ in engines]
    for before,after,lag in edges:
        deps[after][before]=lag
        succ[before].append((after,lag))
    return SimpleNamespace(ops=[SimpleNamespace(engine=e) for e in engines],
                           deps=deps,succ=succ)


class BoundaryBoundTests(unittest.TestCase):
    def test_prefix_respects_asap_and_capacity(self):
        ir=graph(['load','valu','alu'],[(0,1,1),(1,2,1)])
        result=prefix(ir,3)
        self.assertEqual([0,8,9],result['availability_by_cycle'])
        self.assertEqual(9,result['objective_upper_bound'])

    def test_bounds_cover_exhaustive_small_schedules(self):
        rng=Random(8631027)
        limits={'alu':12,'valu':6,'load':2,'store':2,'flow':1}
        checked=0
        for _ in range(40):
            engines=[rng.choice(tuple(limits)) for _ in range(5)]
            edges=[(i,j,rng.randrange(2)) for j in range(5) for i in range(j)
                   if rng.randrange(5)==0]
            ir=graph(engines,edges)
            first=prefix(ir,3)['objective_upper_bound']
            last=suffix(ir,3)['objective_upper_bound']
            for times in product(range(4),repeat=5):
                if any(times[b] < times[a]+lag for a,b,lag in edges):continue
                if any(sum(e==eng and t==cycle for e,t in zip(engines,times))>limit
                       for eng,limit in limits.items() for cycle in range(4)):continue
                weights=[8 if e=='valu' else 1 if e=='alu' else 0 for e in engines]
                self.assertLessEqual(sum(w for w,t in zip(weights,times) if t<3),first)
                self.assertLessEqual(sum(w for w,t in zip(weights,times) if t>=1),last)
                checked+=1
        self.assertGreater(checked,1000)

    def test_cycles_are_rejected(self):
        ir=graph(['alu','alu'],[(0,1,1),(1,0,1)])
        with self.assertRaisesRegex(AssertionError,'cyclic'):
            prefix(ir)
        with self.assertRaisesRegex(AssertionError,'cyclic'):
            suffix(ir)

    def test_chain_menu_does_not_claim_a_universal_bound(self):
        row=report()
        self.assertEqual(51,row['without_joint']['maximum_saving'])
        self.assertEqual(53,row['with_joint']['maximum_saving'])
        self.assertEqual(863,row['conditional_cycle_lower_bound'])
        self.assertFalse(row['new_executable'])
        self.assertIn('no universal lower bound',row['scope'])

    def test_bilinear_grid_minimum_and_independent_witness(self):
        row=compute();w=row['witness']
        actual={a+b+c for a in contributions(*w['x']) for b in (0,w['y_delta'])
                for c in contributions(*w['v'])}
        self.assertEqual(32,len(actual))
        self.assertEqual(50,max(actual)-min(actual)+1)
        self.assertEqual(800,row['minimum_word_span'])
        self.assertFalse(row['new_executable'])


if __name__=='__main__':unittest.main()
