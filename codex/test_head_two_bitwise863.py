"""Positive controls and coverage for the mixed two-bitwise exploration."""
import json
from pathlib import Path
import unittest

from codex.scan_head_two_bitwise863 import cases
from codex.audit_head_two_bitwise863 import audit
from codex.audit_head_two_bitwise_refined863 import audit as refined_audit
from codex.search_cross_encoding863 import C1
from codex.search_xor_equivalent_middle863 import execute,search
from codex.test_free_middle_keys863 import frozen

ROOT=Path(__file__).resolve().parent


class MixedTwoBitwiseTests(unittest.TestCase):
    def test_complete_scan_and_automatic_positive_controls(self):
        expected=set(cases())
        self.assertEqual(96,len(expected))
        self.assertEqual(48,len({order for order,first in expected}))
        saved=json.loads((ROOT/'head_two_bitwise863_results.json').read_text())
        self.assertEqual(len(expected),len(saved['results']))
        actual=set()
        for row in saved['results']:
            case=tuple(row['sequence']),row['first_form']
            self.assertIn(case,expected)
            self.assertNotIn(case,actual)
            self.assertEqual((11,C1,None),(row['width'],row['fixed_input_key'],row['fixed_output_key']))
            self.assertIn(row['result'],('projected_grammar_unsat','unknown','equivalent_on_reduced_domain'))
            actual.add(case)
        self.assertEqual(expected,actual)
        coverage=audit()
        self.assertEqual((48,28,20),(coverage['total_orders'],coverage['closed_orders'],coverage['open_orders']))
        self.assertEqual({'projected_grammar_unsat':69,'unknown':27},coverage['raw_counts'])
        for order,first in expected:
            fixed={};rows=[]
            for i,op in enumerate(order):
                if i==0 and op=='mad':
                    role={'affine':(0,1,2),'square_const':(0,0,1),'square_x':(0,0,0)}[first]
                else:
                    role=(i,i+1,i+2) if op=='mad' else (i,i+1,0)
                fixed[i]=role
                rows.append(dict(op=op,operands=list(role),constants=[5+2*i,7+4*i]))
            result=search(order,5,5,4500,first_form=first,fixed_operands=fixed,control_program=rows)
            self.assertEqual('equivalent_on_reduced_domain',result['result'],result)
            for x in range(32):
                self.assertEqual(frozen(x,rows)&31,execute(x,result['program'],5))

    def test_operand_refinements_and_known_programs(self):
        report=refined_audit()
        self.assertEqual((48,31,17),(report['total_orders'],report['closed_orders'],report['open_orders']))
        self.assertEqual({'projected_grammar_unsat':46,'unknown':24},report['raw_refinement_counts'])
        self.assertEqual({'three_operations':8,'still_open':16},report['unknown_reductions'])
        self.assertEqual(8,len(report['remaining_distinct_refinement_classes']))
        saved=json.loads((ROOT/'head_two_bitwise863_refinement.json').read_text())
        self.assertEqual(70,len(saved['results']))
        for row in saved['results']:
            order=tuple(row['sequence'])
            second=tuple(row['fixed_operands']['1'])
            fixed={0:(0,1,2),1:second,2:(1,2,0),3:(0,3,4)}
            rows=[dict(op=op,operands=list(fixed[i]),constants=constants)
                for i,(op,constants) in enumerate(zip(order,([5,7],[11,19],[13,23],[17,29])))]
            result=search(order,5,5,4500,first_form='affine',fixed_operands=fixed,control_program=rows)
            self.assertEqual('equivalent_on_reduced_domain',result['result'],result)
            for x in range(32):
                self.assertEqual(frozen(x,rows)&31,execute(x,result['program'],5))


if __name__=='__main__':unittest.main()
