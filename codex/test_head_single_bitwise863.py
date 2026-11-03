"""Independent controls for the AND/OR and three-MAD closure."""
from pathlib import Path
import json
import random
import subprocess
import unittest

from codex.audit_head_single_bitwise863 import audit
from codex.scan_head_final_quadratic863 import first_is_live
from codex.search_cross_encoding863 import C1,middle
from codex.search_head_polynomial_bitwise863 import control_coefficients,evaluate,search as poly_search
from codex.search_xor_equivalent_middle863 import execute,search
from codex.screen_xor_mad_polynomial import consistent
from codex.test_free_middle_keys863 import frozen

ROOT=Path(__file__).resolve().parent


class SingleBitwiseTests(unittest.TestCase):
    def test_complete_audit(self):
        row=audit()
        self.assertEqual('all_eight_single_and_or_orders_excluded',row['result'])
        self.assertEqual(8,row['closed_orders'])
        self.assertEqual(123008,row['input_module_pairs'])
        self.assertEqual(3973120,row['key_module_triples'])
        self.assertEqual({'projected_grammar_unsat':286,'unknown':20},row['coarse_counts'])
        self.assertEqual(90,row['refinement_queries'])
        self.assertEqual(151200,row['affine_final_wiring_patterns'])

    def test_all_refinements_admit_known_programs_when_first_is_live(self):
        saved=json.loads((ROOT/'head_and_final863_refinement.json').read_text())
        live=dead=0
        for row in saved['results']:
            op,source=row['operator'],row['source']
            second,last=tuple(row['second_operands']),tuple(row['last_operands'])
            raw=row['raw']
            fixed={int(k):tuple(v) for k,v in raw['fixed_operands'].items()}
            order=('mad','mad',op,'mad')
            program=[dict(op=instruction,operands=list(fixed[i]),constants=constants)
                for i,(instruction,constants) in enumerate(zip(order,
                    ([5,7],[11,19],[13,23],[17,29])))]
            result=search(order,5,5,4500,first_form='affine',fixed_operands=fixed,
                          control_program=program)
            if not first_is_live(source,second,last):
                dead+=1
                self.assertEqual('projected_grammar_unsat',result['result'])
                continue
            live+=1
            self.assertEqual('equivalent_on_reduced_domain',result['result'],result)
            for x in range(32):
                self.assertEqual(frozen(x,program)&31,execute(x,result['program'],5))
        self.assertEqual((78,12),(live,dead))

    def test_polynomial_and_or_controls_match_frozen_isa(self):
        rng=random.Random(864390)
        for op in ('&','|'):
            low=poly_search(op,5,5,control=True)
            self.assertEqual('equivalent_on_relaxed_low_domain',low['result'],low)
            coeff=control_coefficients(32)
            rows=[dict(op=code,operands=role,constants=constants) for code,role,constants in (
                ('mad',[0,1,2],[6,17]),('mad',[1,1,2],[21,23]),
                ('mad',[2,2,3],[29,31]),(op,[2,3,0],[0,0]))]
            for x in (0,0xffffffff,*[rng.getrandbits(32) for _ in range(100)]):
                self.assertEqual(frozen(x,rows),evaluate(x,coeff,op,32))

    def test_and_module_accelerator_matches_python_elimination(self):
        executable=Path('/tmp/aopt-and-module-oracle-test')
        subprocess.run(['clang++','-std=c++17','-O3','-o',str(executable),
            str(ROOT/'screen_head_and_module863.cpp')],capture_output=True,text=True,check=True)
        rng=random.Random(864391)
        samples=[0,1,2,3,32,64,511,1023]
        for source in ('input','key'):
            for _ in range(3):
                a,b,k=[rng.randrange(2048) for _ in range(3)]
                datum=(str(len(samples))+'\n'+' '.join(map(str,samples))+'\n'+f'{a} {b} {k}\n')
                result=subprocess.run([str(executable),source,'target'],input=datum,text=True,
                    capture_output=True,check=True)
                items=list(map(int,result.stdout.split()))
                actual=set(items[4:]) if items else set()
                if items:self.assertEqual(items[3],len(actual))
                matrix=[]
                for x in samples:
                    q=((a*x+b)&2047)&(x if source=='input' else k)
                    matrix.append([(x**i*q**j)&2047 for i in range(5) for j in range(5-i)])
                target=[middle(x^C1)&2047 for x in samples]
                expected={key for key in range(0,1024,2)
                          if consistent(matrix,[v^key for v in target],11)}
                self.assertEqual(expected,actual,(source,a,b,k))


if __name__=='__main__':unittest.main()
