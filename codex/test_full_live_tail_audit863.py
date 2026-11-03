"""Independent controls for congruence and high-output tail exclusions."""
import itertools
import json
from pathlib import Path
import random
import unittest

import z3

from codex.audit_full_live_tail863 import (COMPATIBLE, IDENTITY_HIGH, SPLITTING,
    bypass_witnesses, congruence_certificate, coverage, high_identity_lemmas, prefix_query)
from codex.search_full_live_shift863 import MASK, live_words, target
from codex.search_hash_transport863 import OPS, operation
from codex.test_full_live_shift863 import frozen_replay


def evaluate(op, a, b, c):
    if op == 'mad':
        return (a*b+c)&MASK
    if op == '<<':
        return 0 if b >= 32 else (a << b)&MASK
    return {'+': lambda: a+b, '-': lambda: a-b, '*': lambda: a*b,
            '^': lambda: a^b, '&': lambda: a&b, '|': lambda: a|b,
            '<': lambda: int(a < b), '==': lambda: int(a == b)}[op]()&MASK


class FullLiveTailAuditTests(unittest.TestCase):
    def test_congruence_class_and_real_two_instruction_programs(self):
        row = congruence_certificate()
        self.assertEqual(list(range(8)), sorted(row['target_low3']))
        self.assertEqual(100, len(row['excluded_sequences']))
        rng = random.Random(877)
        for sequence in itertools.product(COMPATIBLE+SPLITTING, repeat=2):
            for _ in range(30):
                constants = [[rng.getrandbits(32) if rng.randrange(2) else rng.randrange(32)
                              for _ in range(2)] for _ in range(2)]
                roles = [[rng.randrange(10+i) for _ in range(3)] for i in range(2)]
                outputs = []
                for p in row['prior_inputs']:
                    values = live_words(p, 'tail')
                    for op, cs, rr in zip(sequence, constants, roles):
                        pool = values+cs
                        values.append(evaluate(op, *(pool[j] for j in rr)))
                    outputs.append(values[-1]&0xffff)
                self.assertLessEqual(len(set(outputs)), 4, (sequence, roles, constants))

    def test_shift_splitting_bound_all_full32_words(self):
        aa = z3.BitVecs('split_a0 split_a1 split_a2', 32)
        bb = z3.BitVecs('split_b0 split_b1 split_b2', 32)
        low = lambda x: z3.Extract(15, 0, x)
        solver = z3.SolverFor('QF_BV')
        solver.set(timeout=10000)
        for i in (1, 2):
            solver.add(low(aa[i]) == low(aa[0]), low(bb[i]) == low(bb[0]))
        solver.add(z3.Distinct(*(low(a << b) for a, b in zip(aa, bb))))
        self.assertEqual(z3.unsat, solver.check())

    def test_high_identity_exact_conditions_match_original_operations(self):
        self.assertTrue(all(row['result'] == 'unsat' for row in high_identity_lemmas()))
        a, b, out = z3.BitVecs('exact_high_a exact_high_b exact_high_out', 32)
        for op in IDENTITY_HIGH:
            actual, valid = operation(op, a, b, a)
            canonical = z3.And(a == out, z3.UGT(b, a) if op == '%' else b == (0 if op == '>>' else 1))
            solver = z3.SolverFor('QF_BV')
            solver.set(timeout=10000)
            solver.add(z3.UGT(out, 1 << 31), z3.Xor(z3.And(valid, actual == out), canonical))
            self.assertEqual(z3.unsat, solver.check(), op)
        # Independent exhaustive integer arithmetic at small widths.
        for width in range(2, 9):
            top = (1 << width)-1
            for a, b in itertools.product(range(top+1), range(1, top+1)):
                for value in (a//b, (a+b-1)//b, a%b):
                    if value > 1 << (width-1):
                        self.assertEqual(a, value)

    def test_prefix_controls_and_bypassed_output_witnesses(self):
        for op, control in (('mad', 'mad'), ('>>', 'identity'), ('%', 'identity'), ('cdiv', 'identity')):
            row = prefix_query(op, 10000, control)
            self.assertEqual('sat', row['result'], row)
            for p, output in zip(row['inputs'], row['outputs']):
                self.assertEqual(output, frozen_replay(p, [row['program']], 'tail'))
        witness = bypass_witnesses()
        self.assertEqual(8, len(witness['live_sources']))
        for row in witness['live_sources']:
            self.assertEqual(target(row['prior'], 'tail'), row['target'])
            self.assertNotEqual(row['live'], row['target'])
        self.assertNotEqual(*(row['target'] for row in witness['distinct_constant_witnesses']))

    def test_all_high_prefix_queries_and_complete_pair_coverage(self):
        for op in OPS:
            row = prefix_query(op)
            self.assertEqual('unsat', row['result'], row)
        folder = Path(__file__).parent
        report = json.loads((folder/'full_live_tail_closure863_results.json').read_text())
        searches = [json.loads((folder/f'full_live_tail_shift863_{kind}.json').read_text())
                    for kind in ('results', 'refinement')]
        result = coverage(report, searches)
        self.assertEqual((196, 172, 24), (result['total'], result['closed'], result['unresolved']))
        self.assertEqual(196, len({tuple(row['sequence']) for row in result['orders']}))
        # UNKNOWN from a bounded search cannot become an exclusion.
        unknown = dict(scope='tail', width=32, rows=[dict(sequence=['cdiv', 'mad'], result='unknown')])
        with_unknown = coverage(report, [*searches, unknown])
        self.assertEqual(result, with_unknown)
        self.assertFalse(result['new_executable'])


if __name__ == '__main__':
    unittest.main()
