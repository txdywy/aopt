"""Frozen-ISA, full32 and constructive controls for joint-program search."""
import json
from pathlib import Path
import random
import sys
import unittest

import z3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tests'))
from frozen_problem import Machine, myhash

from codex.search_full_live_shift863 import MASK, samples, target
from codex.search_joint_tail863 import (OPS, PREFIX_COST, constant_values, execute,
    expression, factor_values, initial_words, original_program, search, sequences)


def frozen_execute(prior, program, stage, profile):
    machine = Machine([], [], None)
    core = machine.cores[0]
    values = initial_words(prior, stage)
    for row in program:
        pool = values+list(constant_values(profile))
        a = pool[row['a']]
        if row['op'] == 'mad':
            b, c = row['factor'], pool[row['c']]
        elif row['op'] == '>>':
            b, c = row.get('amount', 16), 0
        else:
            b, c = pool[row['b']], 0
        core.scratch[:24] = [a]*8+[b]*8+[c]*8
        machine.scratch_write = {}
        if row['op'] == 'mad':
            machine.valu(core, 'multiply_add', 24, 0, 8, 16)
        else:
            machine.alu(core, row['op'], 24, 0, 8)
        values.append(machine.scratch_write[24])
    return values[-1]


class JointTailTests(unittest.TestCase):
    def test_original_program_and_preserved_prefix_match_frozen_full_hash(self):
        for stage in PREFIX_COST:
            for profile in ('base', 'extended'):
                program = original_program(stage, profile)
                self.assertEqual(6, PREFIX_COST[stage]+len(program))
                for p in samples():
                    expected = myhash(p) ^ 0xb55a4f09
                    self.assertEqual(expected, execute(p, program, stage, profile))
                    self.assertEqual(expected, frozen_execute(p, program, stage, profile))
                x = z3.BitVec('joint_original_input', 32)
                solver = z3.SolverFor('QF_BV')
                solver.set(timeout=5000)
                solver.add(expression(x, program, stage, profile) != target(x, 'tail'))
                self.assertEqual(z3.unsat, solver.check(), (stage, profile))

    def test_random_programs_agree_across_three_independent_evaluators(self):
        rng = random.Random(880)
        for stage in PREFIX_COST:
            for profile in ('base', 'extended'):
                for _ in range(30):
                    program = []
                    for i in range(rng.randrange(1, 7)):
                        limit = len(initial_words(0, stage))+i+len(constant_values(profile))
                        op = rng.choice(OPS)
                        row = dict(op=op, a=rng.randrange(limit))
                        if op == 'mad':
                            row.update(c=rng.randrange(limit), factor=rng.choice(factor_values(profile)))
                        elif op == '>>':
                            row['amount'] = rng.choice((0, 7, 9, 12, 16, 19, 31, 32))
                        else:
                            row['b'] = rng.randrange(limit)
                        program.append(row)
                    for p in (0, MASK, rng.getrandbits(32)):
                        actual = frozen_execute(p, program, stage, profile)
                        self.assertEqual(actual, execute(p, program, stage, profile))
                        bv = expression(z3.BitVecVal(p, 32), program, stage, profile)
                        self.assertEqual(actual, z3.simplify(bv).as_long())

    def test_automatic_controls_recover_valid_full32_programs(self):
        simple = search(('mad', '^'), 10, 9000, control=True)
        self.assertEqual('proved_equivalent_full32', simple['result'], simple)
        for p in samples():
            b = initial_words(p)[4]
            expected = ((33*b+0xe9f8cc1d)&MASK) ^ b
            self.assertEqual(expected, frozen_execute(p, simple['candidate'], 'first', 'base'))
        known = original_program('branches')
        found = search([row['op'] for row in known], 20, 15000, stage='branches')
        self.assertEqual('proved_equivalent_full32', found['result'], found)
        self.assertEqual(6, found['total_instructions'])
        self.assertFalse(found['new_executable'])
        for p in samples():
            self.assertEqual(target(p, 'tail'), frozen_execute(p, found['candidate'], 'branches', 'base'))

    def test_sequence_costs_and_extended_profile_containment(self):
        self.assertEqual([50, 32, 32, 12], [len(sequences(stage)) for stage in PREFIX_COST])
        for stage in PREFIX_COST:
            for order in sequences(stage):
                self.assertEqual(5, PREFIX_COST[stage]+len(order))
                self.assertEqual(1, order.count('>>'))
                self.assertTrue(set(order) <= {'mad', '^', '>>'})
        self.assertTrue(set(factor_values()) < set(factor_values('extended')))
        self.assertEqual(constant_values(), constant_values('extended')[:len(constant_values())])
        self.assertIn(297, factor_values('extended'))
        self.assertIn(152064, factor_values('extended'))

    def test_history_does_not_promote_unknown_or_controls_to_a_score(self):
        folder = Path(__file__).parent
        r = json.loads((folder/'joint_middle_tail863_results.json').read_text())
        self.assertEqual(50, len(r['rows']))
        self.assertTrue(any(row['result'] == 'unknown' for row in r['rows']))
        self.assertTrue(all(row['result'] == 'proved_equivalent_full32' for row in r['controls']))
        self.assertFalse(r['new_executable'])
        full = json.loads((folder/'joint_middle_tail863_unpinned_control.json').read_text())
        self.assertEqual('unknown', full['result'])
        self.assertFalse(full['new_executable'])

    def test_search_guards(self):
        for kwargs in ({'stage': 'bad'}, {'profile': 'bad'}, {'seconds': 0}, {'check_ms': 0},
                       {'shift_amount': -1}, {'shift_amount': 33}, {'pin_original': True}):
            with self.assertRaises(ValueError):
                search(('mad', '^'), **kwargs)
        with self.assertRaises(ValueError):
            search(('<<',))


if __name__ == '__main__':
    unittest.main()
