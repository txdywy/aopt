"""Frozen-ISA, full32 and constructive controls for joint-program search."""
import json
from pathlib import Path
import random
import sys
import unittest

import z3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tests'))
from frozen_problem import Machine, myhash

from codex.search_full_live_shift863 import MASK, prior_for_b, samples, target
from codex.search_joint_tail863 import (DIRECT_STAGES, OPS, PREFIX_COST,
    constant_values, execute, expression, factor_values, initial_words,
    original_program, reference_value, search, search_samples, sequences)
from codex.audit_joint_affine863 import certificate


def frozen_execute(prior, program, stage, profile):
    machine = Machine([], [], None)
    core = machine.cores[0]
    values = initial_words(prior, stage)
    for row in program:
        pool = values+list(constant_values(profile))+row.get('extra_constants', [])
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
            for profile in ('sparse', 'base', 'extended'):
                program = original_program(stage, profile)
                self.assertEqual(6, PREFIX_COST[stage]+len(program))
                for p in samples():
                    actual_prior = prior_for_b(p) if stage in DIRECT_STAGES else p
                    expected = myhash(actual_prior) ^ 0xb55a4f09
                    self.assertEqual(expected, reference_value(p, stage))
                    self.assertEqual(expected, execute(p, program, stage, profile))
                    self.assertEqual(expected, frozen_execute(p, program, stage, profile))
                x = z3.BitVec('joint_original_input', 32)
                solver = z3.SolverFor('QF_BV')
                solver.set(timeout=5000)
                solver.add(expression(x, program, stage, profile) != reference_value(x, stage))
                self.assertEqual(z3.unsat, solver.check(), (stage, profile))

    def test_random_programs_agree_across_three_independent_evaluators(self):
        rng = random.Random(880)
        for stage in PREFIX_COST:
            for profile in ('sparse', 'base', 'extended'):
                for _ in range(30):
                    program = []
                    extra = [rng.getrandbits(32) for _ in range(rng.randrange(4))]
                    for i in range(rng.randrange(1, 7)):
                        limit = len(initial_words(0, stage))+i+len(constant_values(profile))+len(extra)
                        op = rng.choice(OPS)
                        row = dict(op=op, a=rng.randrange(limit), extra_constants=extra)
                        if op == 'mad':
                            row.update(c=rng.randrange(limit), factor=rng.getrandbits(32))
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
        self.assertEqual([50, 32, 32, 12]*2, [len(sequences(stage)) for stage in PREFIX_COST])
        for stage in PREFIX_COST:
            for order in sequences(stage):
                self.assertEqual(5, PREFIX_COST[stage]+len(order))
                self.assertEqual(1, order.count('>>'))
                self.assertTrue(set(order) <= {'mad', '^', '>>'})
        self.assertTrue(set(factor_values()) < set(factor_values('extended')))
        self.assertEqual(constant_values(), constant_values('extended')[:len(constant_values())])
        self.assertIn(297, factor_values('extended'))
        self.assertIn(152064, factor_values('extended'))

    def test_direct_b_modes_use_the_full_b_domain(self):
        for stage in DIRECT_STAGES:
            self.assertTrue({0, 1, MASK, 1 << 31} <= set(search_samples(stage)))
            self.assertEqual(0x80000000, initial_words(0x80000000, stage)[0])
            for value in (-1, 1 << 32, z3.BitVec('bad_b', 31)):
                with self.assertRaises(ValueError):
                    initial_words(value, stage)

    def test_arbitrary_constants_and_factors_retain_full32_semantics(self):
        program = original_program('b', 'sparse')
        extra = [0xe9f8cc1d, 0xaccf6200, 0xfd7046c5]
        rank = 0
        for i, row in enumerate(program):
            row['extra_constants'] = extra
            if row['op'] == 'mad':
                row['c'] = 1+i+len(constant_values('sparse'))+rank
                rank += 1
        for b in search_samples('b'):
            self.assertEqual(reference_value(b, 'b'), frozen_execute(b, program, 'b', 'sparse'))
            self.assertEqual(reference_value(b, 'b'), execute(b, program, 'b', 'sparse'))
        control = search([r['op'] for r in original_program('bbranches', 'sparse')],
            10, 9000, stage='bbranches', profile='sparse', pin_original=True,
            free_constants=3, free_factors=True)
        self.assertEqual('proved_equivalent_full32', control['result'], control)
        self.assertEqual(3, control['free_constant_count'])
        self.assertTrue(control['arbitrary_mad_factors'])

    def test_affine_fold_certificate_uses_the_frozen_full32_target(self):
        proof = certificate()
        actual = [myhash(prior_for_b(b)) ^ 0xb55a4f09 for b in proof['inputs']]
        self.assertEqual(actual, proof['outputs'])
        self.assertEqual(17692, (actual[2]-2*actual[1]+actual[0]) & 65535)

    def test_affine_fold_property_holds_for_random_frozen_programs(self):
        rng = random.Random(881)
        for _ in range(300):
            program = []
            length = rng.randrange(1, 8)
            shifted = rng.randrange(length)
            extra = [rng.getrandbits(32) for _ in range(3)]
            for i in range(length):
                limit = 3+i+len(constant_values('sparse'))+len(extra)
                row = dict(op='>>' if i == shifted else 'mad',
                    a=rng.randrange(limit), extra_constants=extra)
                if i == shifted:
                    row['amount'] = 16
                else:
                    row.update(c=rng.randrange(limit), factor=rng.getrandbits(32))
                program.append(row)
            start = rng.getrandbits(32)
            values = [frozen_execute((start+i*65536)&MASK, program, 'bbranches', 'sparse')
                      for i in range(3)]
            self.assertEqual(0, (values[2]-2*values[1]+values[0]) & 65535)

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
                       {'shift_amount': -1}, {'shift_amount': 33}, {'pin_original': True},
                       {'free_constants': -1}, {'free_constants': 4}):
            with self.assertRaises(ValueError):
                search(('mad', '^'), **kwargs)
        with self.assertRaises(ValueError):
            search(('<<',))


if __name__ == '__main__':
    unittest.main()
