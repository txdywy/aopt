"""Frozen-ISA and full32 reference controls for the live-value search."""
import random
from pathlib import Path
import sys
import unittest

import z3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tests'))
from frozen_problem import HASH_STAGES, Machine, myhash

from codex.search_full_live_shift863 import (MASK, inverse_target, live_words, output_probes,
    run_query, samples, search, sequences, target)


def frozen_replay(prior, rows, scope):
    """Execute the candidate's operations using the untouched ISA implementation."""
    machine = Machine([], [], None)
    core = machine.cores[0]
    values = live_words(prior, scope)
    for row in rows:
        pool = values+row['constants']
        a, b, c = [pool[i] for i in row['operands']]
        core.scratch[:24] = [a]*8+[b]*8+[c]*8
        machine.scratch_write = {}
        if row['op'] == 'mad':
            machine.valu(core, 'multiply_add', 24, 0, 8, 16)
        else:
            machine.alu(core, row['op'], 24, 0, 8)
        values.append(machine.scratch_write[24])
    return values[-1]


class FullLiveShiftTests(unittest.TestCase):
    def test_output_preimages_are_exact_and_keep_full32_values(self):
        rng = random.Random(876)
        for scope in ('middle', 'tail'):
            ys = output_probes()+[rng.getrandbits(32) for _ in range(1000)]
            for y in ys:
                self.assertEqual(y, target(inverse_target(y, scope), scope))
            for p in samples():
                self.assertEqual(p, inverse_target(target(p, scope), scope))
            self.assertTrue(set(inverse_target(y, scope) for y in output_probes())
                            <= set(samples(scope, True)))
        with self.assertRaises(ValueError):
            inverse_target(-1)
        with self.assertRaises(ValueError):
            inverse_target(0, 'bad')

    def test_targets_match_frozen_hash_on_full32_inputs(self):
        rng = random.Random(873)
        for p in samples()+[rng.getrandbits(32) for _ in range(200)]:
            self.assertEqual(myhash(p), target(p, 'tail') ^ 0xb55a4f09)
            # Replay the first five original hash stages separately.
            word = p
            ops = {'+': lambda x, y: x+y, '^': lambda x, y: x^y,
                   '<<': lambda x, y: x << y, '>>': lambda x, y: x >> y}
            for op1, k1, op2, op3, k3 in HASH_STAGES[:5]:
                word = ops[op2](ops[op1](word, k1)&MASK, ops[op3](word, k3)&MASK)&MASK
            self.assertEqual(word, target(p, 'middle'))

    def test_symbolic_words_preserve_every_full32_intermediate(self):
        rng = random.Random(874)
        for p in samples()+[rng.getrandbits(32) for _ in range(30)]:
            for scope in ('middle', 'tail'):
                expected = live_words(p, scope)
                symbolic = live_words(z3.BitVecVal(p, 32), scope)
                self.assertEqual(expected, [z3.simplify(v).as_long() for v in symbolic])
                self.assertEqual(target(p, scope), z3.simplify(target(z3.BitVecVal(p, 32), scope)).as_long())
        for value in (-1, MASK+1, z3.BitVec('bad_live_width', 11)):
            with self.assertRaises(ValueError):
                live_words(value)
        with self.assertRaises(ValueError):
            live_words(0, 'invalid')
        actual = {live_words(p)[1] for p in samples()}
        self.assertTrue(set(range(16)) <= actual)
        self.assertIn(1 << 19, actual)
        # The preimage p can have high bits even when a is small. Those high
        # bits are genuinely available to a new SHR and cannot be truncated.
        p = ((1-0x7ed55d16)*pow(4097, -1, 1 << 32))&MASK
        self.assertEqual(1, live_words(p)[1])
        self.assertNotEqual((p >> 12)&2047, (p&2047) >> 12)

    def test_discovered_positive_control_replays_on_frozen_machine(self):
        row = search(('>>', 'mad'), timeout_ms=10000, fixed_shift=7, first_source=0, control=True)
        self.assertEqual('proved_equivalent_full32', row['result'], row)
        self.assertEqual('unsat', row['events'][-1]['verification'])
        rng = random.Random(875)
        for p in samples()+[rng.getrandbits(32) for _ in range(100)]:
            expected = ((p >> 7)*33+live_words(p)[4])&MASK
            self.assertEqual(expected, frozen_replay(p, row['candidate'], 'middle'))

    def test_negative_leading_opcode_cli_and_grammar_counts(self):
        row = run_query('tail', ('-', '>>'), 3000)
        self.assertEqual('grammar_excluded', row['result'], row)
        self.assertIn('--sequence=-,>>', row['command'])
        self.assertEqual((48, 27), (len(sequences('middle')), len(sequences('tail'))))
        self.assertTrue(all('>>' in seq for seq in sequences('middle')))
        self.assertTrue(all('mad' in seq for seq in sequences('middle')))

    def test_search_domain_guards(self):
        for kwargs in ({'scope': 'bad'}, {'timeout_ms': 0}, {'seconds': 0},
                       {'fixed_shift': 33}, {'first_source': 5}, {'control': True}):
            with self.assertRaises(ValueError):
                search(('>>', '^'), **kwargs)
        with self.assertRaises(ValueError):
            search(('mad', '>>'), first_source=0)


if __name__ == '__main__':
    unittest.main()
