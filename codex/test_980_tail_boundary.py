"""Scope checks for the 980 export and its fixed terminal dependency floor."""
import base64
import json
import unittest
import zlib
from codex.perf_takehome_980 import _CODE as NEW
from codex.perf_takehome_981 import _CODE as OLD
from codex.schedule_tail_window import schedule_tail


def decode(code):
    prefix, recipes = json.loads(zlib.decompress(base64.b85decode(code)))
    return [{e: [tuple(s) for s in slots] for e, slots in p.items()}
            for p in prefix], recipes


class TailBoundaryTests(unittest.TestCase):
    def test_large_tables_and_earlier_main_stream_unchanged(self):
        old, old_recipes = decode(OLD)
        new, new_recipes = decode(NEW)
        self.assertEqual(old_recipes, new_recipes)
        self.assertEqual(old[:969], new[:969])
        self.assertEqual(len(old), len(new))
        modified_out_of_line = [i for i in range(981, len(old)) if old[i] != new[i]]
        self.assertEqual(len(modified_out_of_line), 64)
        for i in modified_out_of_line:
            self.assertEqual(old[i]['flow'], [('jump',970)])
            self.assertEqual(new[i]['flow'], [('jump',970)])

    def test_nine_cycle_fixed_tail_is_infeasible(self):
        prefix, _ = decode(NEW)
        candidate, report = schedule_tail(prefix, start=970, stop=980,
                                          horizon=9, allow_data_flow=True,
                                          timeout_ms=1000)
        self.assertIsNone(candidate)
        self.assertEqual(report['status'], 'unsat')
        self.assertEqual(report['dependency_floor'], 10)
        self.assertEqual(report['critical_chain'][0]['slot'][0], 'vselect')
        self.assertEqual(report['critical_chain'][-1]['slot'], ('vstore',8,679))


if __name__ == '__main__':
    unittest.main()
