import unittest
from codex.schedule_tail_window import schedule_tail


class MovementBoundsTests(unittest.TestCase):
    def test_overlapping_stores_keep_order_when_coissued(self):
        program = [{'store': [('store', 0, 1)]}, {'store': [('store', 0, 2)]}]
        result, report = schedule_tail(program, start=0, stop=2, horizon=1,
                                       allow_memory=True)
        self.assertEqual(report['status'], 'sat')
        self.assertEqual(result[0]['store'], [('store', 0, 1), ('store', 0, 2)])

    def test_store_to_load_still_requires_next_cycle(self):
        result, report = schedule_tail(
            [{'store': [('store', 0, 1)]}, {'load': [('load', 2, 0)]}],
            start=0, stop=2, horizon=1, allow_memory=True)
        self.assertIsNone(result)
        self.assertEqual(report['status'], 'unsat')

    def test_compress_one_empty_packet(self):
        program = [{'alu': [('+', 0, 1, 2)]}, {},
                   {'alu': [('^', 3, 0, 4)]}]
        result, report = schedule_tail(program, start=0, stop=3, horizon=2,
                                       advance_only=1)
        self.assertEqual(report['status'], 'sat')
        self.assertEqual(result[0]['alu'], [('+', 0, 1, 2)])
        self.assertEqual(result[1]['alu'], [('^', 3, 0, 4)])
        self.assertEqual(result[1]['flow'], [('halt',)])
        with self.assertRaises(ValueError):
            schedule_tail(program, start=0, stop=3, horizon=2,
                          advance_only=0, certificate=report)

    def test_no_movement_cannot_compress(self):
        result, report = schedule_tail([{}, {}, {'alu': [('+', 0, 1, 2)]}],
                                       start=0, stop=3, horizon=2, advance_only=0)
        self.assertIsNone(result)
        self.assertEqual(report['status'], 'unsat')


if __name__ == '__main__':
    unittest.main()
