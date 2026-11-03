import unittest
from codex.audit_compact_liveness import audit


class LivenessTests(unittest.TestCase):
    def test_store_keeps_value(self):
        dead, _ = audit([{'alu': [('+', 0, 1, 2)]},
                         {'store': [('store', 3, 0)], 'flow': [('halt',)]}])
        self.assertEqual(dead, [])

    def test_same_packet_store_reads_old_value(self):
        dead, _ = audit([{'alu': [('+', 0, 1, 2)],
                          'store': [('store', 3, 0)], 'flow': [('halt',)]}])
        self.assertEqual(len(dead), 1)

    def test_indirect_preserves_any_possible_reader(self):
        dead, _ = audit([{'alu': [('+', 0, 1, 2)]},
                         {'flow': [('jump_indirect', 4)]},
                         {'store': [('store', 3, 0)], 'flow': [('halt',)]}])
        self.assertEqual(dead, [])


if __name__ == '__main__':
    unittest.main()
