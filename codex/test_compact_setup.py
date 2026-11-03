"""Regressions for alias precision and the startup transformation."""

import unittest

from codex.reoptimize_compact import extract_precise, memory_locations


class MemoryDependenceTests(unittest.TestCase):
    def test_disjoint_store_and_load_may_reorder(self):
        program = [
            {"load": [("const", 0, 100), ("const", 1, 200)]},
            {"store": [("vstore", 0, 8)]},
            {"load": [("load", 16, 1)]},
        ]
        ops, edges, _, _ = extract_precise(program, 0, 3)
        store = next(i for i, op in enumerate(ops) if op[0] == "store")
        load = next(i for i, op in enumerate(ops) if op[1][0] == "load")
        self.assertNotIn((store, load), edges)

    def test_overlapping_store_to_load_needs_one_cycle(self):
        program = [
            {"load": [("const", 0, 100), ("const", 1, 107)]},
            {"store": [("vstore", 0, 8)]},
            {"load": [("load", 16, 1)]},
        ]
        ops, edges, _, _ = extract_precise(program, 0, 3)
        store = next(i for i, op in enumerate(ops) if op[0] == "store")
        load = next(i for i, op in enumerate(ops) if op[1][0] == "load")
        self.assertEqual(edges[store, load], 1)

    def test_runtime_pointer_still_aliases(self):
        program = [
            {"load": [("const", 0, 100)]},
            {"load": [("load", 1, 0)]},
            {"store": [("vstore", 0, 8)]},
            {"load": [("load", 16, 1)]},
        ]
        ops, edges, _, _ = extract_precise(program, 0, 4)
        store = next(i for i, op in enumerate(ops) if op[0] == "store")
        load = next(i for i, op in enumerate(ops) if op[1] == ("load", 16, 1))
        self.assertEqual(edges[store, load], 1)

    def test_pointer_write_is_visible_after_packet(self):
        program = [
            {"load": [("const", 0, 100)]},
            {"flow": [("add_imm", 0, 0, 8)], "load": [("vload", 8, 0)]},
        ]
        locations = memory_locations(program, 2)
        self.assertEqual(locations[1, "load", 0], frozenset(range(100, 108)))


if __name__ == "__main__":
    unittest.main()
