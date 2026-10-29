"""Focused selection controls and concrete first-MAD word-cut regression."""
import unittest

from codex.search_focused_joint_boundary863 import edge_points, key, manifest, shift_has_mad_ancestor
from codex.search_joint_live_boundary863 import C0, C4, MASK, NODE, NAMES, inputs, parity_models, solve, wanted
from codex.search_live_boundary863 import expression


class FocusedJointBoundaryTests(unittest.TestCase):
    def test_target_inverse_samples_keep_positive_program(self):
        points = edge_points(0)
        self.assertEqual(len(set(points)), len(points))
        self.assertTrue(all(0 <= b <= MASK for b in points))
        image = {wanted(inputs(b), 0) for b in points}
        self.assertTrue(set(range(32)) <= image)
        self.assertTrue({1 << k for k in range(32)} <= image)
        self.assertTrue({MASK ^ (1 << k) for k in range(32)} <= image)
        row = dict(order=['mad', 'shr16', '^', 'mad'],
                   wiring=[[0, 8, 9], [8], [8, 9], [10, 11, 12]])
        pins = [(0, 0, 9), (0, 1, C4), (3, 0, 4097), (3, 1, C0)]
        self.assertEqual(solve(row, 0, 5, pins, True, points)['status'], 'proved_equivalent')

    def test_selection_is_live_and_deduplicated(self):
        rows, orders = manifest(1, 8642931)
        self.assertEqual(len(orders), 44)
        self.assertEqual(len(rows), 44)
        selected = {key(row) for row in rows}
        rerun, _ = manifest(1, 8642931, selected)
        self.assertTrue(selected.isdisjoint(key(row) for row in rerun))
        n = len(NAMES)
        for row in rows+rerun:
            self.assertTrue(shift_has_mad_ancestor(row))
            self.assertTrue(parity_models(row))
            self.assertTrue(all(any(n+j in row['wiring'][i] for i in range(j+1, 4)) for j in range(3)))

    def test_shift_ancestry_controls(self):
        natural = dict(order=['mad', 'shr16', '^', 'mad'],
                       wiring=[[0, 8, 9], [8], [8, 9], [10, 11, 12]])
        self.assertTrue(shift_has_mad_ancestor(natural))
        initial = dict(natural, wiring=[[0, 8, 9], [0], [8, 9], [10, 11, 12]])
        self.assertFalse(shift_has_mad_ancestor(initial))
        bit = dict(order=['^', 'mad', 'shr16', 'mad'],
                   wiring=[[0, 8], [8, 9, 5], [8], [9, 10, 11]])
        self.assertFalse(shift_has_mad_ancestor(bit))
        bit['wiring'][2] = [9]
        self.assertTrue(shift_has_mad_ancestor(bit))

    def test_constant_free_mad_wordcut_is_typed(self):
        row = dict(order=['mad', 'shr16', '^', 'mad'],
                   wiring=[[0, 0, 0], [8], [8, 9], [10, 11, 12]])
        first = dict(order=['mad'], wiring=[row['wiring'][0]])
        images = [expression(first, [[0, 0]], inputs(b)) & 3 for b in range(4)]
        self.assertTrue(all(isinstance(v, int) for v in images))
        self.assertLess(len(set(images)), 4)
        self.assertEqual(solve(row, NODE, 2)['status'], 'unsat')
        # A concrete first-MAD permutation modulo4 exercises the other branch.
        row['wiring'][0] = [1, 2, 3]
        first['wiring'] = [row['wiring'][0]]
        self.assertEqual(len({expression(first, [[0, 0]], inputs(b)) & 3 for b in range(4)}), 4)
        self.assertIn(solve(row, NODE, .1)['status'], ('unsat', 'unknown'))


if __name__ == '__main__':
    unittest.main()
