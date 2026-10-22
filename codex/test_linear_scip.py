"""Cross-check the optional SCIP translation against CP-SAT on integer models."""
import unittest

from ortools.sat.python import cp_model

from codex.linear_scip import LinearScipSolver


class LinearScipTest(unittest.TestCase):
    def test_conditional_rows_and_objective_offset(self):
        for flag in (0, 1):
            for maximize in (False, True):
                with self.subTest(flag=flag, maximize=maximize):
                    model = cp_model.CpModel()
                    x = model.new_int_var(-4, 7, 'x')
                    y = model.new_bool_var('y')
                    z = model.new_bool_var('z')
                    model.add(y == flag)
                    model.add(z == 0)
                    model.add(x <= 2).only_enforce_if([y, z.Not()])
                    model.add(x >= -1).only_enforce_if(y.Not())
                    expression = 3 * x + 4
                    (model.maximize if maximize else model.minimize)(expression)
                    reference, candidate = cp_model.CpSolver(), LinearScipSolver(10)
                    self.assertEqual(reference.solve(model), cp_model.OPTIMAL)
                    self.assertEqual(candidate.solve(model), cp_model.OPTIMAL)
                    self.assertEqual(reference.value(x), candidate.value(x))
                    self.assertEqual(reference.value(expression), candidate.value(expression))
                    self.assertAlmostEqual(reference.objective_value, candidate.best_objective_bound)

    def test_integer_infeasibility(self):
        model = cp_model.CpModel()
        x = model.new_int_var(0, 2, 'x')
        model.add(2 * x == 3)
        self.assertEqual(LinearScipSolver(10).solve(model), cp_model.INFEASIBLE)

    def test_unsupported_constraint_rejected(self):
        model = cp_model.CpModel()
        a, b = model.new_bool_var('a'), model.new_bool_var('b')
        model.add_bool_or([a, b])
        with self.assertRaisesRegex(AssertionError, 'Only linear'):
            LinearScipSolver(10).solve(model)


if __name__ == '__main__':
    unittest.main()
