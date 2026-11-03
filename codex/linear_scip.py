"""SCIP cross-check for the linear subset of the prefix CP-SAT model.

Reject unsupported constraints instead of silently dropping them. Validate
the rounded assignment against every original integer row before returning it.
"""
from time import monotonic

from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model


class LinearScipSolver:
    def __init__(self, seconds, workers=1):
        self.seconds, self.workers = seconds, workers
        self.assignment = None
        self.best_objective_bound = None

    def solve(self, model):
        start = monotonic()
        self.assignment = None
        self.best_objective_bound = None
        proto = model.proto
        solver = pywraplp.Solver.CreateSolver('SCIP')
        assert solver is not None
        solver.SetTimeLimit(round(self.seconds * 1000))
        solver.SetNumThreads(self.workers)
        variables = []
        domains = []
        for var in proto.variables:
            assert len(var.domain) == 2, 'Disjoint domains are unsupported'
            low, high = var.domain
            domains.append((low, high))
            variables.append(solver.IntVar(low, high, var.name))
        for row in proto.constraints:
            assert row.has_linear(), 'Only linear constraints are supported'
            linear = row.linear
            assert len(linear.domain) == 2, 'Disjoint row domains are unsupported'
            expr = solver.Sum(c * variables[i] for i, c in zip(linear.vars, linear.coeffs))
            low, high = linear.domain
            # The sum is zero exactly when all enforcement literals are true.
            inactive = solver.Sum(1 - variables[i] if i >= 0 else variables[-i - 1]
                                  for i in row.enforcement_literal)
            minimum = sum(c * domains[i][0 if c >= 0 else 1]
                          for i, c in zip(linear.vars, linear.coeffs))
            maximum = sum(c * domains[i][1 if c >= 0 else 0]
                          for i, c in zip(linear.vars, linear.coeffs))
            if low > -(1 << 63):
                solver.Add(expr + max(0, low - minimum) * inactive >= low)
            if high < (1 << 63) - 1:
                solver.Add(expr - max(0, maximum - high) * inactive <= high)
        objective = proto.objective
        solver.Minimize(solver.Sum(c * variables[i]
                                  for i, c in zip(objective.vars, objective.coeffs))
                        + objective.offset)
        hint = proto.solution_hint
        solver.SetHint([variables[i] for i in hint.vars], list(hint.values))
        status = solver.Solve()
        self.wall_time = monotonic() - start
        mapped = {solver.OPTIMAL: cp_model.OPTIMAL, solver.FEASIBLE: cp_model.FEASIBLE,
                  solver.INFEASIBLE: cp_model.INFEASIBLE,
                  solver.NOT_SOLVED: cp_model.UNKNOWN}
        result = mapped.get(status, cp_model.UNKNOWN)
        if result in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            self.assignment = [round(var.solution_value()) for var in variables]
            values = self.assignment
            assert all(low <= value <= high
                       for value, (low, high) in zip(values, domains))
            for row in proto.constraints:
                if not all(values[i] if i >= 0 else 1 - values[-i - 1]
                           for i in row.enforcement_literal):
                    continue
                total = sum(c * values[i] for i, c in zip(row.linear.vars, row.linear.coeffs))
                assert row.linear.domain[0] <= total <= row.linear.domain[1]
        if objective.vars:
            self.best_objective_bound = solver.Objective().BestBound() * objective.scaling_factor
        return result

    @staticmethod
    def status_name(status):
        return {cp_model.OPTIMAL: 'OPTIMAL', cp_model.FEASIBLE: 'FEASIBLE',
                cp_model.INFEASIBLE: 'INFEASIBLE', cp_model.UNKNOWN: 'UNKNOWN'}[status]

    def value(self, expression):
        assert self.assignment is not None
        flat = cp_model.FlatIntExpr(expression)
        return flat.offset + sum(c * self.assignment[v.index]
                                 for v, c in zip(flat.vars, flat.coeffs))
