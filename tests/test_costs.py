import unittest

import numpy as np

from credit_risk.costs import (best_threshold, cost_per_1000, theoretical_threshold,
                               total_cost)


class CostTests(unittest.TestCase):
    def test_total_cost_uses_matrix(self):
        y_true = [0, 1, 1, 0]
        y_pred = [1, 0, 1, 0]
        self.assertEqual(total_cost(y_true, y_pred, 1.0, 12.0), 13.0)

    def test_cost_per_1000_scales_by_rows(self):
        y_true = [1, 0, 0, 0]
        y_pred = [0, 0, 0, 0]
        self.assertEqual(cost_per_1000(y_true, y_pred, 1.0, 12.0), 3000.0)

    def test_perfect_decisions_cost_zero(self):
        y_true = np.array([0, 1, 0, 1])
        self.assertEqual(total_cost(y_true, y_true), 0.0)

    def test_theoretical_threshold(self):
        self.assertAlmostEqual(theoretical_threshold(1.0, 12.0), 1 / 13)

    def test_best_threshold_finds_separating_point(self):
        y_true = np.array([0, 0, 0, 1, 1])
        proba = np.array([0.05, 0.1, 0.15, 0.5, 0.6])
        threshold, cost = best_threshold(y_true, proba, np.array([0.2, 0.3, 0.9]))
        self.assertEqual(cost, 0.0)
        self.assertIn(threshold, (0.2, 0.3))


if __name__ == "__main__":
    unittest.main()
