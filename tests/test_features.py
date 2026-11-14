import unittest

import numpy as np
import pandas as pd

from credit_risk.features import build_features


def sample():
    return pd.DataFrame({
        "past_due_30_59": [1, 0, np.nan],
        "past_due_60_89": [0, 2, np.nan],
        "past_due_90": [0, 0, np.nan],
        "monthly_income": [3000.0, 4000.0, np.nan],
        "dependents": [2.0, 0.0, 1.0],
        "debt_ratio": [0.5, 0.2, 0.3],
        "real_estate_loans": [1, 0, 2],
        "open_credit_lines": [3, 0, 1],
        "revolving_utilization": [0.4, 1.3, 0.1],
    })


class FeatureTests(unittest.TestCase):
    def test_total_past_due_sums_columns(self):
        result = build_features(sample(), ["total_past_due"])
        self.assertEqual(result["total_past_due"].iloc[0], 1)
        self.assertEqual(result["total_past_due"].iloc[1], 2)

    def test_total_past_due_all_missing_stays_missing(self):
        result = build_features(sample(), ["total_past_due"])
        self.assertTrue(np.isnan(result["total_past_due"].iloc[2]))

    def test_income_per_person_uses_household_size(self):
        result = build_features(sample(), ["income_per_person"])
        self.assertAlmostEqual(result["income_per_person"].iloc[0], 1000.0)

    def test_over_limit_flag(self):
        result = build_features(sample(), ["over_limit_flag"])
        self.assertEqual(list(result["over_limit_flag"]), [0, 1, 0])

    def test_input_frame_is_not_modified(self):
        frame = sample()
        build_features(frame)
        self.assertNotIn("total_past_due", frame.columns)


if __name__ == "__main__":
    unittest.main()
