import unittest

import numpy as np

from credit_risk.leakage import assert_fit_uses_train_only
from credit_risk.models import build_pipeline
from credit_risk.preprocessing import build_preprocessor
from tests.helpers import make_clean_frame

COLUMNS = ["age", "monthly_income", "past_due_90", "debt_ratio"]


class PreprocessingTests(unittest.TestCase):
    def test_missing_values_are_filled(self):
        frame = make_clean_frame()
        output = build_preprocessor(COLUMNS).fit_transform(frame)
        self.assertFalse(np.isnan(output).any())

    def test_indicator_policy_adds_columns(self):
        frame = make_clean_frame()
        plain = build_preprocessor(COLUMNS, policy="median").fit_transform(frame)
        flagged = build_preprocessor(COLUMNS, policy="median_indicator").fit_transform(frame)
        self.assertGreater(flagged.shape[1], plain.shape[1])

    def test_native_policy_keeps_missing(self):
        frame = make_clean_frame()
        output = build_preprocessor(COLUMNS, policy="native", scale=False).fit_transform(frame)
        self.assertTrue(np.isnan(output).any())

    def test_fit_uses_training_part_only(self):
        frame = make_clean_frame()
        train, valid = frame.iloc[:400], frame.iloc[400:]
        pipeline = build_pipeline("logistic", COLUMNS).fit(train[COLUMNS], train["target"])
        assert_fit_uses_train_only(pipeline, train, COLUMNS)
        with self.assertRaises(AssertionError):
            assert_fit_uses_train_only(pipeline, valid, COLUMNS)

    def test_unknown_policy_raises(self):
        with self.assertRaises(ValueError):
            build_preprocessor(COLUMNS, policy="other")


if __name__ == "__main__":
    unittest.main()
