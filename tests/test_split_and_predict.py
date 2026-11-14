import os
import tempfile
import unittest

from credit_risk.features import build_features
from credit_risk.models import build_pipeline
from credit_risk.predict import predict_frame
from credit_risk.settings import BASE_FEATURES, NEW_FEATURES
from credit_risk.splitting import HoldoutVault, split_holdout
from tests.helpers import make_clean_frame


class SplitTests(unittest.TestCase):
    def test_split_keeps_class_ratio_and_disjoint_ids(self):
        frame = make_clean_frame()
        dev, holdout = split_holdout(frame, 0.2, seed=1)
        self.assertEqual(len(dev) + len(holdout), len(frame))
        self.assertFalse(set(dev["id"]) & set(holdout["id"]))
        self.assertAlmostEqual(dev["target"].mean(), holdout["target"].mean(), delta=0.03)

    def test_vault_opens_only_once(self):
        frame = make_clean_frame(100)
        with tempfile.TemporaryDirectory() as folder:
            vault = HoldoutVault(frame, os.path.join(folder, "log.txt"))
            vault.open("test")
            with self.assertRaises(RuntimeError):
                vault.open("again")


class PredictTests(unittest.TestCase):
    def test_predict_frame_returns_ids_probabilities_decisions(self):
        frame = make_clean_frame()
        columns = BASE_FEATURES + NEW_FEATURES
        features = build_features(frame)
        pipeline = build_pipeline("boosting", columns, policy="native")
        pipeline.fit(features[columns], frame["target"])
        artifact = {"pipeline": pipeline, "threshold": 0.1,
                    "feature_columns": columns, "extra_features": NEW_FEATURES}
        result = predict_frame(frame.head(5), artifact)
        self.assertEqual(list(result.columns), ["id", "probability", "decision"])
        self.assertEqual(len(result), 5)
        self.assertTrue(result["probability"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
