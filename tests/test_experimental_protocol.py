"""Regressões para dobras independentes e contratos de IDs do TP1."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExperimentalProtocolTests(unittest.TestCase):
    def test_svr_search_fits_scaler_only_inside_each_training_fold(self):
        module = load("03_tune_svr")
        sizes = []

        class ObservedScaler(StandardScaler):
            def fit(self, X, y=None, sample_weight=None):
                sizes.append(len(X))
                return super().fit(X, y, sample_weight=sample_weight)

        def search(estimator, params, **kwargs):
            self.assertIsInstance(estimator, Pipeline)
            kwargs["n_jobs"] = 1
            return RandomizedSearchCV(estimator, params, **kwargs)

        rng = np.random.default_rng(42)
        X = rng.normal(size=(30, 3))
        y = rng.uniform(1, 228, 30)
        with tempfile.TemporaryDirectory() as folder, patch.object(
            module, "get_features", return_value=(X, y, X[:6], y[:6])
        ), patch.object(module, "StandardScaler", ObservedScaler), patch.object(
            module, "RandomizedSearchCV", search
        ), patch.object(module, "N_ITER_SEARCH", 1):
            module.main("unused", "unused.csv", "unused", folder)
        self.assertEqual(sizes[:3], [20, 20, 20])
        self.assertEqual(sizes[-1], 30)

    def test_texture_loader_rejects_ids_that_do_not_match_frozen_split(self):
        module = load("07_compare_all_descriptors")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            splits = root / "splits"
            splits.mkdir()
            pd.DataFrame({"id": [1, 2]}).to_csv(splits / "train_ids.csv", index=False)
            pd.DataFrame({"id": [2, 1], "lbp_0": [1, 1]}).to_csv(
                root / "texture_train.csv", index=False
            )
            pd.DataFrame({"id": [2, 1], "boneage": [60, 72], "male": [True, False]}).to_csv(
                root / "metadata_train.csv", index=False
            )
            with self.assertRaisesRegex(ValueError, "IDs|ordem"):
                module.load_texture_split(root, "train", splits)

    def test_custom_resolution_preserves_normalization_and_default(self):
        from PIL import Image
        from image_preprocessing import preprocess_image
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "1.png"
            Image.new("L", (15, 23), 128).save(path)
            default = preprocess_image(path)
            small = preprocess_image(path, size=(128, 128))
        self.assertEqual(default.shape, (224, 224))
        self.assertEqual(small.shape, (128, 128))
        self.assertTrue(np.allclose(small, 128 / 255))

    def test_cv_uses_training_fold_mean_and_preserves_ids(self):
        path = ROOT / "src/08_cross_validation.py"
        self.assertTrue(path.exists(), "Falta o avaliador por dobras")
        module = load("08_cross_validation")
        from sklearn.dummy import DummyRegressor
        X = np.arange(24).reshape(12, 2)
        y = np.arange(12, dtype=float) * 12
        ids = [str(i) for i in range(12)]
        folds = [(np.arange(4, 12), np.arange(0, 4)),
                 (np.r_[0:4, 8:12], np.arange(4, 8)),
                 (np.arange(0, 8), np.arange(8, 12))]
        result, predictions = module.evaluate_folds(
            X, y, ids, folds, {"dummy": DummyRegressor()}, "synthetic"
        )
        self.assertEqual(set(predictions["id"]), set(ids))
        baseline = predictions[predictions.model == "media_treino"]
        self.assertEqual(len(baseline), 12)
        self.assertTrue(np.allclose(baseline[baseline.fold == 1].y_pred, y[4:].mean()))
        self.assertEqual(len(result["media_treino"]["folds"]), 3)
        self.assertGreaterEqual(result["media_treino"]["MAE_meses"]["std"], 0)


if __name__ == "__main__":
    unittest.main()
