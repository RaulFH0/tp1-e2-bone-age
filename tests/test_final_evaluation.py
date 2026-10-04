"""Regressões de alinhamento; fixtures sintéticas, sem avaliar o teste real."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("final_evaluation", ROOT / "src/11_test_evaluation.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class FinalEvaluationTests(unittest.TestCase):
    def test_hog_does_not_use_training_labels_from_an_unverified_cache(self):
        self._check_unverified_cache("features_cache_hog_oficial.joblib", evaluation.get_hog)

    def test_intensity_does_not_use_training_labels_from_an_unverified_cache(self):
        self._check_unverified_cache("features_cache_intensidade.joblib", evaluation.get_intensity)

    def _check_unverified_cache(self, filename, loader):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for image_id in ("1", "2", "3"):
                Image.fromarray(np.arange(1024, dtype=np.uint8).reshape(32, 32)).save(root / f"{image_id}.png")
            meta = pd.DataFrame({"id": ["1", "2", "3"], "boneage": [12, 24, 36], "male": [False, True, False]})
            joblib.dump({"X_train": np.zeros((2, 1)), "y_train": np.array([999, 999])}, root / filename)
            X_train, y_train, X_test, y_test = loader(root, root / "unused.csv", root, root, ["2", "1"], ["3"], meta)
            np.testing.assert_array_equal(y_train, [24, 12])
            np.testing.assert_array_equal(y_test, [36])
            np.testing.assert_array_equal(X_train[:, -1], [1, 0])
            self.assertEqual(len(X_test), 1)

    def test_matching_texture_and_metadata_ids_must_also_match_frozen_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"id": ["WRONG"], "lbp_0": [0]}).to_csv(root / "texture_train.csv", index=False)
            pd.DataFrame({"id": ["WRONG"], "boneage": [12], "male": [0]}).to_csv(root / "metadata_train.csv", index=False)
            with self.assertRaises(ValueError):
                evaluation.load_texture_split(root, "train")

    def test_texture_rejects_historical_pixel_mean_instead_of_twenty_descriptors(self):
        ids = pd.read_csv(ROOT / "data/splits/train_ids.csv", dtype={"id": str})["id"].tolist()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"id": ids, "pixel_mean": np.zeros(len(ids))}).to_csv(root / "texture_train.csv", index=False)
            pd.DataFrame({"id": ids, "boneage": np.full(len(ids), 12), "male": np.zeros(len(ids))}).to_csv(root / "metadata_train.csv", index=False)
            with self.assertRaises(ValueError):
                evaluation.load_texture_split(root, "train")

    def test_texture_rejects_labels_that_differ_from_original_annotations(self):
        for wrong_column, wrong_value in (("boneage", 999), ("male", 0)):
            with self.subTest(column=wrong_column), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                original = self._write_texture_fixture(root)
                packaged = pd.read_csv(root / "metadata_train.csv", dtype={"id": str})
                packaged.loc[0, wrong_column] = wrong_value
                packaged.to_csv(root / "metadata_train.csv", index=False)
                with self.assertRaises(ValueError):
                    evaluation.load_texture_split(root, "train", root, original)

    def test_texture_preserves_frozen_order_and_sex_when_annotations_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = self._write_texture_fixture(root)
            X, y = evaluation.load_texture_split(root, "train", root, original)
            self.assertEqual(X.shape, (2, 21))
            np.testing.assert_array_equal(y, [24, 12])
            np.testing.assert_array_equal(X[:, -1], [1, 0])

    def _write_texture_fixture(self, root):
        ids = ["2", "1"]
        pd.DataFrame({"id": ids}).to_csv(root / "train_ids.csv", index=False)
        columns = [f"lbp_{i}" for i in range(10)] + [
            f"glcm_{property_name}_{statistic}"
            for property_name in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")
            for statistic in ("mean", "std")
        ]
        texture = pd.DataFrame(np.zeros((2, 20)), columns=columns)
        texture.insert(0, "id", ids)
        texture.to_csv(root / "texture_train.csv", index=False)
        pd.DataFrame({"id": ids, "boneage": [24, 12], "male": [1, 0]}).to_csv(root / "metadata_train.csv", index=False)
        return pd.DataFrame({"id": ["1", "2"], "boneage": [12, 24], "male": [False, True]})


if __name__ == "__main__":
    unittest.main()
