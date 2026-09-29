"""Contrato de imagem comum para os descritores da equipe."""

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


class SharedImagePreprocessingTests(unittest.TestCase):
    def test_returns_grayscale_224_float_image_for_all_feature_families(self):
        # Vermelho puro vira luminância 76/255 na conversão de Pillow.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "1377.png"
            Image.new("RGB", (18, 12), (255, 0, 0)).save(path)

            preprocess_image = importlib.import_module(
                "image_preprocessing"
            ).preprocess_image
            result = preprocess_image(path)

        self.assertEqual(result.shape, (224, 224))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.allclose(result, 76 / 255, atol=1 / 255))

    def test_texture_uses_the_same_preprocessed_image_as_other_descriptors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "1377.png"
            Image.new("RGB", (18, 12), (255, 0, 0)).save(path)

            shared_image = importlib.import_module(
                "image_preprocessing"
            ).preprocess_image(path)
            features = importlib.import_module("02_preprocess").extract_features(
                shared_image
            )

        self.assertEqual(len(features), 21)
        self.assertAlmostEqual(features["pixel_mean"], 76 / 255, places=6)
        self.assertTrue(np.isfinite(list(features.values())).all())


if __name__ == "__main__":
    unittest.main()
