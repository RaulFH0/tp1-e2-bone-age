import importlib.util
from pathlib import Path
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location('oof_analysis', ROOT / 'src/09_oof_error_analysis.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class OOFAnalysisTests(unittest.TestCase):
    def test_rejects_external_or_duplicate_ids(self):
        script = module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({'id': ['a', 'b']}).to_csv(root / 'train_ids.csv', index=False)
            for ids in [['a', 'a'], ['a', 'external']]:
                pd.DataFrame({'id': ids, 'family': 'HOG', 'model': 'GradientBoosting',
                              'fold': [1, 2], 'boneage': [12, 228], 'y_pred': [14, 225]}).to_csv(
                    root / 'predictions.csv', index=False)
                with self.assertRaises(ValueError):
                    script.load_predictions(root / 'predictions.csv', root)

    def test_bias_is_prediction_minus_reference(self):
        result = module().summarize_predictions(pd.DataFrame({
            'boneage': [12., 120., 180., 228.], 'y_pred': [16., 124., 184., 232.]}))
        self.assertEqual(result['MAE_meses'], 4.)
        self.assertEqual(result['bias_meses'], 4.)
        self.assertEqual(result['lower_limit_meses'], 4.)
        self.assertEqual(result['upper_limit_meses'], 4.)
        self.assertEqual(sum(item['n'] for item in result['by_age']), 4)
