import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def script():
    spec = importlib.util.spec_from_file_location('reproduce', ROOT / 'src/10_reproduce_from_raw.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReproductionPathsTests(unittest.TestCase):
    def test_discovers_nested_training_folder_without_a_specific_image_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / 'training' / 'training'
            validation = root / 'validation'
            training.mkdir(parents=True)
            validation.mkdir()
            for image_id in ['42', '99']:
                (training / f'{image_id}.png').touch()
            (validation / '42.png').touch()
            self.assertEqual(script().find_image_directory(root, ['42', '99']), training)

    def test_rejects_ambiguous_or_incomplete_image_folders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'a').mkdir()
            (root / 'a' / '42.png').touch()
            with self.assertRaises(ValueError):
                script().find_image_directory(root, ['42', '99'])
            (root / 'a' / '99.png').touch()
            (root / 'b').mkdir()
            for image_id in ['42', '99']:
                (root / 'b' / f'{image_id}.png').touch()
            with self.assertRaises(ValueError):
                script().find_image_directory(root, ['42', '99'])
