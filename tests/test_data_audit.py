"""Behavior checks for the dataset audit using a tiny local fixture."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "00_audit_data.py"


class DatasetAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.images = self.base / "images"
        self.images.mkdir()
        self.csv_path = self.base / "labels.csv"
        self.out = self.base / "audit.json"

    def write_labels(self, fields, rows):
        with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def image(self, image_id):
        Image.new("L", (12, 8), color=40).save(self.images / f"{image_id}.png")

    def run_audit(self):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--csv", str(self.csv_path),
             "--images-dir", str(self.images), "--out", str(self.out)],
            capture_output=True, text=True, check=False,
        )

    def test_reports_missing_image_and_does_not_invent_patient_groups(self):
        self.write_labels(
            ["id", "boneage", "male"],
            [{"id": "101", "boneage": 24, "male": "True"},
             {"id": "102", "boneage": 48, "male": "False"}],
        )
        self.image("101")

        result = self.run_audit()

        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual(report["n_rows"], 2)
        self.assertEqual(report["images"]["found"], 1)
        self.assertEqual(report["images"]["missing"], 1)
        self.assertEqual(report["sex_counts"], {"female": 1, "male": 1})
        self.assertEqual(report["age_months"]["minimum"], 24)
        self.assertEqual(report["age_months"]["maximum"], 48)
        self.assertEqual(report["patient_grouping"]["status"], "not_verifiable")

    def test_rejects_duplicate_image_ids_before_a_split(self):
        self.write_labels(
            ["id", "boneage", "male"],
            [{"id": "101", "boneage": 24, "male": "True"},
             {"id": "101", "boneage": 48, "male": "False"}],
        )
        self.image("101")

        result = self.run_audit()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicados", result.stderr)
        self.assertFalse(self.out.exists())

    def test_reports_patient_group_column_when_present(self):
        self.write_labels(
            ["id", "boneage", "male", "patient_id"],
            [{"id": "101", "boneage": 24, "male": "True", "patient_id": "p1"},
             {"id": "102", "boneage": 48, "male": "False", "patient_id": "p1"}],
        )
        self.image("101")
        self.image("102")

        result = self.run_audit()

        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual(report["patient_grouping"]["status"], "available")
        self.assertEqual(report["patient_grouping"]["unique_groups"], 1)


if __name__ == "__main__":
    unittest.main()
