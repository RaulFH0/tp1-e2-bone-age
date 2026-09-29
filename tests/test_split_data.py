"""Verifica a partição reprodutível da amostra usada no TP1."""

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "src" / "01_split_data.py"


class SampleSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.labels = self.root / "labels.csv"
        self.sample = self.root / "sample_ids.csv"

        rows = []
        chosen = []
        next_id = 1000
        for months in (12, 96, 156, 204):
            for male in (False, True):
                for index in range(50):
                    rows.append({"id": next_id, "boneage": months, "male": male})
                    if index < 25:
                        chosen.append({"id": next_id})
                    next_id += 1

        self.write_csv(self.labels, ("id", "boneage", "male"), rows)
        self.write_csv(self.sample, ("id",), chosen)
        self.chosen = {int(row["id"]) for row in chosen}

    @staticmethod
    def write_csv(path, columns, rows):
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def run_split(self, output):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--csv", str(self.labels),
             "--sample-ids", str(self.sample), "--out", str(output)],
            capture_output=True, text=True,
        )

    @staticmethod
    def read_ids(path):
        with path.open(newline="", encoding="utf-8") as source:
            return [int(row["id"]) for row in csv.DictReader(source)]

    def test_splits_only_sample_ids_without_overlap_and_repeats_deterministically(self):
        first = self.root / "first"
        result = self.run_split(first)
        self.assertEqual(result.returncode, 0, result.stderr)

        parts = {
            name: self.read_ids(first / f"{name}_ids.csv")
            for name in ("train", "val", "test")
        }
        self.assertEqual([len(parts[name]) for name in ("train", "val", "test")],
                         [140, 30, 30])
        self.assertEqual(set().union(*map(set, parts.values())), self.chosen)
        self.assertEqual(sum(map(len, parts.values())),
                         len(set().union(*map(set, parts.values()))))

        second = self.root / "second"
        self.assertEqual(self.run_split(second).returncode, 0)
        for name in parts:
            self.assertEqual((first / f"{name}_ids.csv").read_bytes(),
                             (second / f"{name}_ids.csv").read_bytes())

    def test_rejects_an_id_outside_the_labels_before_writing_splits(self):
        with self.sample.open("a", encoding="utf-8") as output:
            output.write("999999\n")
        destination = self.root / "invalid"
        result = self.run_split(destination)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ausentes no CSV", result.stderr)
        self.assertFalse((destination / "train_ids.csv").exists())


if __name__ == "__main__":
    unittest.main()
