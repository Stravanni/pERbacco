from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from perbacco.data import load_truth_dataset


class TruthDatasetTests(unittest.TestCase):
    def test_truth_only_loader_includes_record_singletons(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_directory = root / "datasets" / "sample"
            dataset_directory.mkdir(parents=True)
            (dataset_directory / "sample.csv").write_text(
                "id,name\n1,one\n2,two\n3,three\n", encoding="utf-8"
            )
            (dataset_directory / "groundtruth.csv").write_text("id1,id2\n1,2\n", encoding="utf-8")
            dataset = load_truth_dataset(root, "sample")
        self.assertEqual(dataset.graph.ids, (1, 2, 3))
        self.assertEqual(len(dataset.graph.edges), 0)
        self.assertEqual(dataset.truth_pair_count, 1)
        self.assertEqual(dataset.entity_sizes, (2, 1))


if __name__ == "__main__":
    unittest.main()
