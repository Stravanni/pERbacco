from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from perbacco.experiments import _cached_summary


class ExperimentCacheTests(unittest.TestCase):
    def test_cache_rejects_a_stale_implementation_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory) / "runs" / "example"
            run_directory.mkdir(parents=True)
            curve = run_directory / "curve.csv"
            curve.write_text("query,recall\n1,1.0\n", encoding="utf-8")
            summary = {
                "run_id": "example",
                "dataset": "tiny",
                "implementation_sha256": "stale",
            }
            (run_directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (run_directory / "manifest.json").write_text(
                json.dumps({"curve_sha256": hashlib.sha256(curve.read_bytes()).hexdigest()}),
                encoding="utf-8",
            )

            cached = _cached_summary(
                directory,
                "example",
                {"run_id": "example", "dataset": "tiny"},
            )

        self.assertIsNone(cached)


if __name__ == "__main__":
    unittest.main()
