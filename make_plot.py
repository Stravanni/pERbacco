"""Compatibility renderer for an existing maintained Figure 4 artifact."""

from __future__ import annotations

import json
from pathlib import Path

from perbacco.experiments import _plot_figure4

if __name__ == "__main__":
    artifact = Path("artifacts/figure4.json")
    if not artifact.is_file():
        raise SystemExit(
            "artifacts/figure4.json is missing; run `perbacco reproduce --scope figure4`"
        )
    _plot_figure4(json.loads(artifact.read_text(encoding="utf-8")), artifact.parent)
    print("wrote artifacts/figure4.png and artifacts/figure4.pdf")
