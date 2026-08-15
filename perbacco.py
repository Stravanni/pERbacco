"""Compatibility entry point for the packaged pERbacco implementation.

The historical implementation remains available from Git history. Keeping this
small shim avoids shadowing ``python/perbacco`` when the repository root is on
``sys.path`` while preserving ``python perbacco.py ...``.
"""

from __future__ import annotations

from pathlib import Path

_PACKAGE_DIRECTORY = Path(__file__).resolve().parent / "python" / "perbacco"
__path__ = [str(_PACKAGE_DIRECTORY)]

from perbacco.core import (
    CDA,
    Batch,
    BatchKind,
    Engine,
    EngineConfig,
    Graph,
    Method,
    PerbaccoError,
    Statistics,
    Status,
)
from perbacco.oracle import (
    EntityView,
    GroundTruthOracle,
    OpenAICompatibleOracle,
    Oracle,
    OracleProtocolError,
    OracleResult,
)
from perbacco.runner import RunEvent, RunResult, run

__all__ = [
    "CDA",
    "Batch",
    "BatchKind",
    "Engine",
    "EngineConfig",
    "EntityView",
    "Graph",
    "GroundTruthOracle",
    "Method",
    "OpenAICompatibleOracle",
    "Oracle",
    "OracleProtocolError",
    "OracleResult",
    "PerbaccoError",
    "RunEvent",
    "RunResult",
    "Statistics",
    "Status",
    "run",
]
__version__ = "0.1.0"


if __name__ == "__main__":
    from perbacco.cli import main

    raise SystemExit(main())
