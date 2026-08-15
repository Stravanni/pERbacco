"""Compatibility batch runner backed by the maintained CLI."""

from __future__ import annotations

import argparse

from perbacco.cli import DATASETS, main


def _run_dataset(dataset: str) -> int:
    precisions = ("0.5", "0.2", "0.05") if dataset == "synth_10000" else (None,)
    for precision in precisions:
        for method, cda in (
            ("subopt", "none"),
            ("perbacco", "louvain"),
            ("perbac", "none"),
            ("online", "none"),
        ):
            arguments = [
                "run",
                "--dataset",
                dataset,
                "--method",
                method,
                "--cda",
                cda,
            ]
            if precision is not None:
                arguments.extend(("--precision", precision))
            status = main(arguments)
            if status:
                return status
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("all", *DATASETS), required=True)
    arguments = parser.parse_args()
    if arguments.dataset == "all":
        raise SystemExit(main(["reproduce", "--scope", "figure4"]))
    raise SystemExit(_run_dataset(arguments.dataset))
