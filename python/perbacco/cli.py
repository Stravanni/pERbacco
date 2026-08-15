from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bounds import phi_bounds
from .core import CDA, Method
from .data import load_dataset, load_truth_dataset
from .experiments import (
    DATASETS,
    consolidated_deviation_report,
    execute_run,
    reproduce_figure4,
    reproduce_table_iii,
    reproduce_table_iv,
)


def _method(value: str) -> Method:
    aliases = {"suboptimal": "subopt", "oracle": "online"}
    selected = aliases.get(value.lower(), value.lower())
    try:
        return Method[selected.upper()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"unknown method: {value}") from exc


def _cda(value: str) -> CDA:
    selected = value.lower()
    if selected in {"false", "none"}:
        return CDA.NONE
    try:
        return CDA[selected.upper()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"unknown CDA: {value}") from exc


def _root_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="perbacco")
    _root_arguments(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bounds = subparsers.add_parser("bounds", help="compute Phi bounds")
    bounds.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    bounds.add_argument("--batch-size", type=int, default=10)
    bounds.add_argument("--precision", default=None)

    one_run = subparsers.add_parser("run", help="run one ground-truth-oracle experiment")
    one_run.add_argument("--dataset", choices=DATASETS, required=True)
    one_run.add_argument("--precision", default=None)
    one_run.add_argument("--method", type=_method, default=Method.PERBACCO)
    one_run.add_argument("--cda", type=_cda, default=CDA.LOUVAIN)
    one_run.add_argument("--lambda-w", type=float, default=0.05)
    one_run.add_argument("--batch-size", type=int, default=10)
    one_run.add_argument("--top-k", type=int, default=1000)
    one_run.add_argument("--seed", type=int, default=42)
    one_run.add_argument("--query-budget", type=int, default=None)

    reproduce = subparsers.add_parser("reproduce", help="reproduce paper artifacts")
    reproduce.add_argument(
        "--scope", choices=("table-iii", "table-iv", "figure4", "all"), default="all"
    )
    return parser


def _legacy(argv: list[str]) -> list[str]:
    """Translate the useful historical single-run flags to the maintained CLI."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch_size", default="10")
    parser.add_argument("--alg_community", default="False")
    parser.add_argument("--lambda_w", default="0.05")
    parser.add_argument("--mu_benefit", default="brmean")
    parser.add_argument("--optimal", default="False")
    parser.add_argument("--synth_precision", default="False")
    parsed, _unknown = parser.parse_known_args(argv)
    if parsed.optimal.lower() == "true":
        method = "subopt"
    elif parsed.alg_community.lower() != "false":
        method = "perbacco"
    elif parsed.mu_benefit == "brmax":
        method = "online"
    else:
        method = "perbac"
    cda = "none" if parsed.alg_community.lower() == "false" else parsed.alg_community
    if cda not in {"none", "louvain"}:
        raise SystemExit(
            f"legacy CDA {cda!r} now requires explicit external communities; use `perbacco run`"
        )
    result = [
        "run",
        "--dataset",
        parsed.dataset,
        "--batch-size",
        parsed.batch_size,
        "--method",
        method,
        "--cda",
        cda,
        "--lambda-w",
        parsed.lambda_w,
    ]
    if parsed.synth_precision.lower() != "false":
        result.extend(("--precision", parsed.synth_precision))
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    commands = {"bounds", "run", "reproduce"}
    if arguments and arguments[0] not in commands and arguments[0].startswith("--dataset"):
        arguments = _legacy(arguments)
    parsed = _parser().parse_args(arguments)
    root = parsed.root.resolve()
    artifacts = (
        (root / parsed.artifacts).resolve()
        if not parsed.artifacts.is_absolute()
        else parsed.artifacts
    )

    if parsed.command == "bounds":
        names = DATASETS if parsed.dataset == "all" else (parsed.dataset,)
        rows = []
        for name in names:
            dataset = load_truth_dataset(root, name)
            lower, upper = phi_bounds(dataset.entity_sizes, parsed.batch_size)
            rows.append({"dataset": name, "lower": lower, "upper": upper})
        print(json.dumps(rows, indent=2))
        return 0

    if parsed.command == "run":
        dataset = (
            load_truth_dataset(root, parsed.dataset)
            if parsed.method == Method.SUBOPT
            else load_dataset(root, parsed.dataset, precision=parsed.precision)
        )
        phi, _upper = phi_bounds(dataset.entity_sizes, parsed.batch_size)
        budget = parsed.query_budget if parsed.query_budget is not None else 3 * phi
        _run_id, _result, summary = execute_run(
            dataset,
            artifacts=artifacts,
            method=parsed.method,
            cda=parsed.cda,
            query_budget=budget,
            phi=phi,
            precision=parsed.precision,
            lambda_w=parsed.lambda_w,
            seed=parsed.seed,
            batch_size=parsed.batch_size,
            top_k=parsed.top_k,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    table_iii = table_iv = figure4 = None
    if parsed.scope in {"table-iii", "all"}:
        table_iii = reproduce_table_iii(root, artifacts)
    if parsed.scope in {"table-iv", "all"}:
        table_iv = reproduce_table_iv(root, artifacts)
    if parsed.scope in {"figure4", "all"}:
        figure4 = reproduce_figure4(root, artifacts)
    report = consolidated_deviation_report(artifacts, table_iii, table_iv, figure4)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
