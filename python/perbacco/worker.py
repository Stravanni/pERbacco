from __future__ import annotations

import argparse
import json
from pathlib import Path

from .communities import external_heavy_communities
from .core import CDA, Method
from .data import load_dataset, load_truth_dataset
from .experiments import execute_run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--precision")
    parser.add_argument(
        "--method", choices=tuple(method.name.lower() for method in Method), required=True
    )
    parser.add_argument(
        "--community-algorithm",
        choices=("none", "paper-greedyhs1000", "builtin-louvain", "louvain", "leiden"),
        required=True,
    )
    parser.add_argument("--query-budget", type=int, required=True)
    parser.add_argument("--phi", type=int, required=True)
    parser.add_argument("--lambda-w", type=float, default=0.05)
    arguments = parser.parse_args()
    dataset = (
        load_truth_dataset(arguments.root, arguments.dataset)
        if arguments.method == "subopt"
        else load_dataset(arguments.root, arguments.dataset, precision=arguments.precision)
    )
    communities = None
    cda_label = None
    if arguments.community_algorithm in {"none", "paper-greedyhs1000"}:
        cda = CDA.NONE
    elif arguments.community_algorithm == "builtin-louvain":
        cda = CDA.LOUVAIN
    else:
        cda = CDA.EXTERNAL
        cda_label = arguments.community_algorithm
        communities = external_heavy_communities(
            dataset.graph,
            algorithm=arguments.community_algorithm,
            lambda_w=arguments.lambda_w,
            batch_size=10,
        )
    _run_id, _result, summary = execute_run(
        dataset,
        artifacts=arguments.artifacts,
        method=Method[arguments.method.upper()],
        cda=cda,
        query_budget=arguments.query_budget,
        phi=arguments.phi,
        precision=arguments.precision,
        lambda_w=arguments.lambda_w,
        external_communities=communities,
        cda_label=cda_label,
        run_tag=arguments.community_algorithm,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
