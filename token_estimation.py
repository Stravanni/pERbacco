from __future__ import annotations

import csv
import json
import math
import os
import random
from argparse import ArgumentParser
from functools import lru_cache
from numbers import Integral
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import ticker
from matplotlib.lines import Line2D
from batch_simple_LLM import (
    build_batch_entities,
    build_few_shot_examples,
    build_generic_entity_payload,
    build_ground_truth_pairs,
    build_match_lookup,
    infer_prompt_profile,
    load_dataset_records,
    load_raw_graph,
    mine_negative_examples,
    mine_positive_examples,
)
from oracle_LL import OpenAIEntityOracle


DEFAULT_BATCH_SIZES = [2, 5, 10, 20, 50]
DEFAULT_PLOT_MODES = ["overview", "heatmap", "pareto"]
SCENARIOS = {
    "p25": 0.25,
    "p50": 0.50,
    "p75": 0.75,
}
COST_MODELS = [
    "recursive_graph_aware",
    "recursive_all_records",
    "one_pass_lower_bound",
]
CSV_FIELDNAMES = [
    "dataset",
    "prompt_profile",
    "prompt_mode",
    "few_shot_pairs_per_class",
    "openai_model",
    "token_source_mode",
    "token_source",
    "dataset_nodes",
    "graph_nodes",
    "graph_edges",
    "duplicate_pair_nodes",
    "singleton_only_nodes",
    "connected_components",
    "largest_component",
    "duplicate_connected_components",
    "duplicate_largest_component",
    "batch_size",
    "cost_model",
    "scenario",
    "estimated_calls",
    "batch_prompt_chars",
    "batch_input_tokens",
    "batch_output_tokens",
    "batch_total_tokens",
    "total_input_tokens",
    "total_output_tokens",
    "total_tokens",
    "pairwise_baseline_calls",
    "pairwise_baseline_total_tokens",
    "saving_factor_vs_pairwise_edges",
    "token_reduction_fraction_vs_pairwise_edges",
]
DATASET_DISPLAY_NAMES = {
    "camera": "Camera",
    "cora": "Cora",
    "funding": "Funding",
    "voters": "Voters",
    "wdc80": "WDC80",
}
REAL_DATASETS = ["camera", "cora", "funding", "voters", "wdc80"]
LINE_COLORS = {
    "camera": "#0072B2",
    "cora": "#D55E00",
    "funding": "#009E73",
    "voters": "#CC79A7",
    "wdc80": "#56B4E9",
}
LINE_MARKERS = {
    "camera": "o",
    "cora": "s",
    "funding": "^",
    "voters": "D",
    "wdc80": "P",
}
def parse_args():
    parser = ArgumentParser(
        description=(
            "Estimate dataset-level LLM token usage across batch sizes for all datasets "
            "with a materialized top-level similarity graph."
        )
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--batch-sizes", nargs="*", type=int, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--prompt-mode", choices=["zero-shot", "few-shot"], default="zero-shot")
    parser.add_argument("--few-shot-pairs-per-class", type=int, default=3)
    parser.add_argument("--token-source", choices=["heuristic", "openai-if-available"], default="heuristic")
    parser.add_argument("--openai-model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results/plot_LLM")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument(
        "--plot-modes",
        nargs="*",
        default=DEFAULT_PLOT_MODES,
        choices=["overview", "heatmap", "pareto"],
    )
    return parser.parse_args()


def sanitize_label(value):
    return str(value).replace("/", "-").replace(" ", "-").replace(":", "-")


def node_sort_key(value):
    if isinstance(value, Integral):
        return (0, int(value), "")
    return (1, 0, str(value))


@lru_cache(maxsize=None)
def q_rec_k(size: int, batch_size: int) -> int:
    if size < batch_size:
        return 0
    quotient = size // batch_size
    remainder = size % batch_size
    return quotient + q_rec_k(quotient + remainder, batch_size)


@lru_cache(maxsize=None)
def r_rec_k(size: int, batch_size: int) -> int:
    if size < batch_size:
        return size
    quotient = size // batch_size
    remainder = size % batch_size
    return r_rec_k(quotient + remainder, batch_size)


def discover_datasets() -> list[str]:
    datasets = []
    for path in sorted(Path("similarity_graph").glob("*.parquet")):
        dataset = path.stem
        dataset_dir = Path("datasets") / dataset
        if not (dataset_dir / f"{dataset}.csv").exists():
            continue
        if not (dataset_dir / "groundtruth.csv").exists():
            continue
        datasets.append(dataset)
    if not datasets:
        raise RuntimeError("No graph-backed datasets were discovered under similarity_graph/*.parquet.")
    return datasets


def resolve_datasets(requested: list[str] | None) -> list[str]:
    available = discover_datasets()
    if requested is None:
        return available

    available_set = set(available)
    missing = [dataset for dataset in requested if dataset not in available_set]
    if missing:
        raise RuntimeError(
            "Requested datasets are not available as top-level graph-backed datasets: "
            + ", ".join(sorted(missing))
        )
    return requested


def build_output_paths(args, model_name: str) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_output_dir = Path("results/token_estimation")

    stem_parts = [
        "token_estimation",
        sanitize_label(model_name),
        args.prompt_mode,
        sanitize_label(args.token_source),
    ]
    if args.prompt_mode == "few-shot":
        stem_parts.append(f"fewshot{args.few_shot_pairs_per_class}x2")
    stem = "_".join(stem_parts)

    return {
        "output_dir": output_dir,
        "legacy_output_dir": legacy_output_dir,
        "csv": output_dir / f"{stem}.csv",
        "json": output_dir / f"{stem}.json",
        "overview_linear_pdf": output_dir / f"{stem}_overview.pdf",
        "overview_linear_png": output_dir / f"{stem}_overview.png",
        "heatmap_pdf": output_dir / f"{stem}_heatmap.pdf",
        "heatmap_png": output_dir / f"{stem}_heatmap.png",
        "pareto_pdf": output_dir / f"{stem}_pareto.pdf",
        "pareto_png": output_dir / f"{stem}_pareto.png",
        "legacy_csv": legacy_output_dir / f"{stem}.csv",
        "legacy_json": legacy_output_dir / f"{stem}.json",
        "legacy_plot_paths": [
            output_dir / f"{stem}_lines.pdf",
            output_dir / f"{stem}_lines.png",
            output_dir / f"{stem}_heatmap.pdf",
            output_dir / f"{stem}_heatmap.png",
            output_dir / f"{stem}_comparison.pdf",
            output_dir / f"{stem}_comparison.png",
        ],
    }


def create_oracle(args) -> OpenAIEntityOracle:
    api_key = None
    if args.token_source == "heuristic":
        api_key = ""
    return OpenAIEntityOracle(
        api_key=api_key,
        model=args.openai_model,
        prompt_mode=args.prompt_mode,
    )


def ensure_records_cover_nodes(records_by_id, all_nodes):
    completed = dict(records_by_id)
    for node in all_nodes:
        completed.setdefault(node, {"id": str(node)})
    return completed


def build_component_graph(all_nodes, df_ground_truth) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(all_nodes)
    for left, right in df_ground_truth.itertuples(index=False):
        graph.add_edge(left, right)
    return graph


def compute_component_sizes(graph: nx.Graph) -> list[int]:
    return sorted((len(component) for component in nx.connected_components(graph)), reverse=True)


def score_dataset_nodes(records_by_id, dataset_nodes, prompt_profile):
    scored_nodes = []
    for node in dataset_nodes:
        payload = build_generic_entity_payload(node, records_by_id, prompt_profile)
        score = len(json.dumps(payload, ensure_ascii=True))
        scored_nodes.append((score, node))
    scored_nodes.sort(key=lambda item: (item[0], node_sort_key(item[1])))
    return scored_nodes


def select_batch_nodes(scored_nodes, batch_size: int, quantile: float) -> list[object]:
    if len(scored_nodes) < batch_size:
        raise RuntimeError(
            f"Cannot select a representative batch of size {batch_size} from only {len(scored_nodes)} nodes."
        )
    position = int(quantile * (len(scored_nodes) - 1))
    start = max(0, min(position - batch_size // 2, len(scored_nodes) - batch_size))
    return [node for _, node in scored_nodes[start : start + batch_size]]


def build_few_shot_examples_for_batch(state, args, batch_nodes, batch_size: int, scenario_label: str):
    if args.prompt_mode != "few-shot":
        return []

    excluded_nodes = set(batch_nodes)
    dataset_seed = sum((index + 1) * ord(char) for index, char in enumerate(state["dataset"]))
    scenario_seed = {
        "p25": 25000,
        "p50": 50000,
        "p75": 75000,
    }[scenario_label]
    rng = random.Random(args.seed + dataset_seed + scenario_seed + batch_size * 1000)

    positive_examples = mine_positive_examples(
        state["ground_truth_pairs"],
        excluded_nodes,
        args.few_shot_pairs_per_class,
        rng,
        prompt_profile=state["prompt_profile"],
        records_by_id=state["records_by_id"],
    )
    negative_examples = mine_negative_examples(
        state["graph"],
        state["match_lookup"],
        excluded_nodes,
        args.few_shot_pairs_per_class,
        rng,
        prompt_profile=state["prompt_profile"],
        records_by_id=state["records_by_id"],
    )
    return build_few_shot_examples(
        state["records_by_id"],
        positive_examples,
        negative_examples,
        state["prompt_profile"],
    )


def estimate_representative_batches(state, oracle, args, batch_size: int) -> dict[str, dict[str, int | str]]:
    estimates = {}
    for scenario_label, quantile in SCENARIOS.items():
        batch_nodes = select_batch_nodes(state["scored_nodes"], batch_size, quantile)
        few_shot_examples = build_few_shot_examples_for_batch(state, args, batch_nodes, batch_size, scenario_label)
        entities, _ = build_batch_entities(batch_nodes, state["records_by_id"], state["prompt_profile"])
        estimate = oracle.estimate_batch_tokens(
            entities,
            prompt_profile=state["prompt_profile"],
            dataset_name=state["dataset"],
            field_names=state["field_names"],
            few_shot_examples=few_shot_examples,
            prompt_mode=args.prompt_mode,
        )
        estimates[scenario_label] = estimate
    return estimates


def estimate_recursive_calls(component_sizes: list[int], batch_size: int, *, include_singletons: bool) -> int:
    if not component_sizes:
        return 0

    minimum = sum(q_rec_k(size, batch_size) for size in component_sizes)
    rest_threshold = 0 if include_singletons else 1
    rests = []
    for size in component_sizes:
        rest = r_rec_k(size, batch_size)
        if rest > rest_threshold:
            rests.append(rest)
    return minimum + math.ceil(sum(rests) / batch_size)


def build_dataset_state(dataset: str):
    records_by_id, field_names = load_dataset_records(dataset)
    df_ground_truth, graph = load_raw_graph(dataset, "False")

    dataset_nodes = sorted(set(graph.nodes()) | set(records_by_id.keys()), key=node_sort_key)
    records_by_id = ensure_records_cover_nodes(records_by_id, dataset_nodes)
    prompt_profile = infer_prompt_profile(dataset, field_names)

    duplicate_pair_nodes = set(df_ground_truth["id1"]) | set(df_ground_truth["id2"])
    singleton_only_nodes = len(set(dataset_nodes) - duplicate_pair_nodes)

    all_records_graph = build_component_graph(dataset_nodes, df_ground_truth)
    duplicate_graph = build_component_graph(duplicate_pair_nodes, df_ground_truth)
    all_component_sizes = compute_component_sizes(all_records_graph)
    duplicate_component_sizes = compute_component_sizes(duplicate_graph) if duplicate_pair_nodes else []

    state = {
        "dataset": dataset,
        "field_names": field_names,
        "records_by_id": records_by_id,
        "graph": graph,
        "graph_nodes": len(graph.nodes()),
        "graph_edges": len(graph.edges()),
        "dataset_nodes": dataset_nodes,
        "dataset_node_count": len(dataset_nodes),
        "prompt_profile": prompt_profile,
        "duplicate_pair_nodes": len(duplicate_pair_nodes),
        "singleton_only_nodes": singleton_only_nodes,
        "connected_components": len(all_component_sizes),
        "largest_component": all_component_sizes[0] if all_component_sizes else 0,
        "duplicate_connected_components": len(duplicate_component_sizes),
        "duplicate_largest_component": duplicate_component_sizes[0] if duplicate_component_sizes else 0,
        "all_component_sizes": all_component_sizes,
        "duplicate_component_sizes": duplicate_component_sizes,
        "scored_nodes": score_dataset_nodes(records_by_id, dataset_nodes, prompt_profile),
    }

    if len(dataset_nodes) == 0:
        raise RuntimeError(f"Dataset {dataset} has no nodes to estimate.")

    state["ground_truth_pairs"] = build_ground_truth_pairs(df_ground_truth)
    state["match_lookup"] = build_match_lookup(df_ground_truth)
    return state


def build_rows_for_dataset(state, oracle, args):
    rows = []
    print(
        f"[{state['dataset']}] nodes={state['dataset_node_count']} graph_edges={state['graph_edges']} "
        f"prompt_profile={state['prompt_profile']}",
        flush=True,
    )

    for batch_size in args.batch_sizes:
        if batch_size < 2:
            raise RuntimeError("Batch sizes must be at least 2.")
        if state["dataset_node_count"] < batch_size:
            raise RuntimeError(
                f"Dataset {state['dataset']} has only {state['dataset_node_count']} nodes, smaller than batch size {batch_size}."
            )

        batch_estimates = estimate_representative_batches(state, oracle, args, batch_size)
        estimated_calls = {
            "recursive_graph_aware": estimate_recursive_calls(
                state["duplicate_component_sizes"],
                batch_size,
                include_singletons=False,
            ),
            "recursive_all_records": estimate_recursive_calls(
                state["all_component_sizes"],
                batch_size,
                include_singletons=True,
            ),
            "one_pass_lower_bound": math.ceil(state["dataset_node_count"] / batch_size),
        }

        for cost_model in COST_MODELS:
            for scenario_label in SCENARIOS:
                estimate = batch_estimates[scenario_label]
                calls = estimated_calls[cost_model]
                rows.append(
                    {
                        "dataset": state["dataset"],
                        "prompt_profile": state["prompt_profile"],
                        "prompt_mode": args.prompt_mode,
                        "few_shot_pairs_per_class": args.few_shot_pairs_per_class,
                        "openai_model": oracle.model,
                        "token_source_mode": args.token_source,
                        "token_source": estimate["source"],
                        "dataset_nodes": state["dataset_node_count"],
                        "graph_nodes": state["graph_nodes"],
                        "graph_edges": state["graph_edges"],
                        "duplicate_pair_nodes": state["duplicate_pair_nodes"],
                        "singleton_only_nodes": state["singleton_only_nodes"],
                        "connected_components": state["connected_components"],
                        "largest_component": state["largest_component"],
                        "duplicate_connected_components": state["duplicate_connected_components"],
                        "duplicate_largest_component": state["duplicate_largest_component"],
                        "batch_size": batch_size,
                        "cost_model": cost_model,
                        "scenario": scenario_label,
                        "estimated_calls": calls,
                        "batch_prompt_chars": int(estimate["prompt_chars"]),
                        "batch_input_tokens": int(estimate["input_tokens"]),
                        "batch_output_tokens": int(estimate["output_tokens"]),
                        "batch_total_tokens": int(estimate["total_tokens"]),
                        "total_input_tokens": int(estimate["input_tokens"]) * calls,
                        "total_output_tokens": int(estimate["output_tokens"]) * calls,
                        "total_tokens": int(estimate["total_tokens"]) * calls,
                    }
                )

        median_row = batch_estimates["p50"]
        print(
            f"  batch={batch_size}: source={median_row['source']} "
            f"batch_tokens={median_row['total_tokens']} "
            f"calls(graph-aware/all/one-pass)="
            f"{estimated_calls['recursive_graph_aware']}/"
            f"{estimated_calls['recursive_all_records']}/"
            f"{estimated_calls['one_pass_lower_bound']}",
            flush=True,
        )

    return rows


def add_saving_factor_fields(rows, *, baseline_batch_size: int = 2):
    grouped = {}
    for row in rows:
        key = (row["dataset"], row["cost_model"], row["scenario"])
        grouped.setdefault(key, {})[row["batch_size"]] = row

    for key, by_batch in grouped.items():
        baseline_row = by_batch.get(baseline_batch_size)
        if baseline_row is None:
            raise RuntimeError(
                f"Missing baseline batch size {baseline_batch_size} for dataset={key[0]} cost_model={key[1]} scenario={key[2]}."
            )
        pairwise_baseline_calls = int(baseline_row["graph_edges"])
        pairwise_baseline_total_tokens = pairwise_baseline_calls * int(baseline_row["batch_total_tokens"])
        for row in by_batch.values():
            total_tokens = int(row["total_tokens"])
            saving_factor = pairwise_baseline_total_tokens / total_tokens if total_tokens else 0.0
            reduction_fraction = (
                1.0 - (total_tokens / pairwise_baseline_total_tokens)
                if pairwise_baseline_total_tokens
                else 0.0
            )
            row["pairwise_baseline_calls"] = pairwise_baseline_calls
            row["pairwise_baseline_total_tokens"] = pairwise_baseline_total_tokens
            row["saving_factor_vs_pairwise_edges"] = round(saving_factor, 6)
            row["token_reduction_fraction_vs_pairwise_edges"] = round(reduction_fraction, 6)
    return rows


def sort_rows(rows, datasets: list[str], batch_sizes: list[int]):
    dataset_order = {dataset: index for index, dataset in enumerate(datasets)}
    batch_order = {batch_size: index for index, batch_size in enumerate(batch_sizes)}
    cost_order = {cost_model: index for index, cost_model in enumerate(COST_MODELS)}
    scenario_order = {label: index for index, label in enumerate(SCENARIOS)}
    return sorted(
        rows,
        key=lambda row: (
            dataset_order[row["dataset"]],
            batch_order[row["batch_size"]],
            cost_order[row["cost_model"]],
            scenario_order[row["scenario"]],
        ),
    )


def write_csv(path: Path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_int(value):
    if value in ("", None):
        return None
    return int(value)


def parse_float(value):
    if value in ("", None):
        return None
    return float(value)


def load_csv_rows(path: Path):
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            normalized = {}
            for field in CSV_FIELDNAMES:
                value = row.get(field, "")
                if field in {
                    "few_shot_pairs_per_class",
                    "dataset_nodes",
                    "graph_nodes",
                    "graph_edges",
                    "duplicate_pair_nodes",
                    "singleton_only_nodes",
                    "connected_components",
                    "largest_component",
                    "duplicate_connected_components",
                    "duplicate_largest_component",
                    "batch_size",
                    "estimated_calls",
                    "batch_prompt_chars",
                    "batch_input_tokens",
                    "batch_output_tokens",
                    "batch_total_tokens",
                    "total_input_tokens",
                    "total_output_tokens",
                    "total_tokens",
                    "pairwise_baseline_calls",
                    "pairwise_baseline_total_tokens",
                }:
                    normalized[field] = parse_int(value)
                elif field in {
                    "saving_factor_vs_pairwise_edges",
                    "token_reduction_fraction_vs_pairwise_edges",
                }:
                    normalized[field] = parse_float(value)
                else:
                    normalized[field] = value
            rows.append(normalized)
    return rows


def load_experiment_rows(path: Path):
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "dataset": row["dataset"],
                    "batch_size": int(row["batch_size"]),
                    "f1": float(row["f1"]),
                    "llm_total_tokens": int(float(row["llm_total_tokens"])),
                }
            )
    return rows


def pretty_dataset_name(dataset: str) -> str:
    return DATASET_DISPLAY_NAMES.get(dataset, dataset)


def format_token_label(value: float | int) -> str:
    absolute = float(value)
    if absolute >= 1_000_000:
        return f"{absolute / 1_000_000:.1f}M"
    if absolute >= 1_000:
        return f"{absolute / 1_000:.1f}K"
    return f"{absolute:.0f}"


def token_axis_formatter(value, _position):
    if value <= 0:
        return ""
    return format_token_label(value)


def build_lookup(rows):
    return {
        (row["dataset"], row["batch_size"], row["cost_model"], row["scenario"]): row
        for row in rows
    }


def best_batch_size_by_cost_model(rows, datasets, batch_sizes):
    summary = {}
    for cost_model in COST_MODELS:
        summary[cost_model] = {}
        for dataset in datasets:
            candidates = [
                row
                for row in rows
                if row["dataset"] == dataset
                and row["cost_model"] == cost_model
                and row["scenario"] == "p50"
            ]
            best = min(candidates, key=lambda row: (row["total_tokens"], row["batch_size"]))
            summary[cost_model][dataset] = {
                "batch_size": best["batch_size"],
                "total_tokens": best["total_tokens"],
            }

        means = {}
        for batch_size in batch_sizes:
            candidates = [
                row["total_tokens"]
                for row in rows
                if row["cost_model"] == cost_model
                and row["scenario"] == "p50"
                and row["batch_size"] == batch_size
            ]
            means[batch_size] = float(np.mean(candidates))
        best_batch = min(batch_sizes, key=lambda batch_size: (means[batch_size], batch_size))
        summary[cost_model]["mean"] = {
            "batch_size": best_batch,
            "total_tokens": means[best_batch],
        }
    return summary


def write_summary_json(path: Path, *, args, oracle, datasets, batch_sizes, rows, paths):
    plot_paths = {}
    if "overview" in args.plot_modes:
        plot_paths["overview_linear"] = {
            "pdf": str(paths["overview_linear_pdf"]),
            "png": str(paths["overview_linear_png"]),
        }
    if "heatmap" in args.plot_modes:
        plot_paths["heatmap"] = {
            "pdf": str(paths["heatmap_pdf"]),
            "png": str(paths["heatmap_png"]),
        }
    if "pareto" in args.plot_modes:
        plot_paths["pareto"] = {
            "pdf": str(paths["pareto_pdf"]),
            "png": str(paths["pareto_png"]),
        }

    payload = {
        "config": {
            "datasets": datasets,
            "batch_sizes": batch_sizes,
            "prompt_mode": args.prompt_mode,
            "few_shot_pairs_per_class": args.few_shot_pairs_per_class,
            "openai_model": oracle.model,
            "token_source_mode": args.token_source,
            "normalization_reference_batch_size": 2,
            "normalization_reference": "pairwise graph-edge queries priced with batch-size-2 tokens",
            "seed": args.seed,
            "plot_modes": args.plot_modes,
            "force_recompute": args.force_recompute,
            "output_dir": str(paths["output_dir"]),
        },
        "artifacts": {
            "csv": str(paths["csv"]),
            "json": str(paths["json"]),
            "plots": plot_paths,
        },
        "best_batch_size": best_batch_size_by_cost_model(rows, datasets, batch_sizes),
        "result_rows": len(rows),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def apply_publication_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "lines.markersize": 4.5,
        }
    )


def cleanup_legacy_plots(paths):
    removable = list(paths["legacy_plot_paths"]) + [
        paths["overview_linear_pdf"],
        paths["overview_linear_png"],
        paths["output_dir"] / f"{paths['csv'].stem}_overview_log.pdf",
        paths["output_dir"] / f"{paths['csv'].stem}_overview_log.png",
        paths["heatmap_pdf"],
        paths["heatmap_png"],
        paths["pareto_pdf"],
        paths["pareto_png"],
    ]
    for plot_path in removable:
        if plot_path.exists():
            plot_path.unlink()


def render_overview(rows, *, datasets, batch_sizes, args, paths):
    lookup = build_lookup(rows)
    fig, ax = plt.subplots(figsize=(4.05, 2.4))

    for dataset in datasets:
        color = LINE_COLORS[dataset]
        marker = LINE_MARKERS[dataset]
        p50_values = np.array(
            [
                float(
                    lookup[(dataset, batch_size, "recursive_all_records", "p50")][
                        "saving_factor_vs_pairwise_edges"
                    ]
                )
                for batch_size in batch_sizes
            ]
        )
        ax.plot(
            batch_sizes,
            p50_values,
            marker=marker,
            color=color,
            label=pretty_dataset_name(dataset),
        )
    all_values = []
    y_max = 1.05
    for dataset in datasets:
        dataset_values = [
            float(
                lookup[(dataset, batch_size, "recursive_all_records", "p50")][
                    "saving_factor_vs_pairwise_edges"
                ]
            )
            for batch_size in batch_sizes
        ]
        all_values.extend(dataset_values)
        dataset_max = max(dataset_values)
        y_max = max(y_max, dataset_max)

    ax.set_xticks(batch_sizes)
    ax.set_ylim(0.0, y_max * 1.08)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0fx"))
    ax.grid(True, which="major", axis="both")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Token Saving Factor w.r.t. Pairwise")
    fig.legend(
        ncol=len(datasets),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.savefig(paths["overview_linear_pdf"], bbox_inches="tight")
    fig.savefig(paths["overview_linear_png"], bbox_inches="tight")
    plt.close(fig)


def render_heatmap(rows, *, datasets, batch_sizes, args, paths):
    b2_csv = Path("results/batch_LLM_b2/experiment_batch_LLM,gpt-5-mini,few-shot,fewshot10x2,x10,seed0.csv")
    main_csv = Path("results/batch_LLM/experiment_batch_LLM,gpt-5-mini,few-shot,fewshot10x2,x10,seed0.csv")
    if not b2_csv.exists() or not main_csv.exists():
        raise RuntimeError(
            "Heatmap source data is missing. Expected existing LLM experiment CSVs under results/batch_LLM_b2 and results/batch_LLM."
        )

    experiment_rows = load_experiment_rows(b2_csv) + load_experiment_rows(main_csv)
    experiment_lookup = {
        (row["dataset"], row["batch_size"]): row
        for row in experiment_rows
        if row["dataset"] in REAL_DATASETS and row["batch_size"] in batch_sizes
    }

    matrix = np.array(
        [[experiment_lookup[(dataset, batch_size)]["f1"] for batch_size in batch_sizes] for dataset in datasets],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(4.05, 2.4))
    image = ax.imshow(matrix, aspect="auto", cmap=plt.cm.RdYlGn, vmin=0.5, vmax=1.0)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    colorbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    colorbar.update_ticks()

    ax.set_xticks(np.arange(len(batch_sizes)))
    ax.set_xticklabels([str(batch_size) for batch_size in batch_sizes])
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels([pretty_dataset_name(dataset) for dataset in datasets])

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Dataset")

    for row_index, dataset in enumerate(datasets):
        for col_index, batch_size in enumerate(batch_sizes):
            value = matrix[row_index, col_index]
            color = "white" if value >= 0.80 else "#111111"
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=6.0,
            )

    fig.savefig(paths["heatmap_pdf"], bbox_inches="tight")
    fig.savefig(paths["heatmap_png"], bbox_inches="tight")
    plt.close(fig)


def load_pareto_fscore_lookup(*, args, oracle, datasets, batch_sizes):
    model_name = sanitize_label(oracle.model)
    preferred_paths = [
        Path(args.output_dir) / f"experiment_batch_LLM,{model_name},augmented-camera30,b2,seed0.csv",
        Path("results/batch_LLM_augmented") / f"experiment_batch_LLM,{model_name},augmented-camera30,b2,seed0.csv",
    ]
    source_rows = None
    for path in preferred_paths:
        if path.exists():
            source_rows = load_experiment_rows(path)
            break

    if source_rows is None:
        source_rows = []
        fallback_paths = [
            Path("results/batch_LLM") / f"experiment_batch_LLM,{model_name},few-shot,fewshot10x2,x10,seed0.csv",
            Path("results/batch_LLM_b2") / f"experiment_batch_LLM,{model_name},few-shot,fewshot10x2,x10,seed0.csv",
            Path("results/batch_LLM_camera_override")
            / f"camera,b10,x10,{model_name},few-shot,fewshot30x2,seed0,camera_catalog.csv",
        ]
        for path in fallback_paths:
            if path.exists():
                source_rows.extend(load_experiment_rows(path))

    lookup = {}
    for row in source_rows:
        key = (row["dataset"], row["batch_size"])
        if row["dataset"] in datasets and row["batch_size"] in batch_sizes:
            lookup[key] = row

    missing = [
        f"{dataset}:b{batch_size}"
        for dataset in datasets
        for batch_size in batch_sizes
        if (dataset, batch_size) not in lookup
    ]
    if missing:
        raise RuntimeError(
            "Pareto source data is missing F-score rows for: " + ", ".join(missing[:8]) + ("..." if len(missing) > 8 else "")
        )
    return lookup


def render_pareto(rows, *, datasets, batch_sizes, args, paths, oracle):
    lookup = build_lookup(rows)
    fscore_lookup = load_pareto_fscore_lookup(
        args=args,
        oracle=oracle,
        datasets=datasets,
        batch_sizes=batch_sizes,
    )

    fig, ax_left = plt.subplots(figsize=(4.6, 3.0))
    ax_right = ax_left.twinx()
    saving_max = 1.0
    fscore_min = 1.0
    fscore_max = 0.0
    dataset_handles = []

    for dataset in datasets:
        saving_values = []
        fscore_values = []
        for batch_size in batch_sizes:
            token_row = lookup[(dataset, batch_size, "recursive_all_records", "p50")]
            saving_value = float(token_row["saving_factor_vs_pairwise_edges"])
            fscore_value = float(fscore_lookup[(dataset, batch_size)]["f1"])
            saving_values.append(saving_value)
            fscore_values.append(fscore_value)

        dataset_line = ax_left.plot(
            batch_sizes,
            fscore_values,
            color=LINE_COLORS[dataset],
            marker=LINE_MARKERS[dataset],
            linestyle="-",
            alpha=0.9,
            label=pretty_dataset_name(dataset),
        )[0]
        ax_right.plot(
            batch_sizes,
            saving_values,
            color=LINE_COLORS[dataset],
            marker=LINE_MARKERS[dataset],
            linestyle="--",
            markerfacecolor="white",
            markeredgewidth=1.0,
            alpha=0.9,
        )

        dataset_handles.append(dataset_line)
        saving_max = max(saving_max, max(saving_values))
        fscore_min = min(fscore_min, min(fscore_values))
        fscore_max = max(fscore_max, max(fscore_values))

    ax_left.set_xticks(batch_sizes)
    ax_left.set_xlim(min(batch_sizes), max(batch_sizes))
    ax_left.set_ylim(max(0.0, fscore_min - 0.03), min(1.01, fscore_max + 0.02))
    ax_right.set_ylim(0.0, saving_max * 1.08)
    ax_right.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0fx"))
    ax_left.grid(True, which="major", axis="both")
    ax_left.set_xlabel("Batch Size")
    ax_left.set_ylabel("F-score")
    ax_right.set_ylabel("Token Saving Factor w.r.t. Pairwise")

    metric_handles = [
        Line2D([0], [0], color="#333333", linestyle="-", marker="o", label="F-score"),
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle="--",
            marker="o",
            markerfacecolor="white",
            label="Saving",
        ),
    ]

    fig.legend(
        dataset_handles,
        [handle.get_label() for handle in dataset_handles],
        ncol=len(datasets),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    ax_left.legend(
        handles=metric_handles,
        loc="lower right",
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(paths["pareto_pdf"], bbox_inches="tight")
    fig.savefig(paths["pareto_png"], bbox_inches="tight")
    plt.close(fig)


def render_plots(rows, *, datasets, batch_sizes, args, paths, oracle):
    apply_publication_style()
    cleanup_legacy_plots(paths)
    if "overview" in args.plot_modes:
        render_overview(rows, datasets=datasets, batch_sizes=batch_sizes, args=args, paths=paths)
    if "heatmap" in args.plot_modes:
        render_heatmap(rows, datasets=datasets, batch_sizes=batch_sizes, args=args, paths=paths)
    if "pareto" in args.plot_modes:
        render_pareto(rows, datasets=datasets, batch_sizes=batch_sizes, args=args, paths=paths, oracle=oracle)


def matching_config(payload: dict[str, object], *, args, oracle, datasets, batch_sizes) -> bool:
    config = payload.get("config", {})
    return (
        config.get("datasets") == datasets
        and config.get("batch_sizes") == batch_sizes
        and config.get("prompt_mode") == args.prompt_mode
        and config.get("few_shot_pairs_per_class") == args.few_shot_pairs_per_class
        and config.get("openai_model") == oracle.model
        and config.get("token_source_mode") == args.token_source
        and config.get("seed") == args.seed
    )


def can_reuse_existing_results(paths: dict[str, Path], *, args, oracle, datasets, batch_sizes) -> bool:
    if args.force_recompute or not paths["csv"].exists() or not paths["json"].exists():
        return False

    try:
        payload = json.loads(paths["json"].read_text())
    except (json.JSONDecodeError, OSError):
        return False

    return matching_config(payload, args=args, oracle=oracle, datasets=datasets, batch_sizes=batch_sizes)


def resolve_reusable_csv_path(paths: dict[str, Path], *, args, oracle, datasets, batch_sizes) -> Path | None:
    candidates = [
        (paths["csv"], paths["json"]),
        (paths["legacy_csv"], paths["legacy_json"]),
    ]
    for csv_path, json_path in candidates:
        if args.force_recompute or not csv_path.exists() or not json_path.exists():
            continue
        try:
            payload = json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if matching_config(payload, args=args, oracle=oracle, datasets=datasets, batch_sizes=batch_sizes):
            return csv_path
    return None


def main():
    args = parse_args()
    datasets = resolve_datasets(args.datasets)
    batch_sizes = sorted(set(args.batch_sizes))
    oracle = create_oracle(args)
    paths = build_output_paths(args, oracle.model)

    reusable_csv_path = resolve_reusable_csv_path(
        paths,
        args=args,
        oracle=oracle,
        datasets=datasets,
        batch_sizes=batch_sizes,
    )
    if reusable_csv_path is not None:
        print(f"Reusing saved estimation rows from {reusable_csv_path}.", flush=True)
        sorted_rows = load_csv_rows(reusable_csv_path)
        if reusable_csv_path != paths["csv"]:
            write_csv(paths["csv"], sorted_rows)
    else:
        all_rows = []
        for dataset in datasets:
            state = build_dataset_state(dataset)
            all_rows.extend(build_rows_for_dataset(state, oracle, args))

        normalized_rows = add_saving_factor_fields(all_rows, baseline_batch_size=2)
        sorted_rows = sort_rows(normalized_rows, datasets, batch_sizes)
        write_csv(paths["csv"], sorted_rows)

    render_plots(sorted_rows, datasets=datasets, batch_sizes=batch_sizes, args=args, paths=paths, oracle=oracle)
    write_summary_json(
        paths["json"],
        args=args,
        oracle=oracle,
        datasets=datasets,
        batch_sizes=batch_sizes,
        rows=sorted_rows,
        paths=paths,
    )

    print(
        f"Saved CSV to {paths['csv']} and summary JSON to {paths['json']}.",
        flush=True,
    )
    for mode in args.plot_modes:
        if mode == "overview":
            print(
                f"Saved overview plot to {paths['overview_linear_pdf']} and {paths['overview_linear_png']}.",
                flush=True,
            )
        if mode == "heatmap":
            print(
                f"Saved heatmap plot to {paths['heatmap_pdf']} and {paths['heatmap_png']}.",
                flush=True,
            )
        if mode == "pareto":
            print(
                f"Saved pareto plot to {paths['pareto_pdf']} and {paths['pareto_png']}.",
                flush=True,
            )


if __name__ == "__main__":
    main()
