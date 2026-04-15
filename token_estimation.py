from __future__ import annotations

import csv
import hashlib
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
from matplotlib import colors, ticker
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


DEFAULT_PLOT_MODES = ["overview", "heatmap", "combined", "combined_with_avg"]
DEFAULT_SAVING_NORMALIZATION = "batch-size-2"
ESTIMATION_MODEL_VERSION = "phi-ideal-prompt-aware-v6"
NAIVE_EDGE_SAMPLE_SIZE = 50000
SCENARIOS = {
    "p25": 0.25,
    "p50": 0.50,
    "p75": 0.75,
}
COST_MODELS = ["phi_ideal"]
CSV_FIELDNAMES = [
    "dataset",
    "prompt_profile",
    "prompt_mode",
    "few_shot_pairs_per_class",
    "few_shot_positive_pairs",
    "few_shot_negative_pairs",
    "openai_model",
    "token_source_mode",
    "token_source",
    "dataset_recall",
    "dataset_precision",
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
    "phi_calls_min",
    "phi_calls_max",
    "estimated_calls",
    "batch_prompt_chars",
    "batch_input_tokens",
    "batch_output_tokens",
    "batch_total_tokens",
    "total_input_tokens",
    "total_output_tokens",
    "total_tokens",
    "true_cluster_pairwise_comparisons",
    "graph_aware_collapse_calls",
    "saved_pairwise_comparisons",
    "comparison_saving_factor_vs_true_clusters",
    "comparison_saving_factor_vs_batch2",
    "graph_aware_per_call_token_ratio_vs_batch2",
    "mapped_token_saving_factor_vs_batch2",
    "naive_graph_edge_pair_tokens",
    "naive_graph_edge_pairwise_total_tokens",
    "plot_token_total",
    "naive_vs_best_token_saving_factor",
    "pairwise_baseline_calls",
    "pairwise_baseline_total_tokens",
    "saving_factor_vs_pairwise_edges",
    "token_reduction_fraction_vs_pairwise_edges",
    "batch2_baseline_total_tokens",
    "saving_factor_vs_batch2",
    "token_reduction_fraction_vs_batch2",
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
    parser.add_argument("--phi-json", type=str, required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--batch-sizes", nargs="*", type=int, default=None)
    parser.add_argument("--prompt-mode", choices=["zero-shot", "few-shot"], default="few-shot")
    parser.add_argument("--few-shot-pairs-per-class", type=int, default=None)
    parser.add_argument("--few-shot-positive-pairs", type=int, default=None)
    parser.add_argument("--few-shot-negative-pairs", type=int, default=None)
    parser.add_argument("--token-source", choices=["heuristic", "openai-if-available"], default="heuristic")
    parser.add_argument("--openai-model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results/plot_LLM")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument(
        "--saving-normalization",
        choices=[
            "naive-vs-best-token",
            "mapped-token-batch-size-2",
            "comparison-batch-size-2",
            "comparison-true-clusters",
            "token-batch-size-2",
            "token-pairwise-edges",
            "token-pairwise-edges-anchored-b2",
            "pairwise-edges",
            "pairwise-edges-anchored-b2",
            "batch-size-2",
        ],
        default=DEFAULT_SAVING_NORMALIZATION,
    )
    parser.add_argument(
        "--plot-modes",
        nargs="*",
        default=DEFAULT_PLOT_MODES,
        choices=["overview", "heatmap", "combined", "combined_with_avg"],
    )
    args = parser.parse_args()
    if args.few_shot_pairs_per_class is not None:
        if args.few_shot_positive_pairs is None:
            args.few_shot_positive_pairs = args.few_shot_pairs_per_class
        if args.few_shot_negative_pairs is None:
            args.few_shot_negative_pairs = args.few_shot_pairs_per_class
    if args.few_shot_positive_pairs is None:
        args.few_shot_positive_pairs = 10
    if args.few_shot_negative_pairs is None:
        args.few_shot_negative_pairs = 10
    return args


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


def load_phi_payload(phi_json_argument: str) -> dict[str, object]:
    candidate = phi_json_argument.strip()
    if candidate.startswith("{") or candidate.startswith("["):
        payload = json.loads(candidate)
    else:
        path = Path(candidate)
        if not path.exists():
            raise RuntimeError(f"--phi-json is neither valid JSON nor an existing path: {phi_json_argument}")
        payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError("Phi JSON must decode to a top-level object.")
    return payload


def resolve_phi_config(phi_payload: dict[str, object], datasets: list[str]) -> dict[str, dict[str, object]]:
    resolved = {}
    for dataset in datasets:
        metadata = phi_payload.get(dataset)
        phi_entry = phi_payload.get(f"{dataset}_Phi")
        if not isinstance(metadata, dict):
            raise RuntimeError(f"Phi JSON is missing metadata for dataset '{dataset}'.")
        if not isinstance(phi_entry, dict):
            raise RuntimeError(f"Phi JSON is missing '{dataset}_Phi'.")

        batch_ranges: dict[int, tuple[int, int]] = {}
        for key, value in phi_entry.items():
            try:
                batch_size = int(key)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid Phi batch size for dataset '{dataset}': {key!r}") from exc
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 2
                or any(not isinstance(item, Integral) for item in value)
            ):
                raise RuntimeError(
                    f"Phi entry for dataset '{dataset}', batch size {batch_size}, must be a two-integer list."
                )
            low, high = int(value[0]), int(value[1])
            if low <= 0 or high <= 0 or low > high:
                raise RuntimeError(
                    f"Phi entry for dataset '{dataset}', batch size {batch_size}, must satisfy 0 < low <= high."
                )
            batch_ranges[batch_size] = (low, high)

        resolved[dataset] = {
            "recall": float(metadata.get("recall", 0.0)),
            "precision": float(metadata.get("precision", 0.0)),
            "batch_ranges": batch_ranges,
        }
    return resolved


def resolve_batch_sizes(args, phi_config: dict[str, dict[str, object]], datasets: list[str]) -> list[int]:
    if args.batch_sizes is None:
        shared = None
        for dataset in datasets:
            keys = set(phi_config[dataset]["batch_ranges"].keys())
            shared = keys if shared is None else shared & keys
        batch_sizes = sorted(shared or [])
        if not batch_sizes:
            raise RuntimeError("No shared batch sizes were found across the selected datasets in the Phi JSON.")
        return batch_sizes

    batch_sizes = sorted(set(args.batch_sizes))
    missing = []
    for dataset in datasets:
        available = set(phi_config[dataset]["batch_ranges"].keys())
        for batch_size in batch_sizes:
            if batch_size not in available:
                missing.append(f"{dataset}:b{batch_size}")
    if missing:
        raise RuntimeError(
            "Requested batch sizes are missing from the Phi JSON for: "
            + ", ".join(missing[:8])
            + ("..." if len(missing) > 8 else "")
        )
    return batch_sizes


def phi_fingerprint(phi_config: dict[str, dict[str, object]], datasets: list[str]) -> str:
    normalized = {
        dataset: {
            "recall": phi_config[dataset]["recall"],
            "precision": phi_config[dataset]["precision"],
            "batch_ranges": {
                str(batch_size): list(phi_config[dataset]["batch_ranges"][batch_size])
                for batch_size in sorted(phi_config[dataset]["batch_ranges"].keys())
            },
        }
        for dataset in datasets
    }
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        stem_parts.append(f"fewshot{args.few_shot_positive_pairs}p{args.few_shot_negative_pairs}n")
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
        "combined_pdf": output_dir / f"{stem}_combined.pdf",
        "combined_png": output_dir / f"{stem}_combined.png",
        "combined_with_avg_pdf": output_dir / f"{stem}_combined_with_avg.pdf",
        "combined_with_avg_png": output_dir / f"{stem}_combined_with_avg.png",
        "intro_tsaving_pdf": output_dir / f"{stem}_intro_tsaving.pdf",
        "intro_tsaving_png": output_dir / f"{stem}_intro_tsaving.png",
        "intro_fscore_pdf": output_dir / f"{stem}_intro_fscore.pdf",
        "intro_fscore_png": output_dir / f"{stem}_intro_fscore.png",
        "intro_accuracy_pdf": output_dir / f"{stem}_intro_accuracy.pdf",
        "intro_accuracy_png": output_dir / f"{stem}_intro_accuracy.png",
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


def compute_components(graph: nx.Graph) -> list[tuple[object, ...]]:
    return [tuple(component) for component in nx.connected_components(graph)]


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


def sample_graph_edges(graph: nx.Graph, *, sample_size: int, rng: random.Random) -> list[tuple[object, object]]:
    sampled_edges: list[tuple[object, object]] = []
    for index, edge in enumerate(graph.edges()):
        if index < sample_size:
            sampled_edges.append(edge)
            continue
        replacement_index = rng.randint(0, index)
        if replacement_index < sample_size:
            sampled_edges[replacement_index] = edge
    return sampled_edges


def estimate_representative_edge_pairs(state, oracle, args) -> dict[str, dict[str, int | str]]:
    if state["graph_edges"] == 0:
        return {
            scenario_label: {
                "source": "heuristic",
                "prompt_chars": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }
            for scenario_label in SCENARIOS
        }

    dataset_seed = sum((index + 1) * ord(char) for index, char in enumerate(state["dataset"]))
    rng = random.Random(args.seed + dataset_seed + 1777)
    sampled_edges = sample_graph_edges(
        state["graph"],
        sample_size=min(state["graph_edges"], NAIVE_EDGE_SAMPLE_SIZE),
        rng=rng,
    )
    scored_edges = []
    for left, right in sampled_edges:
        ordered = tuple(
            sorted(
                (left, right),
                key=lambda node: (state["node_score_lookup"][node], node_sort_key(node)),
            )
        )
        score = state["node_score_lookup"][ordered[0]] + state["node_score_lookup"][ordered[1]]
        scored_edges.append((score, ordered[0], ordered[1]))
    scored_edges.sort(
        key=lambda item: (
            item[0],
            node_sort_key(item[1]),
            node_sort_key(item[2]),
        )
    )

    estimates = {}
    cache = {}
    for scenario_label, quantile in SCENARIOS.items():
        position = int(quantile * (len(scored_edges) - 1))
        _, left, right = scored_edges[position]
        estimate = estimate_specific_batch_tokens(
            state,
            oracle,
            args,
            [left, right],
            scenario_label,
            cache,
        )
        estimates[scenario_label] = estimate
    return estimates


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
        args.few_shot_positive_pairs,
        rng,
        prompt_profile=state["prompt_profile"],
        records_by_id=state["records_by_id"],
    )
    negative_examples = mine_negative_examples(
        state["graph"],
        state["match_lookup"],
        excluded_nodes,
        args.few_shot_negative_pairs,
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


def scenario_index(length: int, scenario_label: str) -> int:
    if length <= 1:
        return 0
    return int(SCENARIOS[scenario_label] * (length - 1))


def choose_representative_node(members, scenario_label: str, node_score_lookup):
    sorted_members = sorted(members, key=lambda node: (node_score_lookup[node], node_sort_key(node)))
    return sorted_members[scenario_index(len(sorted_members), scenario_label)]


def estimate_specific_batch_tokens(state, oracle, args, batch_nodes, scenario_label: str, cache):
    cache_key = (scenario_label, tuple(batch_nodes))
    if cache_key in cache:
        return cache[cache_key]

    few_shot_examples = build_few_shot_examples_for_batch(state, args, batch_nodes, len(batch_nodes), scenario_label)
    entities, _ = build_batch_entities(batch_nodes, state["records_by_id"], state["prompt_profile"])
    estimate = oracle.estimate_batch_tokens(
        entities,
        prompt_profile=state["prompt_profile"],
        dataset_name=state["dataset"],
        field_names=state["field_names"],
        few_shot_examples=few_shot_examples,
        prompt_mode=args.prompt_mode,
    )
    cache[cache_key] = estimate
    return estimate


def empty_usage():
    return {
        "source": "heuristic",
        "prompt_chars": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def add_usage(total, estimate):
    total["prompt_chars"] += int(estimate["prompt_chars"])
    total["input_tokens"] += int(estimate["input_tokens"])
    total["output_tokens"] += int(estimate["output_tokens"])
    total["total_tokens"] += int(estimate["total_tokens"])
    if total["source"] != estimate["source"]:
        total["source"] = estimate["source"] if total["total_tokens"] == int(estimate["total_tokens"]) else "mixed"


def collapse_component_tokens(component_nodes, batch_size: int, scenario_label: str, state, oracle, args, cache):
    if len(component_nodes) <= 1:
        representative = component_nodes[0]
        return empty_usage(), 0, representative

    items = [
        {
            "members": (node,),
            "representative": node,
        }
        for node in sorted(component_nodes, key=lambda node: (state["node_score_lookup"][node], node_sort_key(node)))
    ]
    usage = empty_usage()
    calls = 0

    while len(items) > 1:
        items.sort(
            key=lambda item: (
                state["node_score_lookup"][item["representative"]],
                node_sort_key(item["representative"]),
            )
        )
        if len(items) <= batch_size:
            batch_items = items
            batch_nodes = [item["representative"] for item in batch_items]
            estimate = estimate_specific_batch_tokens(state, oracle, args, batch_nodes, scenario_label, cache)
            add_usage(usage, estimate)
            merged_members = [member for item in batch_items for member in item["members"]]
            representative = choose_representative_node(merged_members, scenario_label, state["node_score_lookup"])
            items = [{"members": tuple(merged_members), "representative": representative}]
            calls += 1
            break

        new_items = []
        index = 0
        while index + batch_size <= len(items):
            batch_items = items[index : index + batch_size]
            batch_nodes = [item["representative"] for item in batch_items]
            estimate = estimate_specific_batch_tokens(state, oracle, args, batch_nodes, scenario_label, cache)
            add_usage(usage, estimate)
            merged_members = [member for item in batch_items for member in item["members"]]
            representative = choose_representative_node(merged_members, scenario_label, state["node_score_lookup"])
            new_items.append({"members": tuple(merged_members), "representative": representative})
            calls += 1
            index += batch_size
        new_items.extend(items[index:])
        items = new_items

    return usage, calls, items[0]["representative"]


def estimate_one_pass_tokens(representatives, batch_size: int, scenario_label: str, state, oracle, args, cache):
    ordered_reps = sorted(
        representatives,
        key=lambda node: (state["node_score_lookup"][node], node_sort_key(node)),
    )
    usage = empty_usage()
    calls = 0
    index = 0
    while index < len(ordered_reps):
        batch_nodes = ordered_reps[index : index + batch_size]
        if len(batch_nodes) <= 1:
            break
        estimate = estimate_specific_batch_tokens(state, oracle, args, batch_nodes, scenario_label, cache)
        add_usage(usage, estimate)
        calls += 1
        index += batch_size
    return usage, calls


def summarize_usage(state, batch_size: int, cost_model: str, scenario_label: str, usage, calls, graph_aware_collapse_calls: int):
    average_prompt_chars = round(usage["prompt_chars"] / calls) if calls else 0
    average_input_tokens = round(usage["input_tokens"] / calls) if calls else 0
    average_output_tokens = round(usage["output_tokens"] / calls) if calls else 0
    average_total_tokens = round(usage["total_tokens"] / calls) if calls else 0
    true_cluster_pairwise_comparisons = int(state["true_cluster_pairwise_comparisons"])
    saved_pairwise_comparisons = max(0, true_cluster_pairwise_comparisons - graph_aware_collapse_calls)
    return {
        "dataset": state["dataset"],
        "prompt_profile": state["prompt_profile"],
        "prompt_mode": state["prompt_mode"],
        "few_shot_pairs_per_class": state["few_shot_pairs_per_class"],
        "openai_model": state["openai_model"],
        "token_source_mode": state["token_source_mode"],
        "token_source": usage["source"],
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
        "batch_prompt_chars": average_prompt_chars,
        "batch_input_tokens": average_input_tokens,
        "batch_output_tokens": average_output_tokens,
        "batch_total_tokens": average_total_tokens,
        "total_input_tokens": usage["input_tokens"],
        "total_output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "true_cluster_pairwise_comparisons": true_cluster_pairwise_comparisons,
        "graph_aware_collapse_calls": graph_aware_collapse_calls,
        "saved_pairwise_comparisons": saved_pairwise_comparisons,
    }


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
    all_components = compute_components(all_records_graph)
    duplicate_components = compute_components(duplicate_graph) if duplicate_pair_nodes else []
    true_cluster_pairwise_comparisons = sum(
        len(component) * (len(component) - 1) // 2 for component in duplicate_components if len(component) > 1
    )
    scored_nodes = score_dataset_nodes(records_by_id, dataset_nodes, prompt_profile)
    node_score_lookup = {node: score for score, node in scored_nodes}

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
        "all_components": all_components,
        "duplicate_components": duplicate_components,
        "true_cluster_pairwise_comparisons": true_cluster_pairwise_comparisons,
        "scored_nodes": scored_nodes,
        "node_score_lookup": node_score_lookup,
    }

    if len(dataset_nodes) == 0:
        raise RuntimeError(f"Dataset {dataset} has no nodes to estimate.")

    state["ground_truth_pairs"] = build_ground_truth_pairs(df_ground_truth)
    state["match_lookup"] = build_match_lookup(df_ground_truth)
    return state


def phi_scenario_calls(call_bounds: tuple[int, int]) -> dict[str, int]:
    lower, upper = call_bounds
    return {
        "p25": lower,
        "p50": (lower + upper) // 2,
        "p75": upper,
    }


def build_rows_for_dataset(state, oracle, args, phi_dataset: dict[str, object], batch_sizes: list[int]):
    rows = []
    state = dict(state)
    state["prompt_mode"] = args.prompt_mode
    state["few_shot_pairs_per_class"] = args.few_shot_pairs_per_class
    state["few_shot_positive_pairs"] = args.few_shot_positive_pairs
    state["few_shot_negative_pairs"] = args.few_shot_negative_pairs
    state["openai_model"] = oracle.model
    state["token_source_mode"] = args.token_source
    print(
        f"[{state['dataset']}] nodes={state['dataset_node_count']} graph_edges={state['graph_edges']} "
        f"prompt_profile={state['prompt_profile']}",
        flush=True,
    )
    naive_edge_pair_estimates = estimate_representative_edge_pairs(state, oracle, args)
    phi_batch_ranges = phi_dataset["batch_ranges"]
    dataset_recall = float(phi_dataset["recall"])
    dataset_precision = float(phi_dataset["precision"])

    for batch_size in batch_sizes:
        if batch_size < 2:
            raise RuntimeError("Batch sizes must be at least 2.")
        if state["dataset_node_count"] < batch_size:
            raise RuntimeError(
                f"Dataset {state['dataset']} has only {state['dataset_node_count']} nodes, smaller than batch size {batch_size}."
            )
        batch_estimates = estimate_representative_batches(state, oracle, args, batch_size)
        phi_calls_min, phi_calls_max = phi_batch_ranges[batch_size]
        scenario_calls = phi_scenario_calls((phi_calls_min, phi_calls_max))
        median_calls = 0
        median_total_tokens = 0
        median_source = "heuristic"

        for scenario_label in SCENARIOS:
            estimate = batch_estimates[scenario_label]
            naive_edge_estimate = naive_edge_pair_estimates[scenario_label]
            calls = int(scenario_calls[scenario_label])
            total_input_tokens = int(estimate["input_tokens"]) * calls
            total_output_tokens = int(estimate["output_tokens"]) * calls
            total_tokens = total_input_tokens + total_output_tokens
            true_cluster_pairwise_comparisons = int(state["true_cluster_pairwise_comparisons"])
            saved_pairwise_comparisons = max(0, true_cluster_pairwise_comparisons - calls)
            naive_pair_tokens = int(naive_edge_estimate["total_tokens"])
            naive_pairwise_total_tokens = naive_pair_tokens * state["graph_edges"]
            row = {
                "dataset": state["dataset"],
                "prompt_profile": state["prompt_profile"],
                "prompt_mode": args.prompt_mode,
                "few_shot_pairs_per_class": args.few_shot_pairs_per_class,
                "few_shot_positive_pairs": args.few_shot_positive_pairs,
                "few_shot_negative_pairs": args.few_shot_negative_pairs,
                "openai_model": oracle.model,
                "token_source_mode": args.token_source,
                "token_source": estimate["source"],
                "dataset_recall": dataset_recall,
                "dataset_precision": dataset_precision,
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
                "cost_model": "phi_ideal",
                "scenario": scenario_label,
                "phi_calls_min": phi_calls_min,
                "phi_calls_max": phi_calls_max,
                "estimated_calls": calls,
                "batch_prompt_chars": int(estimate["prompt_chars"]),
                "batch_input_tokens": int(estimate["input_tokens"]),
                "batch_output_tokens": int(estimate["output_tokens"]),
                "batch_total_tokens": int(estimate["total_tokens"]),
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_tokens,
                "true_cluster_pairwise_comparisons": true_cluster_pairwise_comparisons,
                "graph_aware_collapse_calls": calls,
                "saved_pairwise_comparisons": saved_pairwise_comparisons,
                "comparison_saving_factor_vs_true_clusters": None,
                "comparison_saving_factor_vs_batch2": None,
                "graph_aware_per_call_token_ratio_vs_batch2": None,
                "mapped_token_saving_factor_vs_batch2": None,
                "naive_graph_edge_pair_tokens": naive_pair_tokens,
                "naive_graph_edge_pairwise_total_tokens": naive_pairwise_total_tokens,
                "plot_token_total": None,
                "naive_vs_best_token_saving_factor": None,
                "pairwise_baseline_calls": None,
                "pairwise_baseline_total_tokens": None,
                "saving_factor_vs_pairwise_edges": None,
                "token_reduction_fraction_vs_pairwise_edges": None,
                "batch2_baseline_total_tokens": None,
                "saving_factor_vs_batch2": None,
                "token_reduction_fraction_vs_batch2": None,
            }
            rows.append(row)
            if scenario_label == "p50":
                median_calls = calls
                median_total_tokens = total_tokens
                median_source = estimate["source"]

        print(
            f"  batch={batch_size}: source={median_source} total_tokens(phi_ideal)={median_total_tokens} "
            f"calls(phi_ideal)={median_calls}",
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
        pairwise_baseline_total_tokens = int(baseline_row["naive_graph_edge_pairwise_total_tokens"])
        batch2_baseline_total_tokens = int(baseline_row["total_tokens"])
        batch2_graph_aware_collapse_calls = int(baseline_row["graph_aware_collapse_calls"])
        for row in by_batch.values():
            total_tokens = int(row["total_tokens"])
            true_cluster_pairwise_comparisons = int(row["true_cluster_pairwise_comparisons"])
            graph_aware_collapse_calls = int(row["graph_aware_collapse_calls"])
            pairwise_saving_factor = pairwise_baseline_total_tokens / total_tokens if total_tokens else 0.0
            pairwise_reduction_fraction = (
                1.0 - (total_tokens / pairwise_baseline_total_tokens)
                if pairwise_baseline_total_tokens
                else 0.0
            )
            batch2_saving_factor = batch2_baseline_total_tokens / total_tokens if total_tokens else 0.0
            batch2_reduction_fraction = (
                1.0 - (total_tokens / batch2_baseline_total_tokens)
                if batch2_baseline_total_tokens
                else 0.0
            )
            if true_cluster_pairwise_comparisons == 0 and graph_aware_collapse_calls == 0:
                comparison_saving_factor_vs_true_clusters = 1.0
            elif graph_aware_collapse_calls == 0:
                comparison_saving_factor_vs_true_clusters = 0.0
            else:
                comparison_saving_factor_vs_true_clusters = (
                    true_cluster_pairwise_comparisons / graph_aware_collapse_calls
                )
            if batch2_graph_aware_collapse_calls == 0 and graph_aware_collapse_calls == 0:
                comparison_saving_factor_vs_batch2 = 1.0
            elif graph_aware_collapse_calls == 0:
                comparison_saving_factor_vs_batch2 = 0.0
            else:
                comparison_saving_factor_vs_batch2 = batch2_graph_aware_collapse_calls / graph_aware_collapse_calls
            row["comparison_saving_factor_vs_true_clusters"] = round(
                comparison_saving_factor_vs_true_clusters,
                6,
            )
            row["comparison_saving_factor_vs_batch2"] = round(comparison_saving_factor_vs_batch2, 6)
            row["pairwise_baseline_calls"] = pairwise_baseline_calls
            row["pairwise_baseline_total_tokens"] = pairwise_baseline_total_tokens
            row["saving_factor_vs_pairwise_edges"] = round(pairwise_saving_factor, 6)
            row["token_reduction_fraction_vs_pairwise_edges"] = round(pairwise_reduction_fraction, 6)
            row["batch2_baseline_total_tokens"] = batch2_baseline_total_tokens
            row["saving_factor_vs_batch2"] = round(batch2_saving_factor, 6)
            row["token_reduction_fraction_vs_batch2"] = round(batch2_reduction_fraction, 6)

    by_dataset_scenario = {}
    for row in rows:
        key = (row["dataset"], row["scenario"])
        by_dataset_scenario.setdefault(key, {})[row["batch_size"]] = row

    for key, by_batch in by_dataset_scenario.items():
        baseline_row = by_batch[baseline_batch_size]
        baseline_calls = int(baseline_row["graph_aware_collapse_calls"])
        baseline_avg_tokens = float(baseline_row["batch_total_tokens"] or 0)
        naive_baseline_total = int(baseline_row["naive_graph_edge_pairwise_total_tokens"])

        for row in by_batch.values():
            current_calls = int(row["graph_aware_collapse_calls"])
            current_avg_tokens = float(row["batch_total_tokens"] or 0)

            if baseline_avg_tokens == 0.0 and current_avg_tokens == 0.0:
                per_call_ratio = 1.0
            elif current_avg_tokens == 0.0:
                per_call_ratio = 0.0
            else:
                per_call_ratio = baseline_avg_tokens / current_avg_tokens

            mapped_token_saving_factor = float(row["comparison_saving_factor_vs_batch2"]) * per_call_ratio
            row["graph_aware_per_call_token_ratio_vs_batch2"] = round(per_call_ratio, 6)
            row["mapped_token_saving_factor_vs_batch2"] = round(mapped_token_saving_factor, 6)

            if int(row["batch_size"]) == baseline_batch_size:
                plot_token_total = naive_baseline_total
            else:
                plot_token_total = int(row["total_tokens"])
            if naive_baseline_total == 0 and plot_token_total == 0:
                naive_vs_best_token_saving_factor = 1.0
            elif plot_token_total == 0:
                naive_vs_best_token_saving_factor = 0.0
            else:
                naive_vs_best_token_saving_factor = naive_baseline_total / plot_token_total
            row["plot_token_total"] = plot_token_total
            row["naive_vs_best_token_saving_factor"] = round(naive_vs_best_token_saving_factor, 6)
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
                    "few_shot_positive_pairs",
                    "few_shot_negative_pairs",
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
                    "phi_calls_min",
                    "phi_calls_max",
                    "estimated_calls",
                    "batch_prompt_chars",
                    "batch_input_tokens",
                    "batch_output_tokens",
                    "batch_total_tokens",
                    "total_input_tokens",
                    "total_output_tokens",
                    "total_tokens",
                    "true_cluster_pairwise_comparisons",
                    "graph_aware_collapse_calls",
                    "saved_pairwise_comparisons",
                    "naive_graph_edge_pair_tokens",
                    "naive_graph_edge_pairwise_total_tokens",
                    "plot_token_total",
                    "pairwise_baseline_calls",
                    "pairwise_baseline_total_tokens",
                    "batch2_baseline_total_tokens",
                }:
                    normalized[field] = parse_int(value)
                elif field in {
                    "dataset_recall",
                    "dataset_precision",
                    "naive_vs_best_token_saving_factor",
                    "graph_aware_per_call_token_ratio_vs_batch2",
                    "mapped_token_saving_factor_vs_batch2",
                    "comparison_saving_factor_vs_true_clusters",
                    "comparison_saving_factor_vs_batch2",
                    "saving_factor_vs_pairwise_edges",
                    "token_reduction_fraction_vs_pairwise_edges",
                    "saving_factor_vs_batch2",
                    "token_reduction_fraction_vs_batch2",
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
            normalized = dict(row)
            normalized["dataset"] = row["dataset"]
            normalized["batch_size"] = int(row["batch_size"])
            normalized["f1"] = float(row["f1"])
            normalized["llm_total_tokens"] = int(float(row["llm_total_tokens"]))
            for field in [
                "queried_pair_events",
                "gt_duplicate_pairs",
                "correctly_identified_duplicate_pairs",
                "predicted_positive_pairs",
            ]:
                if field in row and row[field] not in ("", None):
                    normalized[field] = int(float(row[field]))
            rows.append(normalized)
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


def saving_factor_field(args) -> str:
    if args.saving_normalization == "naive-vs-best-token":
        return "naive_vs_best_token_saving_factor"
    if args.saving_normalization == "mapped-token-batch-size-2":
        return "mapped_token_saving_factor_vs_batch2"
    if args.saving_normalization == "comparison-true-clusters":
        return "comparison_saving_factor_vs_true_clusters"
    if args.saving_normalization == "comparison-batch-size-2":
        return "comparison_saving_factor_vs_batch2"
    if args.saving_normalization in {"pairwise-edges", "token-pairwise-edges"}:
        return "saving_factor_vs_pairwise_edges"
    if args.saving_normalization in {"pairwise-edges-anchored-b2", "token-pairwise-edges-anchored-b2"}:
        return "saving_factor_vs_pairwise_edges"
    return "saving_factor_vs_batch2"


def saving_axis_label(args) -> str:
    if args.saving_normalization in {"comparison-true-clusters", "comparison-batch-size-2"}:
        return "Comparison Saving Factor"
    return "Token Saving Factor"


def normalization_reference_text(args) -> str:
    if args.saving_normalization == "naive-vs-best-token":
        return (
            "naive batch-size-2 graph-edge pairwise total tokens as the baseline, "
            "versus Phi-driven ideal-method total tokens for batch sizes greater than 2"
        )
    if args.saving_normalization == "mapped-token-batch-size-2":
        return (
            "graph-aware comparison saving at batch size 2 divided by graph-aware comparison saving at batch size b, "
            "scaled by the ratio of average tokens per graph-aware call at batch size 2 versus batch size b"
        )
    if args.saving_normalization == "comparison-true-clusters":
        return "true duplicate-cluster pairwise comparisons divided by graph-aware collapse calls"
    if args.saving_normalization == "comparison-batch-size-2":
        return "within each dataset/scenario, graph-aware collapse calls at batch size 2 divided by graph-aware collapse calls at batch size b"
    if args.saving_normalization in {"pairwise-edges", "token-pairwise-edges"}:
        return "pairwise graph-edge queries priced with batch-size-2 tokens"
    if args.saving_normalization in {"pairwise-edges-anchored-b2", "token-pairwise-edges-anchored-b2"}:
        return "pairwise graph-edge queries priced with batch-size-2 tokens, with batch size 2 anchored at 1 in the overview plot"
    return "within each dataset/cost-model/scenario, total tokens at batch size 2"


def plotted_saving_factor(row, *, args) -> float:
    if args.saving_normalization in {"pairwise-edges-anchored-b2", "token-pairwise-edges-anchored-b2"} and int(row["batch_size"]) == 2:
        return 1.0
    return float(row[saving_factor_field(args)])


def plotted_cost_model(args) -> str:
    return "phi_ideal"


def saving_tick_formatter(value, _position):
    if value < 0:
        return ""
    if value < 10:
        return f"{value:.1f}x"
    return f"{value:.0f}x"


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
    if "combined" in args.plot_modes:
        plot_paths["combined"] = {
            "pdf": str(paths["combined_pdf"]),
            "png": str(paths["combined_png"]),
        }
    if "combined_with_avg" in args.plot_modes:
        plot_paths["combined_with_avg"] = {
            "pdf": str(paths["combined_with_avg_pdf"]),
            "png": str(paths["combined_with_avg_png"]),
        }
        plot_paths["intro_tsaving"] = {
            "pdf": str(paths["intro_tsaving_pdf"]),
            "png": str(paths["intro_tsaving_png"]),
        }
        plot_paths["intro_fscore"] = {
            "pdf": str(paths["intro_fscore_pdf"]),
            "png": str(paths["intro_fscore_png"]),
        }
        plot_paths["intro_accuracy"] = {
            "pdf": str(paths["intro_accuracy_pdf"]),
            "png": str(paths["intro_accuracy_png"]),
        }

    payload = {
        "config": {
            "datasets": datasets,
            "batch_sizes": batch_sizes,
            "prompt_mode": args.prompt_mode,
            "few_shot_pairs_per_class": args.few_shot_pairs_per_class,
            "few_shot_positive_pairs": args.few_shot_positive_pairs,
            "few_shot_negative_pairs": args.few_shot_negative_pairs,
            "openai_model": oracle.model,
            "token_source_mode": args.token_source,
            "estimation_model_version": ESTIMATION_MODEL_VERSION,
            "naive_edge_sample_size": NAIVE_EDGE_SAMPLE_SIZE,
            "normalization_reference_batch_size": 2,
            "saving_normalization": args.saving_normalization,
            "normalization_reference": normalization_reference_text(args),
            "phi_fingerprint": args.phi_fingerprint,
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
        paths["output_dir"] / f"{paths['csv'].stem}_pareto.pdf",
        paths["output_dir"] / f"{paths['csv'].stem}_pareto.png",
        paths["heatmap_pdf"],
        paths["heatmap_png"],
        paths["combined_pdf"],
        paths["combined_png"],
        paths["combined_with_avg_pdf"],
        paths["combined_with_avg_png"],
        paths["intro_tsaving_pdf"],
        paths["intro_tsaving_png"],
        paths["intro_fscore_pdf"],
        paths["intro_fscore_png"],
        paths["intro_accuracy_pdf"],
        paths["intro_accuracy_png"],
    ]
    for plot_path in removable:
        if plot_path.exists():
            plot_path.unlink()


def render_overview(rows, *, datasets, batch_sizes, args, paths):
    lookup = build_lookup(rows)
    fig, ax = plt.subplots(figsize=(4.05, 2.4))
    cost_model = plotted_cost_model(args)

    y_min = float("inf")
    for dataset in datasets:
        color = LINE_COLORS[dataset]
        marker = LINE_MARKERS[dataset]
        p50_values = np.array(
            [
                plotted_saving_factor(
                    lookup[(dataset, batch_size, cost_model, "p50")],
                    args=args,
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
    y_max = 1.05
    for dataset in datasets:
        dataset_values = [
            plotted_saving_factor(
                lookup[(dataset, batch_size, cost_model, "p50")],
                args=args,
            )
            for batch_size in batch_sizes
        ]
        dataset_max = max(dataset_values)
        dataset_min = min(dataset_values)
        y_max = max(y_max, dataset_max)
        y_min = min(y_min, dataset_min)

    ax.set_xticks(batch_sizes)
    if y_min == float("inf"):
        y_min = 0.0
    padding = max(0.05, (y_max - y_min) * 0.08)
    ax.set_ylim(max(0.0, y_min - padding), y_max + padding)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(saving_tick_formatter))
    ax.grid(True, which="major", axis="both")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel(saving_axis_label(args))
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
        [
            [experiment_lookup.get((dataset, batch_size), {}).get("f1", np.nan) for batch_size in batch_sizes]
            for dataset in datasets
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(4.05, 2.4))
    masked_matrix = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked_matrix, aspect="auto", cmap=plt.cm.RdYlGn, vmin=0.5, vmax=1.0)
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
            if np.isnan(value):
                continue
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


def load_augmented_fscore_lookup(*, args, oracle, datasets, batch_sizes):
    model_name = sanitize_label(oracle.model)
    candidate_paths = [
        Path(args.output_dir) / f"experiment_batch_LLM,{model_name},augmented-camera30,b2,seed0.csv",
        Path("results/plot_LLM") / f"experiment_batch_LLM,{model_name},augmented-camera30,b2,seed0.csv",
        Path("results/batch_LLM_augmented") / f"experiment_batch_LLM,{model_name},augmented-camera30,b2,seed0.csv",
    ]

    source_rows = None
    for path in candidate_paths:
        if path.exists():
            source_rows = load_experiment_rows(path)
            break
    if source_rows is None:
        raise RuntimeError("Combined plot requires the augmented batch-LLM CSV in results/plot_LLM or results/batch_LLM_augmented.")

    lookup = {}
    for row in source_rows:
        key = (row["dataset"], row["batch_size"])
        if row["dataset"] in datasets and row["batch_size"] in batch_sizes:
            lookup[key] = row["f1"]

    return lookup


def load_augmented_metric_rows(*, args, oracle, datasets, batch_sizes):
    model_name = sanitize_label(oracle.model)
    candidate_paths = [
        Path(args.output_dir) / f"experiment_batch_LLM,{model_name},augmented-camera30,b2,seed0.csv",
        Path("results/plot_LLM") / f"experiment_batch_LLM,{model_name},augmented-camera30,b2,seed0.csv",
        Path("results/batch_LLM_augmented") / f"experiment_batch_LLM,{model_name},augmented-camera30,b2,seed0.csv",
    ]
    source_rows = None
    for path in candidate_paths:
        if path.exists():
            source_rows = load_experiment_rows(path)
            break
    if source_rows is None:
        raise RuntimeError("Combined plot requires the augmented batch-LLM CSV in results/plot_LLM or results/batch_LLM_augmented.")

    full_rows = {}
    for row in source_rows:
        key = (row["dataset"], row["batch_size"])
        if row["dataset"] in datasets and row["batch_size"] in batch_sizes:
            full_rows[key] = row
    return full_rows


def combined_with_avg_data(rows, *, datasets, batch_sizes, args, oracle):
    lookup = build_lookup(rows)
    fscore_lookup = load_augmented_fscore_lookup(
        args=args,
        oracle=oracle,
        datasets=datasets,
        batch_sizes=batch_sizes,
    )
    cost_model = plotted_cost_model(args)
    x_positions = np.arange(len(batch_sizes))
    saving_matrix = np.array(
        [
            [
                plotted_saving_factor(
                    lookup[(dataset, batch_size, cost_model, "p50")],
                    args=args,
                )
                for batch_size in batch_sizes
            ]
            for dataset in datasets
        ],
        dtype=float,
    )
    fscore_matrix = np.array(
        [[fscore_lookup.get((dataset, batch_size), np.nan) for batch_size in batch_sizes] for dataset in datasets],
        dtype=float,
    )
    return {
        "x_positions": x_positions,
        "saving_matrix": saving_matrix,
        "saving_mean": np.mean(saving_matrix, axis=0),
        "saving_std": np.std(saving_matrix, axis=0),
        "fscore_matrix": fscore_matrix,
    }


def accuracy_from_experiment_row(row: dict[str, object]) -> float:
    queried = int(float(row["queried_pair_events"]))
    gt_duplicates = int(float(row["gt_duplicate_pairs"]))
    tp = int(float(row["correctly_identified_duplicate_pairs"]))
    predicted_positive = int(float(row["predicted_positive_pairs"]))
    fn = gt_duplicates - tp
    fp = predicted_positive - tp
    tn = queried - tp - fn - fp
    if queried <= 0:
        return float("nan")
    return (tp + tn) / queried


def intro_fscore_cmap():
    cmap = colors.LinearSegmentedColormap.from_list(
        "intro_fscore",
        [
            "#f6caca",
            "#fde6e6",
            "#ffffff",
        ],
    )
    cmap.set_bad(color="#f4f4f4")
    return cmap


def render_intro_tsaving(data, *, batch_sizes, args, paths):
    x_positions = data["x_positions"]
    saving_mean = data["saving_mean"]
    saving_std = data["saving_std"]
    lower_band = np.maximum(0.0, saving_mean - saving_std)
    upper_band = saving_mean + saving_std

    fig, ax = plt.subplots(figsize=(2.25, 1.2))
    ax.fill_between(
        x_positions,
        lower_band,
        upper_band,
        color="#c7c7c7",
        alpha=0.55,
        linewidth=0.0,
    )
    ax.plot(
        x_positions,
        saving_mean,
        marker="o",
        color="#111111",
        linewidth=1.8,
        markersize=5.5,
        markeredgewidth=1.0,
        markerfacecolor="white",
        markeredgecolor="#111111",
    )
    y_min = float(np.min(lower_band)) if lower_band.size else 0.0
    y_max = max(1.05, float(np.max(upper_band)) if upper_band.size else 1.05)
    padding = max(0.05, (y_max - y_min) * 0.08)
    ax.set_ylim(max(0.0, y_min - padding), y_max + padding)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(batch_size) for batch_size in batch_sizes])
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(saving_tick_formatter))
    ax.grid(True, which="major", axis="both")
    ax.set_title(saving_axis_label(args))
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("")
    fig.savefig(paths["intro_tsaving_pdf"], bbox_inches="tight")
    fig.savefig(paths["intro_tsaving_png"], bbox_inches="tight")
    plt.close(fig)


def render_intro_fscore(data, *, datasets, batch_sizes, paths):
    fscore_matrix = data["fscore_matrix"]
    fig, ax = plt.subplots(figsize=(2.55, 1.2))
    cmap = intro_fscore_cmap()
    norm = colors.Normalize(vmin=0.5, vmax=1.0, clip=True)
    ax.imshow(np.ma.masked_invalid(fscore_matrix), aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(batch_sizes)))
    ax.set_xticklabels([str(batch_size) for batch_size in batch_sizes])
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels([pretty_dataset_name(dataset) for dataset in datasets])
    ax.set_title("Matching detection F-score")
    ax.set_xlabel("Batch Size")

    for row_index in range(fscore_matrix.shape[0]):
        for col_index in range(fscore_matrix.shape[1]):
            value = fscore_matrix[row_index, col_index]
            if np.isnan(value):
                continue
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="#111111",
                fontsize=7.5,
            )

    fig.savefig(paths["intro_fscore_pdf"], bbox_inches="tight")
    fig.savefig(paths["intro_fscore_png"], bbox_inches="tight")
    plt.close(fig)


def render_intro_accuracy(*, args, oracle, datasets, batch_sizes, paths):
    metric_rows = load_augmented_metric_rows(
        args=args,
        oracle=oracle,
        datasets=datasets,
        batch_sizes=batch_sizes,
    )
    accuracy_matrix = np.array(
        [
            [
                accuracy_from_experiment_row(metric_rows[(dataset, batch_size)])
                if (dataset, batch_size) in metric_rows
                else np.nan
                for batch_size in batch_sizes
            ]
            for dataset in datasets
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(2.55, 1.2))
    cmap = intro_fscore_cmap()
    norm = colors.Normalize(vmin=0.5, vmax=1.0, clip=True)
    ax.imshow(np.ma.masked_invalid(accuracy_matrix), aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(batch_sizes)))
    ax.set_xticklabels([str(batch_size) for batch_size in batch_sizes])
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels([pretty_dataset_name(dataset) for dataset in datasets])
    ax.set_title("Matching detection Accuracy")
    ax.set_xlabel("Batch Size")

    for row_index in range(accuracy_matrix.shape[0]):
        for col_index in range(accuracy_matrix.shape[1]):
            value = accuracy_matrix[row_index, col_index]
            if np.isnan(value):
                continue
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="#111111",
                fontsize=7.5,
            )

    fig.savefig(paths["intro_accuracy_pdf"], bbox_inches="tight")
    fig.savefig(paths["intro_accuracy_png"], bbox_inches="tight")
    plt.close(fig)


def render_combined(rows, *, datasets, batch_sizes, args, paths, oracle):
    lookup = build_lookup(rows)
    fscore_lookup = load_augmented_fscore_lookup(
        args=args,
        oracle=oracle,
        datasets=datasets,
        batch_sizes=batch_sizes,
    )
    cost_model = plotted_cost_model(args)

    x_positions = np.arange(len(batch_sizes))
    saving_series = {}
    y_min = float("inf")
    y_max = 1.05
    for dataset in datasets:
        values = np.array(
            [
                plotted_saving_factor(
                    lookup[(dataset, batch_size, cost_model, "p50")],
                    args=args,
                )
                for batch_size in batch_sizes
            ],
            dtype=float,
        )
        saving_series[dataset] = values
        y_min = min(y_min, float(np.min(values)))
        y_max = max(y_max, float(np.max(values)))

    fscore_matrix = np.array(
        [[fscore_lookup.get((dataset, batch_size), np.nan) for batch_size in batch_sizes] for dataset in datasets],
        dtype=float,
    )

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(4.1, 1.8),
        gridspec_kw={"width_ratios": [1.3, 1.5]},
    )

    legend_handles = []
    for dataset in datasets:
        handle = ax_left.plot(
            x_positions,
            saving_series[dataset],
            marker=LINE_MARKERS[dataset],
            color="#111111",
            linewidth=1.8,
            markersize=5.5,
            markeredgewidth=1.0,
            markerfacecolor="white",
            markeredgecolor="#111111",
            label=pretty_dataset_name(dataset),
        )[0]
        legend_handles.append(handle)

    if y_min == float("inf"):
        y_min = 0.0
    padding = max(0.05, (y_max - y_min) * 0.08)
    ax_left.set_ylim(max(0.0, y_min - padding), y_max + padding)
    ax_left.set_xticks(x_positions)
    ax_left.set_xticklabels([str(batch_size) for batch_size in batch_sizes])
    ax_left.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax_left.yaxis.set_major_formatter(ticker.FuncFormatter(saving_tick_formatter))
    ax_left.grid(True, which="major", axis="both")
    ax_left.set_title(saving_axis_label(args))
    ax_left.set_xlabel("Batch Size")
    ax_left.set_ylabel("")

    cmap = colors.LinearSegmentedColormap.from_list("bw_fscore", ["#d9d9d9", "#000000"])
    cmap.set_bad(color="#f1f1f1")
    norm = colors.Normalize(vmin=0.5, vmax=1.0, clip=True)
    ax_right.imshow(np.ma.masked_invalid(fscore_matrix), aspect="auto", cmap=cmap, norm=norm)
    ax_right.set_xticks(np.arange(len(batch_sizes)))
    ax_right.set_xticklabels([str(batch_size) for batch_size in batch_sizes])
    ax_right.set_yticks(np.arange(len(datasets)))
    ax_right.set_yticklabels([pretty_dataset_name(dataset) for dataset in datasets])
    ax_right.set_title("F-score")
    ax_right.set_xlabel("Batch Size")

    for row_index in range(fscore_matrix.shape[0]):
        for col_index in range(fscore_matrix.shape[1]):
            value = fscore_matrix[row_index, col_index]
            if np.isnan(value):
                continue
            text_color = "white" if value >= 0.82 else "#111111"
            ax_right.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7.5,
            )

    fig.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        ncol=len(legend_handles),
        loc="lower left",
        bbox_to_anchor=(0.12, 0.70, 0.86, 0.0),
        mode="expand",
        borderaxespad=0.0,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.5,
    )

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.32, top=0.62, wspace=0.32)
    left_pos = ax_left.get_position()
    right_pos = ax_right.get_position()
    fig.text(
        (left_pos.x0 + left_pos.x1) / 2,
        0.02,
        "(a)",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    fig.text((right_pos.x0 + right_pos.x1) / 2, 0.02, "(b)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.savefig(paths["combined_pdf"], bbox_inches="tight")
    fig.savefig(paths["combined_png"], bbox_inches="tight")
    plt.close(fig)


def render_combined_with_avg(rows, *, datasets, batch_sizes, args, paths, oracle):
    data = combined_with_avg_data(
        rows,
        datasets=datasets,
        batch_sizes=batch_sizes,
        args=args,
        oracle=oracle,
    )
    x_positions = data["x_positions"]
    saving_mean = data["saving_mean"]
    saving_std = data["saving_std"]
    lower_band = np.maximum(0.0, saving_mean - saving_std)
    upper_band = saving_mean + saving_std
    fscore_matrix = data["fscore_matrix"]

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(4.1, 1.8),
        gridspec_kw={"width_ratios": [1.3, 1.5]},
    )

    ax_left.fill_between(
        x_positions,
        lower_band,
        upper_band,
        color="#c7c7c7",
        alpha=0.55,
        linewidth=0.0,
        label="±1 std.",
    )
    ax_left.plot(
        x_positions,
        saving_mean,
        marker="o",
        color="#111111",
        linewidth=1.8,
        markersize=5.5,
        markeredgewidth=1.0,
        markerfacecolor="white",
        markeredgecolor="#111111",
        label="Avg.",
    )

    y_min = float(np.min(lower_band)) if lower_band.size else 0.0
    y_max = max(1.05, float(np.max(upper_band)) if upper_band.size else 1.05)
    padding = max(0.05, (y_max - y_min) * 0.08)
    ax_left.set_ylim(max(0.0, y_min - padding), y_max + padding)
    ax_left.set_xticks(x_positions)
    ax_left.set_xticklabels([str(batch_size) for batch_size in batch_sizes])
    ax_left.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax_left.yaxis.set_major_formatter(ticker.FuncFormatter(saving_tick_formatter))
    ax_left.grid(True, which="major", axis="both")
    ax_left.set_title(saving_axis_label(args))
    ax_left.set_xlabel("Batch Size")
    ax_left.set_ylabel("")

    cmap = colors.LinearSegmentedColormap.from_list("bw_fscore", ["#d9d9d9", "#000000"])
    cmap.set_bad(color="#f1f1f1")
    norm = colors.Normalize(vmin=0.5, vmax=1.0, clip=True)
    ax_right.imshow(np.ma.masked_invalid(fscore_matrix), aspect="auto", cmap=cmap, norm=norm)
    ax_right.set_xticks(np.arange(len(batch_sizes)))
    ax_right.set_xticklabels([str(batch_size) for batch_size in batch_sizes])
    ax_right.set_yticks(np.arange(len(datasets)))
    ax_right.set_yticklabels([pretty_dataset_name(dataset) for dataset in datasets])
    ax_right.set_title("F-score")
    ax_right.set_xlabel("Batch Size")

    for row_index in range(fscore_matrix.shape[0]):
        for col_index in range(fscore_matrix.shape[1]):
            value = fscore_matrix[row_index, col_index]
            if np.isnan(value):
                continue
            text_color = "white" if value >= 0.82 else "#111111"
            ax_right.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7.5,
            )

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.32, top=0.62, wspace=0.32)
    left_pos = ax_left.get_position()
    right_pos = ax_right.get_position()
    fig.text(
        (left_pos.x0 + left_pos.x1) / 2,
        0.02,
        "(a)",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    fig.text((right_pos.x0 + right_pos.x1) / 2, 0.02, "(b)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.savefig(paths["combined_with_avg_pdf"], bbox_inches="tight")
    fig.savefig(paths["combined_with_avg_png"], bbox_inches="tight")
    plt.close(fig)
    render_intro_tsaving(data, batch_sizes=batch_sizes, args=args, paths=paths)
    render_intro_fscore(data, datasets=datasets, batch_sizes=batch_sizes, paths=paths)
    render_intro_accuracy(args=args, oracle=oracle, datasets=datasets, batch_sizes=batch_sizes, paths=paths)

def render_plots(rows, *, datasets, batch_sizes, args, paths, oracle):
    apply_publication_style()
    cleanup_legacy_plots(paths)
    if "overview" in args.plot_modes:
        render_overview(rows, datasets=datasets, batch_sizes=batch_sizes, args=args, paths=paths)
    if "heatmap" in args.plot_modes:
        render_heatmap(rows, datasets=datasets, batch_sizes=batch_sizes, args=args, paths=paths)
    if "combined" in args.plot_modes:
        render_combined(rows, datasets=datasets, batch_sizes=batch_sizes, args=args, paths=paths, oracle=oracle)
    if "combined_with_avg" in args.plot_modes:
        render_combined_with_avg(rows, datasets=datasets, batch_sizes=batch_sizes, args=args, paths=paths, oracle=oracle)


def matching_config(payload: dict[str, object], *, args, oracle, datasets, batch_sizes) -> bool:
    config = payload.get("config", {})
    saved_batch_sizes = config.get("batch_sizes")
    batch_sizes_match = saved_batch_sizes == batch_sizes
    if not batch_sizes_match and isinstance(saved_batch_sizes, list):
        batch_sizes_match = all(batch_size in saved_batch_sizes for batch_size in batch_sizes)
    return (
        config.get("datasets") == datasets
        and batch_sizes_match
        and config.get("prompt_mode") == args.prompt_mode
        and config.get("few_shot_pairs_per_class") == args.few_shot_pairs_per_class
        and config.get("few_shot_positive_pairs") == args.few_shot_positive_pairs
        and config.get("few_shot_negative_pairs") == args.few_shot_negative_pairs
        and config.get("openai_model") == oracle.model
        and config.get("token_source_mode") == args.token_source
        and config.get("estimation_model_version") == ESTIMATION_MODEL_VERSION
        and config.get("phi_fingerprint") == args.phi_fingerprint
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
    phi_payload = load_phi_payload(args.phi_json)
    phi_config = resolve_phi_config(phi_payload, datasets)
    batch_sizes = resolve_batch_sizes(args, phi_config, datasets)
    args.phi_fingerprint = phi_fingerprint(phi_config, datasets)
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
            all_rows.extend(build_rows_for_dataset(state, oracle, args, phi_config[dataset], batch_sizes))

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
        if mode == "combined":
            print(
                f"Saved combined plot to {paths['combined_pdf']} and {paths['combined_png']}.",
                flush=True,
            )
        if mode == "combined_with_avg":
            print(
                f"Saved combined-with-avg plot to {paths['combined_with_avg_pdf']} and {paths['combined_with_avg_png']}.",
                flush=True,
            )


if __name__ == "__main__":
    main()
