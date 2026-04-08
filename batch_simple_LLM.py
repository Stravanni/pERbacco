from __future__ import annotations

import itertools
import json
import random
import re
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd

import llm_config
from class_pERbacco import normalize_identifier
from oracle_LL import OpenAIEntityOracle


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "cora",
            "camera",
            "funding",
            "voters",
            "cddb",
            "restaurant",
            "wdc20",
            "wdc50",
            "wdc80",
            "census",
            "synth_250",
            "synth_1000",
            "synth_5000",
            "synth_10000",
        ],
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-mode", type=str, choices=["zero-shot", "few-shot"], default="few-shot")
    parser.add_argument("--few-shot-pairs-per-class", type=int, default=3)
    parser.add_argument("--openai-model", type=str, default=None)
    parser.add_argument("--synth_precision", type=str, default="False")
    parser.add_argument("--results-path", type=str, default=None)
    return parser.parse_args()


def sanitize_label(value):
    return str(value).replace("/", "-").replace(" ", "-").replace(":", "-")


def default_results_path(args) -> Path:
    directory = Path("results") / args.dataset
    directory.mkdir(parents=True, exist_ok=True)
    parts = [
        f"{args.dataset}_batch_simple_LLM",
        f"b{args.batch_size}",
        f"x{args.num_batches}",
        sanitize_label(args.openai_model or llm_config.OPENAI_MODEL),
        args.prompt_mode,
        f"seed{args.seed}",
    ]
    if "synth" in args.dataset:
        parts.append(f"synth{args.synth_precision}")
    return directory / f"{','.join(parts)}.json"


def load_dataset_records(dataset_name: str) -> tuple[dict[int | str, dict[str, str]], list[str]]:
    path = Path("datasets") / dataset_name / f"{dataset_name}.csv"
    df_records = pd.read_csv(path, dtype=str).fillna("")
    if len(df_records.columns) == 0:
        return {}, []

    first_column = df_records.columns[0]
    if first_column != "id":
        df_records = df_records.rename(columns={first_column: "id"})

    field_names = [column for column in df_records.columns if column != "id"]
    records = {}
    for _, row in df_records.iterrows():
        record = {column: str(value).strip() for column, value in row.items()}
        records[normalize_identifier(record["id"])] = record
    return records, field_names


def load_raw_graph(dataset_name: str, synth_precision: str):
    groundtruth_path = Path("datasets") / dataset_name / "groundtruth.csv"
    df_ground_truth = pd.read_csv(groundtruth_path, dtype=str).fillna("")
    df_ground_truth.columns = ["id1", "id2"]
    df_ground_truth["id1"] = df_ground_truth["id1"].map(normalize_identifier)
    df_ground_truth["id2"] = df_ground_truth["id2"].map(normalize_identifier)

    if synth_precision == "False":
        graph_path = Path("similarity_graph") / f"{dataset_name}.parquet"
    else:
        graph_path = Path("similarity_graph") / f"synth_precision_{synth_precision}" / f"{dataset_name}.parquet"

    if not graph_path.exists():
        raise RuntimeError(f"Missing similarity graph: {graph_path}")

    df_graph = pd.read_parquet(graph_path).rename(columns={"w": "weight"})
    df_graph["id1"] = df_graph["id1"].map(normalize_identifier)
    df_graph["id2"] = df_graph["id2"].map(normalize_identifier)

    graph = nx.from_pandas_edgelist(
        df_graph,
        source="id1",
        target="id2",
        edge_attr="weight",
        create_using=nx.Graph(),
    )
    missing_nodes = set(df_ground_truth.values.flatten()) - set(graph.nodes())
    graph.add_nodes_from(missing_nodes)
    return df_ground_truth, graph


def build_match_lookup(df_ground_truth: pd.DataFrame) -> dict[int | str, set[int | str]]:
    lookup: dict[int | str, set[int | str]] = defaultdict(set)
    for _, row in df_ground_truth.iterrows():
        left = normalize_identifier(row["id1"])
        right = normalize_identifier(row["id2"])
        lookup[left].add(right)
        lookup[right].add(left)
    return lookup


def build_ground_truth_pairs(df_ground_truth: pd.DataFrame) -> set[frozenset[int | str]]:
    pairs = set()
    for _, row in df_ground_truth.iterrows():
        left = normalize_identifier(row["id1"])
        right = normalize_identifier(row["id2"])
        if left != right:
            pairs.add(frozenset((left, right)))
    return pairs


def pair_is_match(left, right, match_lookup) -> bool:
    return right in match_lookup.get(left, set())


def normalize_text(text):
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def token_list(text, limit):
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split()[:limit]


def numeric_signature(text):
    return re.findall(r"\d+(?:\.\d+)?", str(text or ""))[:6]


def infer_prompt_profile(dataset_name, field_names):
    field_set = set(field_names)
    if dataset_name in {"camera", "wdc80"}:
        return "product_catalog"
    if {"brand", "model"} <= field_set or {"brand", "title", "description"} <= field_set:
        return "product_catalog"
    return "generic_tabular"


def build_generic_entity_payload(record_id, records_by_id, prompt_profile):
    record = records_by_id[record_id]
    sanitized_record = {
        key: value
        for key, value in record.items()
        if key != "id" and str(value).strip()
    }
    payload = {
        "entity_id": str(record_id),
        "entity_size": 1,
        "records": [sanitized_record],
    }
    if prompt_profile == "product_catalog":
        title_source = sanitized_record.get("title") or sanitized_record.get("description") or ""
        payload.update(
            {
                "brand_normalized": normalize_text(sanitized_record.get("brand", "")),
                "model_normalized": normalize_text(sanitized_record.get("model", "")),
                "title_tokens": token_list(title_source, 20),
                "description_tokens": token_list(sanitized_record.get("description", ""), 24),
                "numeric_signatures": {
                    key: numeric_signature(value)
                    for key, value in sanitized_record.items()
                    if numeric_signature(value)
                },
            }
        )
    return payload


def build_batch_entities(batch_nodes, records_by_id, prompt_profile):
    entities = []
    alias_to_record_id = {}
    for index, record_id in enumerate(batch_nodes, start=1):
        alias = f"E{index}"
        payload = build_generic_entity_payload(record_id, records_by_id, prompt_profile)
        payload["entity_id"] = alias
        entities.append(payload)
        alias_to_record_id[alias] = record_id
    return entities, alias_to_record_id


def build_pair_example(name, left_record, right_record, is_match):
    answer = {
        "clusters": [{"cluster_id": "c1", "entity_ids": ["A", "B"]}]
        if is_match
        else [
            {"cluster_id": "c1", "entity_ids": ["A"]},
            {"cluster_id": "c2", "entity_ids": ["B"]},
        ]
    }
    return {
        "name": name,
        "entities": [
            {"entity_id": "A", "entity_size": 1, "records": [left_record]},
            {"entity_id": "B", "entity_size": 1, "records": [right_record]},
        ],
        "answer": answer,
    }


def compute_components(graph):
    import networkx as nx

    components = [sorted(component) for component in nx.connected_components(graph)]
    components.sort(key=lambda component: (component[0], len(component)))
    return components


def sample_batch_from_components(components, batch_size, rng):
    shuffled_indices = list(range(len(components)))
    rng.shuffle(shuffled_indices)

    selected_nodes = []
    selected_components = []
    total = 0
    for component_index in shuffled_indices:
        component_nodes = components[component_index]
        selected_components.append(component_index)
        selected_nodes.extend(component_nodes)
        total += len(component_nodes)
        if total >= batch_size:
            overflow = total - batch_size
            cut_nodes = []
            if overflow > 0:
                latest_nodes = component_nodes[:]
                keep_count = len(latest_nodes) - overflow
                kept_nodes = rng.sample(latest_nodes, keep_count)
                kept_nodes_set = set(kept_nodes)
                cut_nodes = [node for node in latest_nodes if node not in kept_nodes_set]
                selected_nodes = selected_nodes[:-len(latest_nodes)] + kept_nodes
            return {
                "nodes": sorted(selected_nodes),
                "component_indices": selected_components,
                "overflow_cut_nodes": sorted(cut_nodes),
            }
    raise RuntimeError(f"Unable to build a batch of size {batch_size}; graph only covered {total} nodes.")


def mine_positive_examples(ground_truth_pairs, excluded_nodes, count, rng):
    allowed = [
        tuple(sorted(pair))
        for pair in ground_truth_pairs
        if pair.isdisjoint(excluded_nodes)
    ]
    if len(allowed) <= count:
        return allowed
    return rng.sample(allowed, count)


def mine_negative_examples(graph, match_lookup, excluded_nodes, count, rng):
    graph_negatives = []
    for left, right in graph.edges():
        if left in excluded_nodes or right in excluded_nodes:
            continue
        if not pair_is_match(left, right, match_lookup):
            graph_negatives.append((left, right))
    if len(graph_negatives) >= count:
        return rng.sample(graph_negatives, count)

    negatives = list(graph_negatives)
    allowed_nodes = [node for node in graph.nodes() if node not in excluded_nodes]
    seen = {frozenset(pair) for pair in negatives}
    max_attempts = max(100, 20 * count)
    attempts = 0
    while len(negatives) < count and len(allowed_nodes) >= 2 and attempts < max_attempts:
        left, right = rng.sample(allowed_nodes, 2)
        pair = frozenset((left, right))
        attempts += 1
        if pair in seen:
            continue
        if pair_is_match(left, right, match_lookup):
            continue
        negatives.append((left, right))
        seen.add(pair)
    return negatives


def build_few_shot_examples(records_by_id, positive_pairs, negative_pairs):
    examples = []
    for index, (left, right) in enumerate(positive_pairs, start=1):
        examples.append(
            build_pair_example(
                f"Positive {index}",
                {k: v for k, v in records_by_id[left].items() if k != "id" and v},
                {k: v for k, v in records_by_id[right].items() if k != "id" and v},
                True,
            )
        )
    for index, (left, right) in enumerate(negative_pairs, start=1):
        examples.append(
            build_pair_example(
                f"Negative {index}",
                {k: v for k, v in records_by_id[left].items() if k != "id" and v},
                {k: v for k, v in records_by_id[right].items() if k != "id" and v},
                False,
            )
        )
    return examples


def score_batch(nodes, predicted_positive_pairs, match_lookup):
    tp = 0
    fp = 0
    fn = 0
    total_events = 0
    for left, right in itertools.combinations(nodes, 2):
        predicted_match = frozenset((left, right)) in predicted_positive_pairs
        actual_match = pair_is_match(left, right, match_lookup)
        total_events += 1
        if predicted_match and actual_match:
            tp += 1
        elif predicted_match and not actual_match:
            fp += 1
        elif (not predicted_match) and actual_match:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "queried_pair_events": total_events,
        "predicted_positive_pairs": tp + fp,
        "actual_positive_pairs": tp + fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def aggregate_metrics(batch_metrics):
    tp = sum(item["tp"] for item in batch_metrics)
    fp = sum(item["fp"] for item in batch_metrics)
    fn = sum(item["fn"] for item in batch_metrics)
    queried_pair_events = sum(item["queried_pair_events"] for item in batch_metrics)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "queried_pair_events": queried_pair_events,
        "predicted_positive_pairs": tp + fp,
        "actual_positive_pairs": tp + fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    args = parse_args()

    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2")
    if args.num_batches < 1:
        raise ValueError("--num-batches must be at least 1")

    df_ground_truth, graph = load_raw_graph(args.dataset, args.synth_precision)
    records_by_id, field_names = load_dataset_records(args.dataset)
    match_lookup = build_match_lookup(df_ground_truth)
    ground_truth_pairs = build_ground_truth_pairs(df_ground_truth)
    components = compute_components(graph)
    prompt_profile = infer_prompt_profile(args.dataset, field_names)

    print(
        f"{args.dataset} nodes, matches, edges {len(graph.nodes())} {len(df_ground_truth)} {len(graph.edges())}",
        flush=True,
    )

    if len(graph.nodes()) < args.batch_size:
        raise RuntimeError(
            f"Dataset {args.dataset} has only {len(graph.nodes())} nodes, smaller than batch size {args.batch_size}."
        )

    oracle = OpenAIEntityOracle(
        model=args.openai_model,
        prompt_mode=args.prompt_mode,
    )

    results_path = Path(args.results_path) if args.results_path else default_results_path(args)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    batch_results = []
    batch_metrics = []

    for batch_index in range(args.num_batches):
        batch_rng = random.Random(args.seed + batch_index)
        batch_selection = sample_batch_from_components(components, args.batch_size, batch_rng)
        batch_nodes = batch_selection["nodes"]

        few_shot_examples = []
        positive_examples = []
        negative_examples = []
        if args.prompt_mode == "few-shot":
            excluded_nodes = set(batch_nodes)
            example_rng = random.Random(args.seed + 100000 + batch_index)
            positive_examples = mine_positive_examples(
                ground_truth_pairs,
                excluded_nodes,
                args.few_shot_pairs_per_class,
                example_rng,
            )
            negative_examples = mine_negative_examples(
                graph,
                match_lookup,
                excluded_nodes,
                args.few_shot_pairs_per_class,
                example_rng,
            )
            few_shot_examples = build_few_shot_examples(records_by_id, positive_examples, negative_examples)

        entities, alias_to_record_id = build_batch_entities(batch_nodes, records_by_id, prompt_profile)
        response = oracle.resolve_batch(
            entities,
            prompt_profile=prompt_profile,
            dataset_name=args.dataset,
            field_names=field_names,
            few_shot_examples=few_shot_examples,
            prompt_mode=args.prompt_mode,
        )

        predicted_positive_pairs = set()
        for cluster in response["clusters"]:
            cluster_nodes = [alias_to_record_id[entity_id] for entity_id in cluster["entity_ids"]]
            for left, right in itertools.combinations(cluster_nodes, 2):
                predicted_positive_pairs.add(frozenset((left, right)))

        metrics = score_batch(batch_nodes, predicted_positive_pairs, match_lookup)
        batch_metrics.append(metrics)

        batch_results.append(
            {
                "batch_index": batch_index,
                "seed": args.seed + batch_index,
                "component_indices": batch_selection["component_indices"],
                "nodes": batch_nodes,
                "entity_aliases": {alias: str(record_id) for alias, record_id in alias_to_record_id.items()},
                "overflow_cut_nodes": batch_selection["overflow_cut_nodes"],
                "few_shot_positive_pairs": [list(map(str, pair)) for pair in positive_examples],
                "few_shot_negative_pairs": [list(map(str, pair)) for pair in negative_examples],
                "few_shot_example_count": len(few_shot_examples),
                "response_id": response.get("response_id", ""),
                "usage": response["usage"],
                "clusters": response["clusters"],
                "metrics": metrics,
            }
        )

        print(
            f"batch {batch_index + 1}/{args.num_batches}: "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"tokens={response['usage']['llm_total_tokens']}",
            flush=True,
        )

    aggregate = aggregate_metrics(batch_metrics)
    total_usage = {
        "llm_input_tokens": sum(item["usage"]["llm_input_tokens"] for item in batch_results),
        "llm_output_tokens": sum(item["usage"]["llm_output_tokens"] for item in batch_results),
        "llm_total_tokens": sum(item["usage"]["llm_total_tokens"] for item in batch_results),
    }

    artifact = {
        "config": {
            "dataset": args.dataset,
            "batch_size": args.batch_size,
            "num_batches": args.num_batches,
            "seed": args.seed,
            "prompt_mode": args.prompt_mode,
            "few_shot_pairs_per_class": args.few_shot_pairs_per_class,
            "openai_model": oracle.model,
            "prompt_profile": prompt_profile,
            "synth_precision": args.synth_precision,
            "field_names": field_names,
            "results_path": str(results_path),
        },
        "graph_summary": {
            "nodes": len(graph.nodes()),
            "edges": len(graph.edges()),
            "connected_components": len(components),
            "largest_component_size": max(len(component) for component in components),
        },
        "aggregate_metrics": aggregate,
        "aggregate_usage": total_usage,
        "batches": batch_results,
    }

    results_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=True) + "\n")

    print(
        f"final: precision={aggregate['precision']:.4f} "
        f"recall={aggregate['recall']:.4f} "
        f"f1={aggregate['f1']:.4f} "
        f"queried_pair_events={aggregate['queried_pair_events']} "
        f"results={results_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
