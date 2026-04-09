from __future__ import annotations

import itertools
import json
import random
import re
import csv
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
    parser.add_argument(
        "--prompt-profile",
        type=str,
        choices=["bibliographic", "generic_tabular", "product_catalog", "camera_catalog", "funding_award"],
        default=None,
    )
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


def summary_csv_path(results_path: Path) -> Path:
    return results_path.with_suffix(".csv")


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


MARKETPLACE_NOISE_PATTERNS = [
    r"\bbest price in india\b",
    r"\bspecs? and review\b",
    r"\bprice[\s-]?hunt\b",
    r"\bvalid in\b",
    r"\bebay\b",
    r"\bamazon\b",
    r"\bflipkart\b",
    r"\bbuy now\b",
    r"\bshopping\b",
    r"\bfor sale\b",
]
CAMERA_SPEC_FIELDS = ("mp", "optical_zoom", "digital_zoom", "screen_size", "type")
FUNDING_COARSE_LOCATION_TOKENS = {
    "bronx",
    "brooklyn",
    "manhattan",
    "queens",
    "staten",
    "island",
    "new",
    "york",
    "ny",
}


def strip_marketplace_boilerplate(text):
    lowered = str(text or "").lower()
    parts = []
    for piece in re.split(r"[|]", lowered):
        cleaned = piece
        for pattern in MARKETPLACE_NOISE_PATTERNS:
            cleaned = re.sub(pattern, " ", cleaned)
        cleaned = re.sub(r"\b(?:delhi|mumbai|bangalore|hyderabad|chennai|kolkata|ahmedabad|surat)\b", " ", cleaned)
        cleaned = re.sub(r"\b\d{4}\b", " ", cleaned)
        cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
        cleaned = " ".join(cleaned.split())
        if cleaned:
            parts.append(cleaned)
    return " ".join(parts)


def normalized_nonempty(value):
    normalized = normalize_text(value)
    return normalized if normalized else ""


def record_without_id(record):
    return {
        key: value
        for key, value in record.items()
        if key != "id" and str(value).strip()
    }


def camera_brand_model_signature(sanitized_record):
    brand = normalized_nonempty(sanitized_record.get("brand", ""))
    model = normalized_nonempty(sanitized_record.get("model", ""))
    if brand and model:
        return f"{brand}::{model}"
    return brand or model


def camera_spec_signatures(sanitized_record):
    signatures = {}
    for field_name in CAMERA_SPEC_FIELDS:
        value = normalized_nonempty(sanitized_record.get(field_name, ""))
        if value:
            signatures[field_name] = value
    return signatures


def normalize_amount_signature(value):
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        numeric_value = float(text)
    except ValueError:
        return normalized_nonempty(text)
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return f"{numeric_value:.2f}".rstrip("0").rstrip(".")


def funding_address_core(address):
    normalized = normalized_nonempty(address)
    if not normalized:
        return ""
    normalized = re.sub(r"\b\d{5}(?: \d{4})?\b", " ", normalized)
    normalized = re.sub(r"\bny\b", " ", normalized)
    normalized = " ".join(normalized.split())
    return normalized


def funding_address_specificity(address):
    normalized = normalized_nonempty(address)
    if not normalized:
        return "missing"
    tokens = normalized.split()
    has_digit = any(char.isdigit() for char in normalized)
    if has_digit:
        return "specific"
    if len(tokens) <= 4 and set(tokens).issubset(FUNDING_COARSE_LOCATION_TOKENS):
        return "coarse"
    if len(tokens) <= 3:
        return "coarse"
    return "specific"


def funding_record_signatures(sanitized_record):
    return {
        "organization_normalized": normalized_nonempty(sanitized_record.get("name", "")),
        "address_normalized": normalized_nonempty(sanitized_record.get("address", "")),
        "address_core_normalized": funding_address_core(sanitized_record.get("address", "")),
        "address_specificity": funding_address_specificity(sanitized_record.get("address", "")),
        "agency_normalized": normalized_nonempty(sanitized_record.get("agency", "")),
        "year_signature": normalized_nonempty(sanitized_record.get("year", "")),
        "amount_signature": normalize_amount_signature(sanitized_record.get("amount", "")),
        "name_tokens": token_list(sanitized_record.get("name", ""), 12),
        "address_tokens": token_list(sanitized_record.get("address", ""), 16),
    }


def camera_overlap_tokens(record):
    model_tokens = set(token_list(record.get("model", ""), 8))
    description_tokens = set(token_list(strip_marketplace_boilerplate(record.get("description", "")), 18))
    return model_tokens | description_tokens


def classify_camera_negative(left_record, right_record):
    left_brand = normalized_nonempty(left_record.get("brand", ""))
    right_brand = normalized_nonempty(right_record.get("brand", ""))
    left_model = normalized_nonempty(left_record.get("model", ""))
    right_model = normalized_nonempty(right_record.get("model", ""))

    if left_brand and left_brand == right_brand and left_model and right_model and left_model != right_model:
        return 1

    overlap = camera_overlap_tokens(left_record) & camera_overlap_tokens(right_record)
    if overlap:
        return 2

    return 3


def classify_funding_negative(left_record, right_record):
    left_signatures = funding_record_signatures(left_record)
    right_signatures = funding_record_signatures(right_record)

    same_name = (
        left_signatures["organization_normalized"]
        and left_signatures["organization_normalized"] == right_signatures["organization_normalized"]
    )
    same_agency = (
        left_signatures["agency_normalized"]
        and left_signatures["agency_normalized"] == right_signatures["agency_normalized"]
    )
    year_differs = (
        left_signatures["year_signature"]
        and right_signatures["year_signature"]
        and left_signatures["year_signature"] != right_signatures["year_signature"]
    )
    amount_differs = (
        left_signatures["amount_signature"]
        and right_signatures["amount_signature"]
        and left_signatures["amount_signature"] != right_signatures["amount_signature"]
    )
    weak_address = (
        left_signatures["address_specificity"] != "specific"
        or right_signatures["address_specificity"] != "specific"
    )

    if same_name and same_agency and year_differs and amount_differs and weak_address:
        return 1
    if same_name and same_agency and weak_address:
        return 2
    if same_name:
        return 3
    return 4


def classify_funding_positive(left_record, right_record):
    left_signatures = funding_record_signatures(left_record)
    right_signatures = funding_record_signatures(right_record)

    same_name = (
        left_signatures["organization_normalized"]
        and left_signatures["organization_normalized"] == right_signatures["organization_normalized"]
    )
    same_agency = (
        left_signatures["agency_normalized"]
        and left_signatures["agency_normalized"] == right_signatures["agency_normalized"]
    )
    same_amount = (
        left_signatures["amount_signature"]
        and left_signatures["amount_signature"] == right_signatures["amount_signature"]
    )
    specific_address_match = (
        left_signatures["address_normalized"]
        and left_signatures["address_normalized"] == right_signatures["address_normalized"]
        and left_signatures["address_specificity"] == "specific"
        and right_signatures["address_specificity"] == "specific"
    )
    specific_address_core_match = (
        left_signatures["address_core_normalized"]
        and left_signatures["address_core_normalized"] == right_signatures["address_core_normalized"]
        and left_signatures["address_specificity"] == "specific"
        and right_signatures["address_specificity"] == "specific"
    )
    both_weak_address = (
        left_signatures["address_specificity"] != "specific"
        and right_signatures["address_specificity"] != "specific"
    )

    if same_name and (specific_address_match or specific_address_core_match):
        return 1
    if same_name and same_agency and same_amount and both_weak_address:
        return 2
    if same_name:
        return 3
    return 4


def collect_camera_graph_negatives(graph, match_lookup, excluded_nodes, count, records_by_id, rng):
    bucket_limits = {1: max(count * 2, count), 2: max(count * 3, count), 3: max(count * 4, count)}
    buckets = {1: [], 2: [], 3: []}
    seen = set()

    for left, right in graph.edges():
        if left in excluded_nodes or right in excluded_nodes:
            continue
        if pair_is_match(left, right, match_lookup):
            continue
        normalized_pair = frozenset((left, right))
        if normalized_pair in seen:
            continue

        hardness = classify_camera_negative(records_by_id[left], records_by_id[right])
        if len(buckets[hardness]) >= bucket_limits[hardness]:
            continue

        buckets[hardness].append((left, right))
        seen.add(normalized_pair)
        if (
            len(buckets[1]) >= min(count, bucket_limits[1])
            and len(buckets[1]) + len(buckets[2]) >= count
            and len(buckets[3]) >= min(count, bucket_limits[3])
        ):
            break

    negatives = []
    selected_seen = set()
    for bucket_index in (1, 2, 3):
        bucket = buckets[bucket_index][:]
        rng.shuffle(bucket)
        for pair in bucket:
            normalized_pair = frozenset(pair)
            if normalized_pair in selected_seen:
                continue
            negatives.append(pair)
            selected_seen.add(normalized_pair)
            if len(negatives) >= count:
                return negatives
    return negatives


def collect_funding_positive_examples(ground_truth_pairs, excluded_nodes, count, records_by_id, rng):
    buckets = {1: [], 2: [], 3: [], 4: []}
    for pair in ground_truth_pairs:
        if not pair.isdisjoint(excluded_nodes):
            continue
        left, right = tuple(sorted(pair))
        hardness = classify_funding_positive(records_by_id[left], records_by_id[right])
        buckets[hardness].append((left, right))

    positives = []
    seen = set()
    for bucket_index in (1, 2, 3, 4):
        bucket = buckets[bucket_index][:]
        rng.shuffle(bucket)
        for pair in bucket:
            normalized_pair = frozenset(pair)
            if normalized_pair in seen:
                continue
            positives.append(pair)
            seen.add(normalized_pair)
            if len(positives) >= count:
                return positives
    return positives


def collect_funding_graph_negatives(graph, match_lookup, excluded_nodes, count, records_by_id, rng):
    bucket_limits = {1: max(count * 3, count), 2: max(count * 3, count), 3: max(count * 4, count), 4: max(count * 2, count)}
    buckets = {1: [], 2: [], 3: [], 4: []}
    seen = set()

    for left, right in graph.edges():
        if left in excluded_nodes or right in excluded_nodes:
            continue
        if pair_is_match(left, right, match_lookup):
            continue
        normalized_pair = frozenset((left, right))
        if normalized_pair in seen:
            continue

        hardness = classify_funding_negative(records_by_id[left], records_by_id[right])
        if len(buckets[hardness]) >= bucket_limits[hardness]:
            continue

        buckets[hardness].append((left, right))
        seen.add(normalized_pair)
        if len(buckets[1]) >= count and len(buckets[2]) >= count:
            break

    negatives = []
    selected_seen = set()
    for bucket_index in (1, 2, 3, 4):
        bucket = buckets[bucket_index][:]
        rng.shuffle(bucket)
        for pair in bucket:
            normalized_pair = frozenset(pair)
            if normalized_pair in selected_seen:
                continue
            negatives.append(pair)
            selected_seen.add(normalized_pair)
            if len(negatives) >= count:
                return negatives
    return negatives


def infer_prompt_profile(dataset_name, field_names):
    field_set = set(field_names)
    if dataset_name in {"camera", "wdc80"}:
        return "product_catalog"
    if {"brand", "model"} <= field_set or {"brand", "title", "description"} <= field_set:
        return "product_catalog"
    return "generic_tabular"


def to_json_compatible(value):
    if isinstance(value, dict):
        return {str(key): to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(item) for item in value]
    if isinstance(value, set):
        return [to_json_compatible(item) for item in sorted(value, key=lambda item: str(item))]
    if hasattr(value, "item") and callable(value.item):
        try:
            return to_json_compatible(value.item())
        except (TypeError, ValueError):
            pass
    return value


def build_generic_entity_payload(record_id, records_by_id, prompt_profile):
    record = records_by_id[record_id]
    sanitized_record = record_without_id(record)
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
    if prompt_profile == "camera_catalog":
        payload.update(
            {
                "brand_normalized": normalized_nonempty(sanitized_record.get("brand", "")),
                "model_normalized": normalized_nonempty(sanitized_record.get("model", "")),
                "brand_model_signature": camera_brand_model_signature(sanitized_record),
                "model_tokens": token_list(sanitized_record.get("model", ""), 8),
                "description_core_tokens": token_list(
                    strip_marketplace_boilerplate(sanitized_record.get("description", "")),
                    24,
                ),
                "spec_signatures": camera_spec_signatures(sanitized_record),
            }
        )
    if prompt_profile == "funding_award":
        payload.update(funding_record_signatures(sanitized_record))
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


def build_pair_example(name, left_entity, right_entity, is_match):
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
        "entities": [left_entity, right_entity],
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


def mine_positive_examples(ground_truth_pairs, excluded_nodes, count, rng, *, prompt_profile=None, records_by_id=None):
    if prompt_profile == "funding_award" and records_by_id:
        positives = collect_funding_positive_examples(
            ground_truth_pairs,
            excluded_nodes,
            count,
            records_by_id,
            rng,
        )
        if len(positives) >= count:
            return positives
    allowed = [
        tuple(sorted(pair))
        for pair in ground_truth_pairs
        if pair.isdisjoint(excluded_nodes)
    ]
    if len(allowed) <= count:
        return allowed
    return rng.sample(allowed, count)


def mine_negative_examples(graph, match_lookup, excluded_nodes, count, rng, *, prompt_profile=None, records_by_id=None):
    if prompt_profile == "camera_catalog" and records_by_id:
        graph_negatives = collect_camera_graph_negatives(
            graph,
            match_lookup,
            excluded_nodes,
            count,
            records_by_id,
            rng,
        )
        if len(graph_negatives) >= count:
            return graph_negatives
    elif prompt_profile == "funding_award" and records_by_id:
        graph_negatives = collect_funding_graph_negatives(
            graph,
            match_lookup,
            excluded_nodes,
            count,
            records_by_id,
            rng,
        )
        if len(graph_negatives) >= count:
            return graph_negatives
    else:
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


def build_few_shot_examples(records_by_id, positive_pairs, negative_pairs, prompt_profile):
    examples = []
    for index, (left, right) in enumerate(positive_pairs, start=1):
        left_entity = build_generic_entity_payload(left, records_by_id, prompt_profile)
        right_entity = build_generic_entity_payload(right, records_by_id, prompt_profile)
        left_entity["entity_id"] = "A"
        right_entity["entity_id"] = "B"
        examples.append(
            build_pair_example(
                f"Positive {index}",
                left_entity,
                right_entity,
                True,
            )
        )
    for index, (left, right) in enumerate(negative_pairs, start=1):
        left_entity = build_generic_entity_payload(left, records_by_id, prompt_profile)
        right_entity = build_generic_entity_payload(right, records_by_id, prompt_profile)
        left_entity["entity_id"] = "A"
        right_entity["entity_id"] = "B"
        examples.append(
            build_pair_example(
                f"Negative {index}",
                left_entity,
                right_entity,
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
    prompt_profile = args.prompt_profile or infer_prompt_profile(args.dataset, field_names)

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
                prompt_profile=prompt_profile,
                records_by_id=records_by_id,
            )
            negative_examples = mine_negative_examples(
                graph,
                match_lookup,
                excluded_nodes,
                args.few_shot_pairs_per_class,
                example_rng,
                prompt_profile=prompt_profile,
                records_by_id=records_by_id,
            )
            few_shot_examples = build_few_shot_examples(
                records_by_id,
                positive_examples,
                negative_examples,
                prompt_profile,
            )

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
                "validation": response.get("validation", {"repaired": False}),
                "clusters": response["clusters"],
                "metrics": metrics,
            }
        )

        repair_suffix = ""
        if response.get("validation", {}).get("repaired"):
            repair_suffix = " repaired=1"
        print(
            f"batch {batch_index + 1}/{args.num_batches}: "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"tokens={response['usage']['llm_total_tokens']}"
            f"{repair_suffix}",
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
        "aggregate_validation": {
            "repaired_batches": sum(
                1 for item in batch_results if item.get("validation", {}).get("repaired")
            )
        },
        "batches": batch_results,
    }

    results_path.write_text(json.dumps(to_json_compatible(artifact), indent=2, ensure_ascii=True) + "\n")

    csv_path = summary_csv_path(results_path)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "batch_size",
                "num_batches",
                "seed",
                "prompt_mode",
                "few_shot_pairs_per_class",
                "openai_model",
                "prompt_profile",
                "synth_precision",
                "connected_components",
                "graph_nodes",
                "graph_edges",
                "queried_pair_events",
                "gt_duplicate_pairs",
                "correctly_identified_duplicate_pairs",
                "predicted_positive_pairs",
                "precision",
                "recall",
                "f1",
                "llm_input_tokens",
                "llm_output_tokens",
                "llm_total_tokens",
                "json_results_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": args.dataset,
                "batch_size": args.batch_size,
                "num_batches": args.num_batches,
                "seed": args.seed,
                "prompt_mode": args.prompt_mode,
                "few_shot_pairs_per_class": args.few_shot_pairs_per_class,
                "openai_model": oracle.model,
                "prompt_profile": prompt_profile,
                "synth_precision": args.synth_precision,
                "connected_components": len(components),
                "graph_nodes": len(graph.nodes()),
                "graph_edges": len(graph.edges()),
                "queried_pair_events": aggregate["queried_pair_events"],
                "gt_duplicate_pairs": aggregate["actual_positive_pairs"],
                "correctly_identified_duplicate_pairs": aggregate["tp"],
                "predicted_positive_pairs": aggregate["predicted_positive_pairs"],
                "precision": aggregate["precision"],
                "recall": aggregate["recall"],
                "f1": aggregate["f1"],
                "llm_input_tokens": total_usage["llm_input_tokens"],
                "llm_output_tokens": total_usage["llm_output_tokens"],
                "llm_total_tokens": total_usage["llm_total_tokens"],
                "json_results_path": str(results_path),
            }
        )

    print(
        f"final: gt_duplicate_pairs={aggregate['actual_positive_pairs']} "
        f"correctly_identified_duplicate_pairs={aggregate['tp']} "
        f"precision={aggregate['precision']:.4f} "
        f"recall={aggregate['recall']:.4f} "
        f"f1={aggregate['f1']:.4f} "
        f"queried_pair_events={aggregate['queried_pair_events']} "
        f"results={results_path} "
        f"csv={csv_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
