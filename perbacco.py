import csv
import itertools
import json
import math
import os
import random
import statistics
import time
from argparse import ArgumentParser
from pathlib import Path

import networkx as nx
import pandas as pd

from class_pERbacco import class_entity, q_rec_k, r_rec_k, read_graph


k_minimum_queries = 3


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
    )
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--alg_community", type=str, choices=["louvain", "leiden", "lpa", "infomap", "False"])
    parser.add_argument("--lambda_w")
    parser.add_argument("--mu_benefit", type=str, choices=["brspecial", "brmean", "brmax"])
    parser.add_argument("--optimal", type=str)
    parser.add_argument("--synth_precision")
    parser.add_argument("--oracle_backend", type=str, choices=["groundtruth", "openai"], default="groundtruth")
    parser.add_argument("--openai_model", type=str, default=None)
    parser.add_argument("--prompt_mode", type=str, choices=["zero-shot", "few-shot"], default="zero-shot")
    parser.add_argument("--max_llm_calls", type=int, default=None)
    return parser.parse_args()


def sanitize_label(value):
    return str(value).replace("/", "-").replace(" ", "-").replace(":", "-")


def compute_max_query(df_ground_truth, batch_size):
    graph_opt = nx.Graph()
    for _, row in df_ground_truth.iterrows():
        left, right = row
        graph_opt.add_edge(left, right, weight=0, max_weight=0)

    components = list(nx.connected_components(graph_opt))
    component_sizes = [len(component) for component in sorted(components, key=len, reverse=True)]

    minimum_number_queries = 0
    for size in component_sizes:
        minimum_number_queries += q_rec_k(size, batch_size)

    list_rest = []
    for size in component_sizes:
        rest = r_rec_k(size, batch_size)
        if rest > 1:
            list_rest.append(rest)

    lower_minimum_number_queries = minimum_number_queries + math.ceil(sum(list_rest) / batch_size)
    return lower_minimum_number_queries * k_minimum_queries


def build_output_path(
    dname,
    batch_size,
    alg_community,
    lambda_w,
    mu_benefit,
    optimal,
    synth_precision,
    oracle_backend="groundtruth",
    openai_model=None,
    prompt_mode="zero-shot",
    max_llm_calls=None,
):
    directory = Path("results") / dname
    if oracle_backend == "openai":
        method_label = "LLM-pERbacco"
        if alg_community == "False":
            method_label = "LLM-pERbac" if mu_benefit == "brmean" else "LLM-Online"
        if optimal == "True":
            method_label = "LLM-SubOpt"

        parts = [f"{dname}_{method_label}", str(batch_size)]
        if alg_community != "False":
            parts.extend([alg_community[:3], str(lambda_w)])
        parts.extend([sanitize_label(openai_model or "default"), prompt_mode])
        if max_llm_calls is not None:
            parts.append(f"cap{max_llm_calls}")
        return directory / f"{','.join(parts)}.csv"

    if optimal == "True":
        return directory / f"{dname}_suboptimal,{batch_size}.csv"

    if "synth" in dname:
        return directory / f"{dname},{synth_precision},{batch_size},{alg_community[:3]},{mu_benefit}.csv"

    if alg_community != "False":
        return directory / f"{dname}_pERbacco,{batch_size},{alg_community[:3]},{lambda_w}.csv"

    if mu_benefit == "brmean":
        return directory / f"{dname}_pERbac,{batch_size}.csv"

    return directory / f"{dname}_Online,{batch_size}.csv"


def count_csv_rows(path):
    if not path.exists():
        return None
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def determine_expected_calls(args, effective_max_query):
    baseline_path = build_output_path(
        args.dataset,
        args.batch_size,
        args.alg_community,
        args.lambda_w,
        args.mu_benefit,
        args.optimal,
        args.synth_precision,
        oracle_backend="groundtruth",
    )
    baseline_rows = count_csv_rows(baseline_path)
    if baseline_rows is None:
        return effective_max_query, None
    return min(effective_max_query, baseline_rows), baseline_path


def print_llm_estimate(perbacco, args, effective_max_query):
    if perbacco.oracle_client is None:
        return

    expected_calls, baseline_path = determine_expected_calls(args, effective_max_query)
    batch_estimates = perbacco.build_estimation_batches(args.batch_size)
    experiment_estimate = perbacco.oracle_client.estimate_experiment_tokens(expected_calls, batch_estimates)

    print(
        "LLM preflight:",
        f"model={perbacco.oracle_client.model}",
        f"prompt_mode={perbacco.oracle_client.prompt_mode}",
        f"hard_call_cap={effective_max_query}",
        f"expected_calls={expected_calls}",
        flush=True,
    )
    if baseline_path is not None:
        print(f"Using baseline row count from {baseline_path} to estimate expected calls.", flush=True)

    for label in ["low", "avg", "high"]:
        estimate = batch_estimates[label]
        totals = experiment_estimate["scenarios"][label]
        print(
            f"  {label}: source={estimate['source']} "
            f"batch_input={estimate['input_tokens']} batch_output={estimate['output_tokens']} "
            f"total_input={totals['input_tokens']} total_output={totals['output_tokens']} "
            f"total_tokens={totals['total_tokens']}",
            flush=True,
        )


def write_time_summary(args, perbacco, total_time, number_query, number_query_first_part, partial_time, ran_second_part):
    if args.optimal == "True" or number_query == 0 or not ran_second_part:
        return

    time_per_query = total_time / number_query
    with open("time.txt", "a") as handle:
        if args.alg_community == "False":
            if perbacco.mu_benefit == "brmax":
                handle.write(
                    f"{args.dataset}_{args.batch_size} Online: time per query = {time_per_query:.3f}, "
                    f"total time = {total_time:.3f}, total queries = {int(number_query)}\n"
                )
            else:
                handle.write(
                    f"{args.dataset}_{args.batch_size} pERbac: time per query = {time_per_query:.3f}, "
                    f"total time = {total_time:.3f}, total queries = {int(number_query)}\n"
                )
        else:
            second_part_queries = max(0, number_query - number_query_first_part)
            if second_part_queries > 0:
                partial_time_per_query = partial_time / second_part_queries
                handle.write(
                    f"{args.dataset}_{args.batch_size} pERbacco: time per query second part = "
                    f"{partial_time_per_query:.3f}, partial time = {partial_time:.3f}, "
                    f"partial queries = {int(second_part_queries)}\n"
                )
            handle.write(
                f"{args.dataset}_{args.batch_size} pERbacco: time per query = {time_per_query:.3f}, "
                f"total time = {total_time:.3f}, total queries = {int(number_query)}\n"
            )


def append_result(
    results,
    args,
    perbacco,
    ideal_tracker,
    number_query,
    effective_max_query,
    kind,
    time_query,
    metrics,
    delta_true_positive_pairs,
    delta_predicted_pairs,
    batch,
):
    query_stats = perbacco.last_query_stats or {}
    progress = number_query / effective_max_query if effective_max_query else 1.0
    ideal_metrics = ideal_tracker.compute_metrics() if ideal_tracker is not None else None
    ideal_recall = ideal_metrics["recall"] if ideal_metrics is not None else None
    ideal_precision = ideal_metrics["precision"] if ideal_metrics is not None else None
    recall_gap_vs_ideal = (
        metrics["recall"] - ideal_recall if ideal_recall is not None else None
    )
    precision_gap_vs_ideal = (
        metrics["precision"] - ideal_precision if ideal_precision is not None else None
    )
    ideal_gap_text = ""
    if recall_gap_vs_ideal is not None and precision_gap_vs_ideal is not None:
        ideal_gap_text = (
            f"recall_gap_vs_ideal={recall_gap_vs_ideal:+.3f}"
            f" precision_gap_vs_ideal={precision_gap_vs_ideal:+.3f}"
        )
    ideal_gap_suffix = f" {ideal_gap_text}" if ideal_gap_text else ""

    print(
        f"{number_query}/{effective_max_query} {args.dataset} {args.batch_size} {kind} "
        f"progress={progress:.1%} temperature={round(perbacco.temperature, 2)} "
        f"recall={metrics['recall']:.3f} precision={metrics['precision']:.3f} "
        f"tp_pairs={metrics['true_positive_pairs']} predicted_pairs={metrics['predicted_pairs']} "
        f"{ideal_gap_suffix} "
        f"delta_tp={delta_true_positive_pairs} delta_pred={delta_predicted_pairs} "
        f"tokens={query_stats.get('llm_input_tokens', 0)}/{query_stats.get('llm_output_tokens', 0)}/"
        f"{query_stats.get('llm_total_tokens', 0)} "
        f"time={time_query:.3f} batch={sorted(batch)}",
        flush=True,
    )

    results.loc[len(results)] = {
        "number_query": number_query,
        "kind": kind,
        "temperature": round(perbacco.temperature, 2),
        "recall": round(metrics["recall"], 6),
        "precision": round(metrics["precision"], 6),
        "total_match": metrics["true_positive_pairs"],
        "predicted_pairs": metrics["predicted_pairs"],
        "true_positive_pairs": metrics["true_positive_pairs"],
        "len_df_benefit": len(perbacco.df_benefit),
        "progressive_recall": delta_true_positive_pairs,
        "predicted_delta_pairs": delta_predicted_pairs,
        "progress_percent": round(progress * 100, 2),
        "llm_input_tokens": query_stats.get("llm_input_tokens", 0),
        "llm_output_tokens": query_stats.get("llm_output_tokens", 0),
        "llm_total_tokens": query_stats.get("llm_total_tokens", 0),
        "ideal_recall": round(ideal_recall, 6) if ideal_recall is not None else None,
        "ideal_precision": round(ideal_precision, 6) if ideal_precision is not None else None,
        "recall_gap_vs_ideal": round(recall_gap_vs_ideal, 6) if recall_gap_vs_ideal is not None else None,
        "precision_gap_vs_ideal": round(precision_gap_vs_ideal, 6) if precision_gap_vs_ideal is not None else None,
        "time": time_query,
    }

    return metrics


def save_auxiliary_lists(args, perbacco):
    list_size = []
    list_match = []
    total_size = 0
    total_match = 0

    if len(perbacco.list_community) > 1:
        for community in perbacco.list_community:
            size, match, _match_star = perbacco.info_plot_community(community, plot=False)
            total_size += size
            total_match += match
            list_match.append(total_match)
            list_size.append(total_size)

    prefix_lists = f"{args.batch_size},{args.alg_community},{args.lambda_w}"
    lists = {
        "probmatch": perbacco.list_prob_match,
        "probnomatch": perbacco.list_prob_no_match,
        "benefitmatch": perbacco.list_benefit_match,
        "benefitnomatch": perbacco.list_benefit_no_match,
        "listmatch": list_match,
        "listsize": list_size,
        "sum_heavy": perbacco.sum_heavy_comm,
        "number_heavy": perbacco.number_heavy,
    }

    directory = Path("results") / "lists_match" / args.dataset
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{prefix_lists}.json").open("w") as handle:
        json.dump(lists, handle)


def maybe_build_optimal_graph(df_ground_truth, batch_size):
    graph_opt = nx.Graph()
    for _, row in df_ground_truth.iterrows():
        left, right = row
        graph_opt.add_edge(left, right, weight=0, max_weight=0)

    components = list(nx.connected_components(graph_opt))
    components_sorted = sorted(components, key=len, reverse=True)
    constant = int(batch_size * (batch_size - 1) / 2 * len(components_sorted[0]))
    count_component = 0
    for component in components_sorted:
        count_component += 1
        for left, right in itertools.combinations(component, 2):
            weight = round(1 + 1 / (count_component * constant), 15)
            graph_opt[left][right]["max_weight"] = weight
            graph_opt[left][right]["weight"] = weight
    return graph_opt


def main():
    args = parse_args()

    if args.max_llm_calls is not None and args.max_llm_calls <= 0:
        raise SystemExit("--max_llm_calls must be positive when provided.")

    lambda_w = args.lambda_w
    if lambda_w != "False":
        lambda_w = float(lambda_w)

    synth_precision = args.synth_precision
    if synth_precision != "False":
        synth_precision = float(synth_precision)

    df_ground_truth, graph = read_graph(args.dataset, synth_precision)
    max_query = compute_max_query(df_ground_truth, args.batch_size)
    effective_max_query = max_query
    if args.max_llm_calls is not None:
        effective_max_query = min(max_query, args.max_llm_calls)

    random.seed(42)
    print("PRINT max_query for", args.dataset, "and batch_size", args.batch_size, max_query, flush=True)
    if effective_max_query != max_query:
        print(f"Applying query cap: {effective_max_query}", flush=True)

    if args.optimal == "True":
        graph = maybe_build_optimal_graph(df_ground_truth, args.batch_size)
        args.alg_community = "False"
        lambda_w = "False"
        args.mu_benefit = "brmax"

    perbacco = class_entity(
        args.dataset,
        graph,
        df_ground_truth,
        args.batch_size,
        args.alg_community,
        args.mu_benefit,
        lambda_w,
        oracle_backend=args.oracle_backend,
        openai_model=args.openai_model,
        prompt_mode=args.prompt_mode,
    )

    perbacco.create_list_community()
    ideal_tracker = None
    if args.oracle_backend == "openai":
        ideal_tracker = class_entity(
            args.dataset,
            graph,
            df_ground_truth,
            args.batch_size,
            args.alg_community,
            args.mu_benefit,
            lambda_w,
            oracle_backend="groundtruth",
            openai_model=args.openai_model,
            prompt_mode=args.prompt_mode,
        )
    if args.oracle_backend == "openai":
        print_llm_estimate(perbacco, args, effective_max_query)

    results = pd.DataFrame(
        columns=[
            "number_query",
            "kind",
            "temperature",
            "recall",
            "precision",
            "total_match",
            "predicted_pairs",
            "true_positive_pairs",
            "len_df_benefit",
            "progressive_recall",
            "predicted_delta_pairs",
            "progress_percent",
            "llm_input_tokens",
            "llm_output_tokens",
            "llm_total_tokens",
            "ideal_recall",
            "ideal_precision",
            "recall_gap_vs_ideal",
            "precision_gap_vs_ideal",
            "time",
        ]
    )

    number_query = 0
    old_true_positive_pairs = 0
    old_predicted_pairs = 0
    list_match_community_batch = []
    list_match_representative_batch = []
    perbacco.temperature = perbacco.batch_size
    total_time = 0
    nodes = set(perbacco.graph.nodes())

    if len(perbacco.list_community) > 1:
        print("WITH COMMUNITY, record ratio is:", perbacco.sum_heavy_comm / len(nodes), flush=True)
        perbacco.with_community = "T"
    else:
        print("WO COMMUNITY", flush=True)
        perbacco.list_community = [nodes]
        perbacco.create_dict_comm()
        perbacco.with_community = "F"
    if ideal_tracker is not None:
        ideal_tracker.list_community = [set(community) for community in perbacco.list_community]
        ideal_tracker.create_dict_comm()
        ideal_tracker.with_community = perbacco.with_community

    start_time = time.perf_counter()
    first_part = False
    time_query_start = time.perf_counter()

    if perbacco.with_community == "T":
        first_part = True
        for community_nodes in perbacco.list_community[:-1]:
            if number_query >= effective_max_query:
                break

            perbacco.query(community_nodes, "skip")
            if ideal_tracker is not None:
                ideal_tracker.query(community_nodes, "skip")
            current = set(community_nodes)

            while len(current) >= perbacco.batch_size and number_query < effective_max_query:
                subgraph = perbacco.graph.subgraph(current).copy()
                vertex_weight_sum = {
                    node: sum(data["weight"] for _, _, data in subgraph.edges(node, data=True))
                    for node in subgraph.nodes()
                }
                sorted_edges = sorted(subgraph.edges(data=True), key=lambda item: item[2]["weight"], reverse=True)
                dict_selected_vertices = {node: 0 for node in subgraph.nodes()}
                query_comm = perbacco.greedy_heaviest_subgraph(
                    subgraph,
                    vertex_weight_sum,
                    sorted_edges,
                    dict_selected_vertices,
                )
                current = current.difference(query_comm)

                number_query += 1
                perbacco.query(query_comm, "entity")
                if ideal_tracker is not None:
                    ideal_tracker.query(query_comm, "entity")
                time_query_stop = time.perf_counter()
                time_query = time_query_stop - time_query_start
                time_query_start = time.perf_counter()
                metrics = perbacco.compute_metrics()

                metrics = append_result(
                    results,
                    args,
                    perbacco,
                    ideal_tracker,
                    number_query,
                    effective_max_query,
                    "COMMUNITY",
                    time_query,
                    metrics,
                    metrics["true_positive_pairs"] - old_true_positive_pairs,
                    metrics["predicted_pairs"] - old_predicted_pairs,
                    query_comm,
                )
                delta_predicted_pairs = metrics["predicted_pairs"] - old_predicted_pairs
                list_match_community_batch.append(delta_predicted_pairs)
                old_true_positive_pairs = metrics["true_positive_pairs"]
                old_predicted_pairs = metrics["predicted_pairs"]

                set_higher_temperature = perbacco.compute_entity_higher_temperature()
                while len(set_higher_temperature) == perbacco.batch_size and number_query < effective_max_query:
                    current = current.difference(set(set_higher_temperature))
                    number_query += 1

                    perbacco.query(set_higher_temperature, "entity")
                    if ideal_tracker is not None:
                        ideal_tracker.query(set_higher_temperature, "entity")
                    time_query_stop = time.perf_counter()
                    time_query = time_query_stop - time_query_start
                    time_query_start = time.perf_counter()
                    metrics = perbacco.compute_metrics()

                    metrics = append_result(
                        results,
                        args,
                        perbacco,
                        ideal_tracker,
                        number_query,
                        effective_max_query,
                        "CURRENT",
                        time_query,
                        metrics,
                        metrics["true_positive_pairs"] - old_true_positive_pairs,
                        metrics["predicted_pairs"] - old_predicted_pairs,
                        set_higher_temperature,
                    )
                    delta_predicted_pairs = metrics["predicted_pairs"] - old_predicted_pairs
                    list_match_representative_batch.append(delta_predicted_pairs)
                    threshold_temp = (
                        statistics.mean(list_match_community_batch)
                        if len(list_match_community_batch) > 0
                        else perbacco.batch_size / 2
                    )

                    if list_match_representative_batch[-1] <= threshold_temp:
                        perbacco.temperature *= 2

                    set_higher_temperature = perbacco.compute_entity_higher_temperature()
                    old_true_positive_pairs = metrics["true_positive_pairs"]
                    old_predicted_pairs = metrics["predicted_pairs"]

                perbacco.temperature *= 1 - 1 / perbacco.batch_size

    number_query_first_part = number_query
    stop_time = time.perf_counter()
    total_time += stop_time - start_time

    if number_query < effective_max_query:
        all_nodes = list(perbacco.list_community[-1])
        perbacco.query(all_nodes, "last")
        if ideal_tracker is not None:
            ideal_tracker.query(all_nodes, "last")

    perbacco.temperature = 0
    start_time = time.perf_counter()
    second_part = False
    random.seed(42)
    perbacco.df_benefit = perbacco.df_benefit.sort_values(by="benefit", ascending=False, ignore_index=False)
    if ideal_tracker is not None:
        ideal_tracker.df_benefit = ideal_tracker.df_benefit.sort_values(by="benefit", ascending=False, ignore_index=False)

    while number_query < effective_max_query and len(perbacco.df_benefit) > 0:
        second_part = True
        number_query += 1
        batch = perbacco.compute_entity_higher_temperature()
        perbacco.query(batch, "entity")
        if ideal_tracker is not None:
            ideal_tracker.query(batch, "entity")
        time_query_stop = time.perf_counter()
        time_query = time_query_stop - time_query_start
        time_query_start = time.perf_counter()
        metrics = perbacco.compute_metrics()

        metrics = append_result(
            results,
            args,
            perbacco,
            ideal_tracker,
            number_query,
            effective_max_query,
            "CURRENT",
            time_query,
            metrics,
            metrics["true_positive_pairs"] - old_true_positive_pairs,
            metrics["predicted_pairs"] - old_predicted_pairs,
            batch,
        )
        list_match_community_batch.append(metrics["predicted_pairs"] - old_predicted_pairs)
        old_true_positive_pairs = metrics["true_positive_pairs"]
        old_predicted_pairs = metrics["predicted_pairs"]

    stop_time = time.perf_counter()
    partial_time = stop_time - start_time
    total_time += partial_time
    write_time_summary(args, perbacco, total_time, number_query, number_query_first_part, partial_time, second_part)

    save_auxiliary_lists(args, perbacco)

    output_path = build_output_path(
        args.dataset,
        args.batch_size,
        args.alg_community,
        lambda_w,
        args.mu_benefit,
        args.optimal,
        synth_precision,
        oracle_backend=args.oracle_backend,
        openai_model=args.openai_model,
        prompt_mode=args.prompt_mode,
        max_llm_calls=args.max_llm_calls if args.oracle_backend == "openai" else None,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    if len(results) > 0 and ideal_tracker is not None:
        final_metrics = perbacco.compute_metrics()
        final_ideal_metrics = ideal_tracker.compute_metrics()
        final_recall_gap = final_metrics["recall"] - final_ideal_metrics["recall"]
        final_precision_gap = final_metrics["precision"] - final_ideal_metrics["precision"]
        if pd.notna(final_recall_gap) and pd.notna(final_precision_gap):
            print(
                "Final gaps vs ideal:",
                f"recall_gap_vs_ideal={final_recall_gap:+.6f}",
                f"precision_gap_vs_ideal={final_precision_gap:+.6f}",
                flush=True,
            )
    print(f"Saved results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
