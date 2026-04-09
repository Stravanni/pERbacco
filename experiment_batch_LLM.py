from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import threading
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

import llm_config


REAL_DATASETS = ["camera", "cora", "funding", "voters", "wdc80"]
DEFAULT_BATCH_SIZES = [5, 10, 20, 50]
MASTER_FIELDNAMES = [
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
    "csv_results_path",
    "duration_seconds",
    "status",
]
INT_FIELDS = {
    "batch_size",
    "num_batches",
    "seed",
    "few_shot_pairs_per_class",
    "connected_components",
    "graph_nodes",
    "graph_edges",
    "queried_pair_events",
    "gt_duplicate_pairs",
    "correctly_identified_duplicate_pairs",
    "predicted_positive_pairs",
    "llm_input_tokens",
    "llm_output_tokens",
    "llm_total_tokens",
}
FLOAT_FIELDS = {
    "precision",
    "recall",
    "f1",
    "duration_seconds",
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None, choices=REAL_DATASETS)
    parser.add_argument("--batch-sizes", nargs="*", type=int, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--few-shot-pairs-per-class", type=int, default=10)
    parser.add_argument("--prompt-mode", type=str, choices=["zero-shot", "few-shot"], default="few-shot")
    parser.add_argument("--openai-model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="results/batch_LLM")
    parser.add_argument(
        "--plot-modes",
        nargs="*",
        default=["heatmap", "bars", "lines"],
        choices=["heatmap", "bars", "lines"],
    )
    return parser.parse_args()


def sanitize_label(value):
    return str(value).replace("/", "-").replace(" ", "-").replace(":", "-")


def resolve_datasets(requested: list[str] | None) -> list[str]:
    candidates = requested or REAL_DATASETS
    available = []
    for dataset in candidates:
        graph_path = Path("similarity_graph") / f"{dataset}.parquet"
        if graph_path.exists():
            available.append(dataset)
    if not available:
        raise RuntimeError("No requested datasets have a materialized top-level similarity graph.")
    return available


def experiment_stem(args, model_name: str) -> str:
    return ",".join(
        [
            "experiment_batch_LLM",
            sanitize_label(model_name),
            args.prompt_mode,
            f"fewshot{args.few_shot_pairs_per_class}x2",
            f"x{args.num_batches}",
            f"seed{args.seed}",
        ]
    )


def build_output_paths(args, model_name: str) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = experiment_stem(args, model_name)
    return {
        "output_dir": output_dir,
        "runs_dir": output_dir / "runs",
        "master_csv": output_dir / f"{stem}.csv",
        "master_json": output_dir / f"{stem}.json",
        "heatmap_pdf": output_dir / f"{stem}_heatmap.pdf",
        "heatmap_png": output_dir / f"{stem}_heatmap.png",
        "bars_pdf": output_dir / f"{stem}_bars.pdf",
        "bars_png": output_dir / f"{stem}_bars.png",
        "lines_pdf": output_dir / f"{stem}_lines.pdf",
        "lines_png": output_dir / f"{stem}_lines.png",
    }


def run_result_json_path(runs_dir: Path, dataset: str, batch_size: int, args, model_name: str) -> Path:
    dataset_dir = runs_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    filename = ",".join(
        [
            dataset,
            f"b{batch_size}",
            f"x{args.num_batches}",
            sanitize_label(model_name),
            args.prompt_mode,
            f"fewshot{args.few_shot_pairs_per_class}x2",
            f"seed{args.seed}",
        ]
    )
    return dataset_dir / f"{filename}.json"


def summary_csv_path(json_path: Path) -> Path:
    return json_path.with_suffix(".csv")


def config_key(config: dict[str, object]) -> tuple[str, int]:
    return str(config["dataset"]), int(config["batch_size"])


def sort_records(records: list[dict[str, object]], datasets: list[str], batch_sizes: list[int]) -> list[dict[str, object]]:
    dataset_order = {dataset: index for index, dataset in enumerate(datasets)}
    batch_order = {batch_size: index for index, batch_size in enumerate(batch_sizes)}
    return sorted(
        records,
        key=lambda item: (
            dataset_order[str(item["dataset"])],
            batch_order[int(item["batch_size"])],
        ),
    )


def parse_int(value):
    if value in ("", None):
        return None
    return int(float(value))


def parse_float(value):
    if value in ("", None):
        return None
    return float(value)


def normalize_record_row(
    row: dict[str, str],
    *,
    duration_seconds: float | None = None,
    default_status: str = "completed",
) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for field in MASTER_FIELDNAMES:
        if field == "csv_results_path":
            normalized[field] = row.get(field, "")
            continue
        if field == "status":
            normalized[field] = row.get(field) or default_status
            continue
        if field == "duration_seconds":
            value = row.get(field, "")
            normalized[field] = duration_seconds if duration_seconds is not None else parse_float(value)
            continue

        value = row.get(field, "")
        if field in INT_FIELDS:
            normalized[field] = parse_int(value)
        elif field in FLOAT_FIELDS:
            normalized[field] = parse_float(value)
        else:
            normalized[field] = value

    return normalized


def load_result_records(csv_path: Path, duration_seconds: float | None = None) -> list[dict[str, object]]:
    if not csv_path.exists():
        raise RuntimeError(f"Missing child summary CSV: {csv_path}")

    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    records = []
    for row in rows:
        normalized = normalize_record_row(row, duration_seconds=duration_seconds)
        if not normalized.get("csv_results_path"):
            normalized["csv_results_path"] = str(csv_path)
        records.append(normalized)
    return records


def load_completed_result(csv_path: Path, duration_seconds: float | None = None) -> dict[str, object]:
    records = load_result_records(csv_path, duration_seconds=duration_seconds)
    if len(records) != 1:
        raise RuntimeError(f"Expected exactly one row in {csv_path}, found {len(records)}")
    normalized = records[0]
    if not normalized.get("csv_results_path"):
        normalized["csv_results_path"] = str(csv_path)
    return normalized


def write_master_csv(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MASTER_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def mean_f1_by_batch_size(records: list[dict[str, object]], datasets: list[str], batch_sizes: list[int]) -> dict[int, float | None]:
    values = {batch_size: [] for batch_size in batch_sizes}
    for record in records:
        values[int(record["batch_size"])].append(float(record["f1"]))

    summary = {}
    for batch_size in batch_sizes:
        scores = values[batch_size]
        summary[batch_size] = None if not scores else float(np.mean(scores))
    return summary


def write_master_json(
    path: Path,
    *,
    args,
    datasets: list[str],
    batch_sizes: list[int],
    model_name: str,
    records: list[dict[str, object]],
    failed: list[dict[str, object]],
) -> None:
    completed_keys = {config_key(record) for record in records}
    pending = [
        {"dataset": dataset, "batch_size": batch_size}
        for dataset in datasets
        for batch_size in batch_sizes
        if (dataset, batch_size) not in completed_keys
    ]

    payload = {
        "config": {
            "datasets": datasets,
            "batch_sizes": batch_sizes,
            "num_batches": args.num_batches,
            "few_shot_pairs_per_class": args.few_shot_pairs_per_class,
            "prompt_mode": args.prompt_mode,
            "openai_model": model_name,
            "seed": args.seed,
            "workers": args.workers,
            "plot_modes": args.plot_modes,
        },
        "completed_configs": len(records),
        "failed_configs": len(failed),
        "pending_configs": pending,
        "mean_f1_by_batch_size": mean_f1_by_batch_size(records, datasets, batch_sizes),
        "results": records,
        "failed": failed,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def f1_matrix(records: list[dict[str, object]], datasets: list[str], batch_sizes: list[int]) -> np.ndarray:
    matrix = np.full((len(datasets) + 1, len(batch_sizes)), np.nan)
    lookup = {
        (str(record["dataset"]), int(record["batch_size"])): float(record["f1"])
        for record in records
    }
    for row_index, dataset in enumerate(datasets):
        for col_index, batch_size in enumerate(batch_sizes):
            value = lookup.get((dataset, batch_size))
            if value is not None:
                matrix[row_index, col_index] = value

    for col_index in range(len(batch_sizes)):
        column = matrix[: len(datasets), col_index]
        if np.all(np.isnan(column)):
            continue
        matrix[len(datasets), col_index] = float(np.nanmean(column))
    return matrix


def plot_title(args, model_name: str, partial: bool) -> str:
    state = "Partial" if partial else "Final"
    return (
        f"{state} LLM Batch ER F1\n"
        f"model={model_name}, few-shot={args.few_shot_pairs_per_class}+{args.few_shot_pairs_per_class}, "
        f"batches={args.num_batches}, seed={args.seed}"
    )


def render_heatmap(
    records: list[dict[str, object]],
    *,
    datasets: list[str],
    batch_sizes: list[int],
    args,
    model_name: str,
    pdf_path: Path,
    png_path: Path,
    title_text: str | None = None,
    footer_text: str | None = None,
) -> None:
    matrix = f1_matrix(records, datasets, batch_sizes)[: len(datasets), :]
    labels = datasets

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#f1f1f1")
    norm = colors.Normalize(vmin=0.5, vmax=1.0, clip=True)

    fig, ax = plt.subplots(figsize=(4.05, 2.4))
    image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(len(batch_sizes)))
    ax.set_xticklabels([str(batch_size) for batch_size in batch_sizes])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("F-score")
    ax.set_xlabel("Batch Size")

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if np.isnan(value):
                text = "pending"
                color = "#666666"
            else:
                text = f"{value:.3f}"
                color = "white" if value >= 0.8 else "#1f1f1f"
            ax.text(col_index, row_index, text, ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_grouped_bars(
    records: list[dict[str, object]],
    *,
    datasets: list[str],
    batch_sizes: list[int],
    args,
    model_name: str,
    pdf_path: Path,
    png_path: Path,
    title_text: str | None = None,
) -> None:
    matrix = f1_matrix(records, datasets, batch_sizes)
    labels = datasets + ["mean"]
    x = np.arange(len(batch_sizes))
    width = 0.12

    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    for index, label in enumerate(labels):
        offset = (index - (len(labels) - 1) / 2) * width
        values = matrix[index]
        style = {"linewidth": 1.0, "alpha": 0.9}
        if label == "mean":
            style.update({"color": "#222222", "edgecolor": "#111111"})
        bars = ax.bar(x + offset, values, width, label=label, **style)
        for bar, value in zip(bars, values, strict=False):
            if np.isnan(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(batch_size) for batch_size in batch_sizes])
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("F1")
    ax.set_title(title_text or plot_title(args, model_name, len(records) < len(datasets) * len(batch_sizes)))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_lines(
    records: list[dict[str, object]],
    *,
    datasets: list[str],
    batch_sizes: list[int],
    args,
    model_name: str,
    pdf_path: Path,
    png_path: Path,
    title_text: str | None = None,
) -> None:
    matrix = f1_matrix(records, datasets, batch_sizes)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))

    for row_index, dataset in enumerate(datasets):
        ax.plot(batch_sizes, matrix[row_index], marker="o", linewidth=1.5, alpha=0.9, label=dataset)
    ax.plot(
        batch_sizes,
        matrix[len(datasets)],
        marker="o",
        linewidth=2.8,
        color="#111111",
        label="mean",
    )

    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("F1")
    ax.set_title(title_text or plot_title(args, model_name, len(records) < len(datasets) * len(batch_sizes)))
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def update_artifacts(
    *,
    args,
    datasets: list[str],
    batch_sizes: list[int],
    model_name: str,
    records: list[dict[str, object]],
    failed: list[dict[str, object]],
    paths: dict[str, Path],
) -> None:
    sorted_records = sort_records(records, datasets, batch_sizes)
    write_master_csv(paths["master_csv"], sorted_records)
    write_master_json(
        paths["master_json"],
        args=args,
        datasets=datasets,
        batch_sizes=batch_sizes,
        model_name=model_name,
        records=sorted_records,
        failed=failed,
    )
    if "heatmap" in args.plot_modes:
        render_heatmap(
            sorted_records,
            datasets=datasets,
            batch_sizes=batch_sizes,
            args=args,
            model_name=model_name,
            pdf_path=paths["heatmap_pdf"],
            png_path=paths["heatmap_png"],
        )
    if "bars" in args.plot_modes:
        render_grouped_bars(
            sorted_records,
            datasets=datasets,
            batch_sizes=batch_sizes,
            args=args,
            model_name=model_name,
            pdf_path=paths["bars_pdf"],
            png_path=paths["bars_png"],
        )
    if "lines" in args.plot_modes:
        render_lines(
            sorted_records,
            datasets=datasets,
            batch_sizes=batch_sizes,
            args=args,
            model_name=model_name,
            pdf_path=paths["lines_pdf"],
            png_path=paths["lines_png"],
        )


def prefixed_print(lock: threading.Lock, prefix: str, text: str) -> None:
    with lock:
        print(f"{prefix} {text}", flush=True)


def partial_summary(records: list[dict[str, object]], batch_sizes: list[int]) -> str:
    means = mean_f1_by_batch_size(records, REAL_DATASETS, batch_sizes)
    parts = []
    for batch_size in batch_sizes:
        value = means[batch_size]
        if value is None:
            parts.append(f"b{batch_size}=pending")
        else:
            parts.append(f"b{batch_size}={value:.4f}")
    return ", ".join(parts)


def build_configs(args, datasets: list[str], paths: dict[str, Path], model_name: str) -> list[dict[str, object]]:
    configs = []
    for dataset in datasets:
        for batch_size in args.batch_sizes:
            json_path = run_result_json_path(paths["runs_dir"], dataset, batch_size, args, model_name)
            configs.append(
                {
                    "dataset": dataset,
                    "batch_size": batch_size,
                    "json_results_path": json_path,
                    "csv_results_path": summary_csv_path(json_path),
                }
            )
    return configs


def run_worker(config: dict[str, object], *, args, repo_root: Path, model_name: str, print_lock: threading.Lock):
    dataset = str(config["dataset"])
    batch_size = int(config["batch_size"])
    prefix = f"[{dataset} b={batch_size}]"
    child_json_path = Path(config["json_results_path"])
    child_csv_path = Path(config["csv_results_path"])

    command = [
        sys.executable,
        str(repo_root / "batch_simple_LLM.py"),
        "--dataset",
        dataset,
        "--batch-size",
        str(batch_size),
        "--num-batches",
        str(args.num_batches),
        "--prompt-mode",
        args.prompt_mode,
        "--few-shot-pairs-per-class",
        str(args.few_shot_pairs_per_class),
        "--seed",
        str(args.seed),
        "--results-path",
        str(child_json_path),
    ]
    if args.openai_model:
        command.extend(["--openai-model", args.openai_model])

    prefixed_print(print_lock, prefix, "starting")
    start = time.time()
    log_tail: list[str] = []
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    assert process.stdout is not None
    for line in process.stdout:
        text = line.rstrip()
        if not text:
            continue
        log_tail.append(text)
        log_tail = log_tail[-20:]
        prefixed_print(print_lock, prefix, text)

    return_code = process.wait()
    duration = time.time() - start
    if return_code != 0:
        return {
            "status": "failed",
            "dataset": dataset,
            "batch_size": batch_size,
            "returncode": return_code,
            "duration_seconds": duration,
            "json_results_path": str(child_json_path),
            "csv_results_path": str(child_csv_path),
            "log_tail": log_tail,
        }

    record = load_completed_result(child_csv_path, duration_seconds=duration)
    prefixed_print(print_lock, prefix, f"completed in {duration:.1f}s with f1={float(record['f1']):.4f}")
    return {"status": "completed", "record": record}


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    model_name = args.openai_model or llm_config.OPENAI_MODEL
    datasets = resolve_datasets(args.datasets)
    batch_sizes = sorted(set(args.batch_sizes))
    paths = build_output_paths(args, model_name)
    paths["runs_dir"].mkdir(parents=True, exist_ok=True)

    print(
        "experiment configuration: "
        f"datasets={datasets}, batch_sizes={batch_sizes}, prompt_mode={args.prompt_mode}, "
        f"few_shot_pairs_per_class={args.few_shot_pairs_per_class}, num_batches={args.num_batches}, "
        f"seed={args.seed}, workers={args.workers}, model={model_name}",
        flush=True,
    )

    configs = build_configs(args, datasets, paths, model_name)
    records: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    print_lock = threading.Lock()

    scheduled = []
    for config in configs:
        csv_path = Path(config["csv_results_path"])
        if csv_path.exists():
            try:
                record = load_completed_result(csv_path)
            except Exception as exc:  # noqa: BLE001
                print(f"[resume] ignoring unreadable existing result {csv_path}: {exc}", flush=True)
                scheduled.append(config)
                continue
            records.append(record)
            print(
                f"[resume] loaded existing result for {config['dataset']} b={config['batch_size']} "
                f"f1={float(record['f1']):.4f}",
                flush=True,
            )
        else:
            scheduled.append(config)

    update_artifacts(
        args=args,
        datasets=datasets,
        batch_sizes=batch_sizes,
        model_name=model_name,
        records=records,
        failed=failed,
        paths=paths,
    )

    total_configs = len(configs)
    if scheduled:
        print(f"launching {len(scheduled)} remaining configurations with {args.workers} workers", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_config = {
                executor.submit(
                    run_worker,
                    config,
                    args=args,
                    repo_root=repo_root,
                    model_name=model_name,
                    print_lock=print_lock,
                ): config
                for config in scheduled
            }
            for future in as_completed(future_to_config):
                result = future.result()
                if result["status"] == "completed":
                    records.append(result["record"])
                else:
                    failed.append(result)
                    print(
                        f"[failed] {result['dataset']} b={result['batch_size']} "
                        f"returncode={result['returncode']}",
                        flush=True,
                    )

                update_artifacts(
                    args=args,
                    datasets=datasets,
                    batch_sizes=batch_sizes,
                    model_name=model_name,
                    records=records,
                    failed=failed,
                    paths=paths,
                )
                print(
                    f"[partial] completed={len(records)}/{total_configs} "
                    f"failed={len(failed)} mean_f1_by_batch_size: {partial_summary(records, batch_sizes)}",
                    flush=True,
                )

    sorted_records = sort_records(records, datasets, batch_sizes)
    print(
        f"final summary: completed={len(sorted_records)}/{total_configs} "
        f"failed={len(failed)} mean_f1_by_batch_size: {partial_summary(sorted_records, batch_sizes)} "
        f"master_csv={paths['master_csv']} heatmap={paths['heatmap_pdf']}",
        flush=True,
    )

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
