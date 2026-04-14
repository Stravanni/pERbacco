from __future__ import annotations

import csv
import json
import os
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from experiment_batch_LLM import (
    REAL_DATASETS,
    load_result_records,
    mean_f1_by_batch_size,
    render_grouped_bars,
    render_heatmap,
    render_lines,
    sort_records,
    write_master_csv,
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--base-master-csv", type=str, required=True)
    parser.add_argument("--override-csvs", nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, default="results/batch_LLM_augmented")
    parser.add_argument("--output-stem", type=str, default=None)
    parser.add_argument("--title-text", type=str, default=None)
    parser.add_argument("--footer-text", type=str, default=None)
    parser.add_argument("--notes", nargs="*", default=None)
    parser.add_argument(
        "--plot-modes",
        nargs="*",
        default=["heatmap", "bars", "lines"],
        choices=["heatmap", "bars", "lines"],
    )
    return parser.parse_args()


def sanitize_label(value):
    return str(value).replace("/", "-").replace(" ", "-").replace(":", "-")


def load_any_csv(path: Path):
    return load_result_records(path)


def build_output_paths(output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "master_csv": output_dir / f"{stem}.csv",
        "master_json": output_dir / f"{stem}.json",
        "heatmap_pdf": output_dir / f"{stem}_heatmap.pdf",
        "heatmap_png": output_dir / f"{stem}_heatmap.png",
        "bars_pdf": output_dir / f"{stem}_bars.pdf",
        "bars_png": output_dir / f"{stem}_bars.png",
        "lines_pdf": output_dir / f"{stem}_lines.pdf",
        "lines_png": output_dir / f"{stem}_lines.png",
    }


def merge_records(base_records, override_records):
    merged = {
        (str(record["dataset"]), int(record["batch_size"])): dict(record)
        for record in base_records
    }
    for record in override_records:
        merged[(str(record["dataset"]), int(record["batch_size"]))] = dict(record)
    return list(merged.values())


def default_stem(base_records):
    model_name = sanitize_label(base_records[0]["openai_model"]) if base_records else "unknown-model"
    return f"experiment_batch_LLM,{model_name},augmented-camera30,b2,seed0"


def default_notes():
    return [
        "Batch sizes 5, 10, 20, 50 are reused from the original baseline master CSV unless overridden.",
        "Camera batch size 10 is overridden by a camera_catalog rerun with few-shot pairs per class set to 30.",
        "Batch size 2 rows come from a separate baseline rerun with few-shot pairs per class set to 10.",
    ]


def default_title_text():
    return (
        "LLM Batch ER F1\n"
        "baseline [5,10,20,50] + camera b=10 camera_catalog few-shot=30 + baseline b=2"
    )


def default_footer_text():
    return (
        "Mean row is the unweighted average across datasets. "
        "Baseline rows reuse the original matrix; camera b=10 is overridden; b=2 uses baseline prompting."
    )


def load_saving_lookup(output_dir: Path, datasets: list[str], batch_sizes: list[int]):
    candidates = sorted(output_dir.glob("token_estimation_*.csv"))
    if not candidates:
        fallback_dir = Path("results/plot_LLM")
        candidates = sorted(fallback_dir.glob("token_estimation_*.csv"))
    if not candidates:
        return None

    lookup = {}
    with candidates[0].open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["cost_model"] != "recursive_all_records" or row["scenario"] != "p50":
                continue
            dataset = row["dataset"]
            batch_size = int(row["batch_size"])
            if dataset not in datasets or batch_size not in batch_sizes:
                continue
            saving_value = 1.0 if batch_size == 2 else float(row["saving_factor_vs_pairwise_edges"])
            lookup[(dataset, batch_size)] = saving_value
    return lookup or None


def main():
    args = parse_args()
    base_csv = Path(args.base_master_csv)
    override_csvs = [Path(item) for item in args.override_csvs]

    base_records = load_any_csv(base_csv)
    override_records = []
    for path in override_csvs:
        override_records.extend(load_any_csv(path))

    merged_records = merge_records(base_records, override_records)
    datasets = [dataset for dataset in REAL_DATASETS if any(record["dataset"] == dataset for record in merged_records)]
    batch_sizes = sorted({int(record["batch_size"]) for record in merged_records})
    sorted_records = sort_records(merged_records, datasets, batch_sizes)

    stem = args.output_stem or default_stem(sorted_records)
    paths = build_output_paths(Path(args.output_dir), stem)
    write_master_csv(paths["master_csv"], sorted_records)

    provenance = {
        "base_master_csv": str(base_csv),
        "override_csvs": [str(path) for path in override_csvs],
        "notes": args.notes if args.notes is not None else default_notes(),
    }
    payload = {
        "config": {
            "datasets": datasets,
            "batch_sizes": batch_sizes,
            "plot_modes": args.plot_modes,
            "openai_models": sorted({str(record["openai_model"]) for record in sorted_records}),
        },
        "provenance": provenance,
        "completed_configs": len(sorted_records),
        "mean_f1_by_batch_size": mean_f1_by_batch_size(sorted_records, datasets, batch_sizes),
        "results": sorted_records,
    }
    paths["master_json"].write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")

    title_text = args.title_text or default_title_text()
    footer_text = args.footer_text or default_footer_text()
    plot_args = SimpleNamespace(num_batches=10, few_shot_pairs_per_class="mixed", seed=0)
    model_label = ", ".join(sorted({str(record["openai_model"]) for record in sorted_records}))
    saving_lookup = load_saving_lookup(Path(args.output_dir), datasets, batch_sizes)

    if "heatmap" in args.plot_modes:
        render_heatmap(
            sorted_records,
            datasets=datasets,
            batch_sizes=batch_sizes,
            args=plot_args,
            model_name=model_label,
            pdf_path=paths["heatmap_pdf"],
            png_path=paths["heatmap_png"],
            title_text=title_text,
            footer_text=footer_text,
            saving_lookup=saving_lookup,
        )
    if "bars" in args.plot_modes:
        render_grouped_bars(
            sorted_records,
            datasets=datasets,
            batch_sizes=batch_sizes,
            args=plot_args,
            model_name=model_label,
            pdf_path=paths["bars_pdf"],
            png_path=paths["bars_png"],
            title_text=title_text,
        )
    if "lines" in args.plot_modes:
        render_lines(
            sorted_records,
            datasets=datasets,
            batch_sizes=batch_sizes,
            args=plot_args,
            model_name=model_label,
            pdf_path=paths["lines_pdf"],
            png_path=paths["lines_png"],
            title_text=title_text,
        )

    print(
        f"composed results: rows={len(sorted_records)} "
        f"csv={paths['master_csv']} "
        f"heatmap={paths['heatmap_pdf']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
