import argparse
import csv
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from perbacco import build_output_path


DATASET = "cora"
LAMBDA_W = "0.05"
DEFAULT_BATCH_SIZES = (2, 5, 10, 20, 40)

RESULTS_DIR = Path("results") / DATASET
FIGURES_DIR = Path("figures")

REQUIRED_MODULES = (
    "IPython",
    "brokenaxes",
    "fastparquet",
    "igraph",
    "leidenalg",
    "matplotlib",
    "networkx",
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "sklearn",
    "torch",
    "tqdm",
)

MAX_QUERY_PATTERN = re.compile(r"^PRINT max_query for\s+\S+\s+and batch_size\s+\d+\s+(\d+)\s*$")
QUERY_PROGRESS_PATTERN = re.compile(r"^(?P<current>\d+)/(?P<total>\d+)\s")


def q_rec_k(size, batch_size):
    if size < batch_size:
        return 0
    quotient = size // batch_size
    remainder = size % batch_size
    return quotient + q_rec_k(quotient + remainder, batch_size)


def r_rec_k(size, batch_size):
    if size < batch_size:
        return size
    quotient = size // batch_size
    remainder = size % batch_size
    return r_rec_k(quotient + remainder, batch_size)


class DisjointSet:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, item):
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item):
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left, right):
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        rank_left = self.rank[root_left]
        rank_right = self.rank[root_right]
        if rank_left < rank_right:
            self.parent[root_left] = root_right
        elif rank_left > rank_right:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1


def ensure_dependencies():
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            "Missing Python dependencies required by perbacco.py: "
            f"{joined}. Install them in the current environment, then rerun."
        )


def compute_phi(dataset_dir, batch_size):
    groundtruth_path = dataset_dir / "groundtruth.csv"
    disjoint_set = DisjointSet()

    with groundtruth_path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty ground truth file: {groundtruth_path}")

        for row in reader:
            if len(row) < 2:
                continue
            left = row[0]
            right = row[1]
            disjoint_set.add(left)
            disjoint_set.add(right)
            disjoint_set.union(left, right)

    component_sizes = {}
    for node in disjoint_set.parent:
        root = disjoint_set.find(node)
        component_sizes[root] = component_sizes.get(root, 0) + 1

    minimum_queries = 0
    rests = []
    for size in component_sizes.values():
        minimum_queries += q_rec_k(size, batch_size)
        rest = r_rec_k(size, batch_size)
        if rest > 1:
            rests.append(rest)

    return minimum_queries + ((sum(rests) + batch_size - 1) // batch_size)


def llm_output_name(batch_size, args):
    return build_output_path(
        DATASET,
        batch_size,
        "louvain",
        LAMBDA_W,
        "brmean",
        "False",
        "False",
        oracle_backend="openai",
        openai_model=args.openai_model,
        prompt_mode=args.prompt_mode,
        few_shot_pairs_per_class=args.few_shot_pairs_per_class if args.prompt_mode == "few-shot" else None,
        max_llm_calls=args.max_llm_calls,
    ).name


def build_methods(batch_size, args):
    methods = [
        {
            "label": "SubOpt",
            "color": "green",
            "linestyle": ":",
            "filename": build_output_path(
                DATASET,
                batch_size,
                "False",
                "False",
                "brmax",
                "True",
                "False",
            ).name,
            "args": [
                "--alg_community",
                "False",
                "--lambda_w",
                "False",
                "--mu_benefit",
                "brmax",
                "--optimal",
                "True",
                "--synth_precision",
                "False",
            ],
        },
        {
            "label": "pERbacco+LLM",
            "color": "black",
            "linestyle": "-",
            "marker": "o",
            "track_query_progress": True,
            "filename": llm_output_name(batch_size, args),
            "args": [
                "--alg_community",
                "louvain",
                "--lambda_w",
                LAMBDA_W,
                "--mu_benefit",
                "brmean",
                "--optimal",
                "False",
                "--synth_precision",
                "False",
                "--oracle_backend",
                "openai",
                "--openai_model",
                args.openai_model,
                "--prompt_mode",
                args.prompt_mode,
            ],
        },
        {
            "label": "pERbacco",
            "color": "red",
            "linestyle": "-",
            "filename": build_output_path(
                DATASET,
                batch_size,
                "louvain",
                LAMBDA_W,
                "brmean",
                "False",
                "False",
            ).name,
            "args": [
                "--alg_community",
                "louvain",
                "--lambda_w",
                LAMBDA_W,
                "--mu_benefit",
                "brmean",
                "--optimal",
                "False",
                "--synth_precision",
                "False",
            ],
        },
        {
            "label": "pERbac",
            "color": "blue",
            "linestyle": "--",
            "filename": build_output_path(
                DATASET,
                batch_size,
                "False",
                "False",
                "brmean",
                "False",
                "False",
            ).name,
            "args": [
                "--alg_community",
                "False",
                "--lambda_w",
                "False",
                "--mu_benefit",
                "brmean",
                "--optimal",
                "False",
                "--synth_precision",
                "False",
            ],
        },
        {
            "label": "Online",
            "color": "orange",
            "linestyle": "-.",
            "filename": build_output_path(
                DATASET,
                batch_size,
                "False",
                "False",
                "brmax",
                "False",
                "False",
            ).name,
            "args": [
                "--alg_community",
                "False",
                "--lambda_w",
                "False",
                "--mu_benefit",
                "brmax",
                "--optimal",
                "False",
                "--synth_precision",
                "False",
            ],
        },
    ]
    if args.prompt_mode == "few-shot":
        methods[1]["args"].extend(
            [
                "--few-shot-pairs-per-class",
                str(args.few_shot_pairs_per_class),
            ]
        )
    if args.max_llm_calls is not None:
        methods[1]["args"].extend(["--max_llm_calls", str(args.max_llm_calls)])
    return methods


def run_method(batch_size, method, force):
    from tqdm import tqdm

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / method["filename"]
    if output_path.exists() and not force:
        print(f"Skipping b={batch_size} {method['label']}: {output_path} already exists", flush=True)
        return output_path

    command = [
        sys.executable,
        "perbacco.py",
        "--dataset",
        DATASET,
        "--batch_size",
        str(batch_size),
        *method["args"],
    ]
    print(f"Starting b={batch_size} {method['label']}: {output_path}", flush=True)
    print("Running:", " ".join(command), flush=True)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    query_bar = None
    last_query_value = 0
    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            tqdm.write(line)

            if not method.get("track_query_progress"):
                continue

            max_query_match = MAX_QUERY_PATTERN.match(line)
            if max_query_match is not None:
                total = int(max_query_match.group(1))
                if query_bar is None:
                    query_bar = tqdm(
                        total=total,
                        desc=f"b={batch_size} {method['label']} queries",
                        unit="query",
                        leave=False,
                        dynamic_ncols=True,
                    )
                else:
                    query_bar.total = total
                    query_bar.refresh()
                continue

            query_progress_match = QUERY_PROGRESS_PATTERN.match(line)
            if query_progress_match is None:
                continue

            current = int(query_progress_match.group("current"))
            total = int(query_progress_match.group("total"))
            if query_bar is None:
                query_bar = tqdm(
                    total=total,
                    desc=f"b={batch_size} {method['label']} queries",
                    unit="query",
                    leave=False,
                    dynamic_ncols=True,
                )
            elif query_bar.total != total:
                query_bar.total = total

            increment = current - last_query_value
            if increment > 0:
                query_bar.update(increment)
                last_query_value = current
            query_bar.refresh()
    finally:
        if query_bar is not None:
            query_bar.close()

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)

    print(f"Completed b={batch_size} {method['label']}: {output_path}", flush=True)
    return output_path


def read_recall_series(csv_path, max_query):
    recalls = [0.0]
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            if index > max_query:
                break
            recall = float(row["recall"])
            recalls.append(recall)
            if recall >= 1.0:
                break
    return recalls


def build_plot(batch_size, phi, max_query, methods):
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(4.6, 2.7))
    for method in methods:
        csv_path = RESULTS_DIR / method["filename"]
        recalls = read_recall_series(csv_path, max_query)
        plt.plot(
            range(len(recalls)),
            recalls,
            label=method["label"],
            color=method["color"],
            linestyle=method["linestyle"],
            marker=method.get("marker"),
            linewidth=2,
        )

    plt.xlim(0, max_query)
    plt.ylim(0, 1)
    plt.xlabel(rf"number query ($\phi_{{{batch_size}}} = {phi}$)")
    plt.ylabel("recall")
    plt.xticks([0, phi, 2 * phi, 3 * phi], [0, phi, 2 * phi, 3 * phi])
    plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()

    pdf_path = FIGURES_DIR / f"fig4_new_{DATASET}_{batch_size}.pdf"
    png_path = FIGURES_DIR / f"fig4_new_{DATASET}_{batch_size}.png"
    plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    return pdf_path, png_path


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Recreate Figure 4 on CORA and add one OpenAI-backed pERbacco line."
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Reuse existing CSV files in results/cora and only generate the plots.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun the experiments even if the expected CSV files already exist.",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-5-mini",
        help="Model to use for the OpenAI-backed line.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["zero-shot", "few-shot"],
        default="few-shot",
        help="Prompting mode for the OpenAI-backed line.",
    )
    parser.add_argument(
        "--few-shot-pairs-per-class",
        type=int,
        default=20,
        help="Positive and negative pair examples per class for the OpenAI-backed line.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
        help="Batch sizes to evaluate.",
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=None,
        help="Optional cap for the OpenAI-backed line; defaults to the full 3phi horizon.",
    )
    return parser.parse_args()


def main():
    from tqdm import tqdm

    args = parse_cli_args()
    ensure_dependencies()

    dataset_dir = Path("datasets") / DATASET
    phi_by_batch = {batch_size: compute_phi(dataset_dir, batch_size) for batch_size in args.batch_sizes}

    if not args.skip_run:
        total_runs = len(args.batch_sizes) * 5
        with tqdm(total=total_runs, desc="Runs", unit="run", dynamic_ncols=True) as run_bar:
            for batch_size in args.batch_sizes:
                methods = build_methods(batch_size, args)
                for method in methods:
                    run_bar.set_postfix_str(f"b={batch_size} {method['label']}")
                    run_method(batch_size, method, force=args.force)
                    run_bar.update(1)

    saved_plots = []
    for batch_size in args.batch_sizes:
        methods = build_methods(batch_size, args)
        missing_outputs = [
            str(RESULTS_DIR / method["filename"])
            for method in methods
            if not (RESULTS_DIR / method["filename"]).exists()
        ]
        if missing_outputs:
            joined = "\n".join(missing_outputs)
            raise SystemExit(f"Missing experiment outputs for batch size {batch_size}:\n{joined}")

        phi = phi_by_batch[batch_size]
        pdf_path, png_path = build_plot(batch_size, phi, 3 * phi, methods)
        saved_plots.extend([pdf_path, png_path])
        print(f"phi_{batch_size}({DATASET}) = {phi}", flush=True)
        print(f"Saved plot to {pdf_path}", flush=True)
        print(f"Saved plot to {png_path}", flush=True)

    return saved_plots


if __name__ == "__main__":
    main()
