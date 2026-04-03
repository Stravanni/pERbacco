import argparse
import csv
import importlib.util
import re
import subprocess
import sys
from pathlib import Path


DATASET = "cora"
BATCH_SIZE = 10
LAMBDA_W = "0.05"

RESULTS_DIR = Path("results") / DATASET
FIGURES_DIR = Path("figures")

BASE_METHODS = (
    {
        "label": "SubOpt",
        "color": "green",
        "linestyle": ":",
        "filename": f"{DATASET}_suboptimal,{BATCH_SIZE}.csv",
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
        "label": "pERbacco",
        "color": "red",
        "linestyle": "-",
        "filename": f"{DATASET}_pERbacco,{BATCH_SIZE},lou,{LAMBDA_W}.csv",
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
        "filename": f"{DATASET}_pERbac,{BATCH_SIZE}.csv",
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
        "color": "yellow",
        "linestyle": "-.",
        "filename": f"{DATASET}_Online,{BATCH_SIZE}.csv",
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
)

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


def sanitize_label(value):
    return str(value).replace("/", "-").replace(" ", "-").replace(":", "-")


MAX_QUERY_PATTERN = re.compile(r"^PRINT max_query for\s+\S+\s+and batch_size\s+\d+\s+(\d+)\s*$")
QUERY_PROGRESS_PATTERN = re.compile(r"^(?P<current>\d+)/(?P<total>\d+)\s")


def build_llm_filename(model, prompt_mode):
    parts = [
        f"{DATASET}_LLM-pERbacco",
        str(BATCH_SIZE),
        "lou",
        LAMBDA_W,
        sanitize_label(model),
        prompt_mode,
    ]
    return f"{','.join(parts)}.csv"


def build_methods(args):
    methods = [dict(method, marker=None) for method in BASE_METHODS]
    if args.include_llm:
        methods.append(
            {
                "label": f"LLM-{args.prompt_mode}",
                "color": "black",
                "linestyle": "-",
                "marker": "o",
                "track_query_progress": True,
                "filename": build_llm_filename(args.openai_model, args.prompt_mode),
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
            }
        )
    return methods


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


def run_method(method, force):
    from tqdm import tqdm

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / method["filename"]
    if output_path.exists() and not force:
        print(f"Skipping {method['label']}: {output_path} already exists", flush=True)
        return output_path

    command = [
        sys.executable,
        "perbacco.py",
        "--dataset",
        DATASET,
        "--batch_size",
        str(BATCH_SIZE),
        *method["args"],
    ]
    print(f"Starting {method['label']}: {output_path}", flush=True)
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
                        desc=f"{method['label']} queries",
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
                    desc=f"{method['label']} queries",
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

    print(f"Completed {method['label']}: {output_path}", flush=True)
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


def build_plot(phi, max_query, methods):
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(4.6, 2.7))
    for method in methods:
        csv_path = RESULTS_DIR / method["filename"]
        recalls = read_recall_series(csv_path, max_query)
        x_values = range(len(recalls))
        plt.plot(
            x_values,
            recalls,
            label=method["label"],
            color=method["color"],
            linestyle=method["linestyle"],
            marker=method.get("marker"),
            linewidth=2,
        )

    plt.xlim(0, max_query)
    plt.ylim(0, 1)
    plt.xlabel(rf"number query ($\phi_{{{BATCH_SIZE}}} = {phi}$)")
    plt.ylabel("recall")
    plt.xticks([0, phi, 2 * phi, 3 * phi], [0, phi, 2 * phi, 3 * phi])
    plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()

    suffix = ""
    if any(method["label"].startswith("LLM-") for method in methods):
        suffix = f"_llm_{sanitize_label(methods[-1]['label'])}"

    pdf_path = FIGURES_DIR / f"{DATASET}_{BATCH_SIZE}{suffix}.pdf"
    png_path = FIGURES_DIR / f"{DATASET}_{BATCH_SIZE}{suffix}.png"
    plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    return pdf_path, png_path


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Run the Cora-only experiments needed for Figure 3(a) and recreate the plot."
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Reuse existing CSV files in results/cora and only generate the plot.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun the experiments even if the expected CSV files already exist.",
    )
    parser.add_argument(
        "--include-llm",
        action="store_true",
        help="Add one LLM-backed pERbacco line to the figure.",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-5-mini",
        help="Model to use for the LLM-backed line.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["zero-shot", "few-shot"],
        default="zero-shot",
        help="Prompting mode for the LLM-backed line.",
    )
    return parser.parse_args()


def main():
    from tqdm import tqdm

    args = parse_cli_args()
    ensure_dependencies()

    methods = build_methods(args)
    phi = compute_phi(Path("datasets") / DATASET, BATCH_SIZE)
    max_query = 3 * phi

    if phi != 137:
        print(f"Warning: computed phi_{BATCH_SIZE} for {DATASET} is {phi}, expected 137.")

    if not args.skip_run:
        with tqdm(total=len(methods), desc="Methods", unit="run", dynamic_ncols=True) as method_bar:
            for method in methods:
                method_bar.set_postfix_str(method["label"])
                run_method(method, force=args.force)
                method_bar.update(1)

    missing_outputs = [
        str(RESULTS_DIR / method["filename"])
        for method in methods
        if not (RESULTS_DIR / method["filename"]).exists()
    ]
    if missing_outputs:
        joined = "\n".join(missing_outputs)
        raise SystemExit(f"Missing experiment outputs:\n{joined}")

    pdf_path, png_path = build_plot(phi, max_query, methods)
    print(f"phi_{BATCH_SIZE}({DATASET}) = {phi}")
    print(f"Saved plot to {pdf_path}")
    print(f"Saved plot to {png_path}")


if __name__ == "__main__":
    main()
