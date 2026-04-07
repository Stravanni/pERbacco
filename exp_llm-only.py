import argparse
import csv
import importlib.util
import json
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path


DEFAULT_DATASET = "cora"
DEFAULT_BATCH_SIZE = 10
LAMBDA_W = "0.05"

DATASET_CHOICES = (
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
)

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


def sanitize_label(value):
    return str(value).replace("/", "-").replace(" ", "-").replace(":", "-")


def build_manifest_path(csv_path):
    return csv_path.parent / f"{csv_path.name}.complete.json"


def build_resume_path(csv_path):
    return csv_path.parent / f"{csv_path.name}.resume.pkl"


def get_results_dir(dataset):
    return Path("results") / dataset


def build_llm_filename(dataset, batch_size, method_key, model, prompt_mode):
    method_label = {
        "perbacco": "LLM-pERbacco",
        "perbac": "LLM-pERbac",
        "online": "LLM-Online",
    }[method_key]

    parts = [f"{dataset}_{method_label}", str(batch_size)]
    if method_key == "perbacco":
        parts.extend(["lou", LAMBDA_W])
    parts.extend([sanitize_label(model), prompt_mode])
    return f"{','.join(parts)}.csv"


def build_methods(args):
    return [
        {
            "label": "Ideal",
            "color": "green",
            "linestyle": ":",
            "marker": None,
            "track_query_progress": True,
            "requires_manifest": False,
            "filename": f"{args.dataset}_suboptimal,{args.batch_size}.csv",
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
                "--oracle_backend",
                "groundtruth",
            ],
        },
        {
            "label": "pERbacco + LLM",
            "color": "red",
            "linestyle": "-",
            "marker": None,
            "track_query_progress": True,
            "requires_manifest": True,
            "filename": build_llm_filename(
                args.dataset,
                args.batch_size,
                "perbacco",
                args.openai_model,
                args.prompt_mode,
            ),
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
            "label": "pERbac + LLM",
            "color": "blue",
            "linestyle": "--",
            "marker": None,
            "track_query_progress": True,
            "requires_manifest": True,
            "filename": build_llm_filename(
                args.dataset,
                args.batch_size,
                "perbac",
                args.openai_model,
                args.prompt_mode,
            ),
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
                "--oracle_backend",
                "openai",
                "--openai_model",
                args.openai_model,
                "--prompt_mode",
                args.prompt_mode,
            ],
        },
        {
            "label": "Online + LLM",
            "color": "yellow",
            "linestyle": "-.",
            "marker": None,
            "track_query_progress": True,
            "requires_manifest": True,
            "filename": build_llm_filename(
                args.dataset,
                args.batch_size,
                "online",
                args.openai_model,
                args.prompt_mode,
            ),
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
                "--oracle_backend",
                "openai",
                "--openai_model",
                args.openai_model,
                "--prompt_mode",
                args.prompt_mode,
            ],
        },
    ]


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


def write_manifest(manifest_path, method, command, output_path):
    manifest = {
        "method": method["label"],
        "csv_path": str(output_path),
        "command": command,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "completed",
    }
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)


def prepare_method_run(args, method, force):
    results_dir = get_results_dir(args.dataset)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / method["filename"]
    manifest_path = build_manifest_path(output_path)
    command = [
        sys.executable,
        "perbacco.py",
        "--dataset",
        args.dataset,
        "--batch_size",
        str(args.batch_size),
        *method["args"],
    ]

    run = {
        **method,
        "output_path": output_path,
        "manifest_path": manifest_path,
        "command": command,
        "skip": False,
        "skip_reason": "",
        "rerun_reason": "",
        "process": None,
        "reader_thread": None,
        "reader_done": False,
        "progress_current": 0,
        "progress_total": 0,
        "finalized": False,
        "insufficient_quota": False,
        "last_error_line": "",
    }

    if output_path.exists() and not force:
        if not method["requires_manifest"] or manifest_path.exists():
            run["skip"] = True
            run["skip_reason"] = f"Skipping {method['label']}: {output_path} already verified"
        else:
            run["rerun_reason"] = (
                f"Re-running {method['label']}: {output_path} exists but has no completion manifest"
            )
    elif force and output_path.exists():
        run["rerun_reason"] = f"Re-running {method['label']}: {output_path} because --force was set"

    return run


def apply_method_total(run, progress_bar, total):
    total = int(total)
    if total <= run["progress_total"]:
        return

    progress_bar.total = (progress_bar.total or 0) + (total - run["progress_total"])
    run["progress_total"] = total
    progress_bar.refresh()


def apply_method_progress(run, progress_bar, current, total):
    apply_method_total(run, progress_bar, total)
    increment = int(current) - run["progress_current"]
    if increment > 0:
        progress_bar.update(increment)
        run["progress_current"] += increment
        progress_bar.refresh()


def handle_method_output_line(run, line, progress_bar):
    from tqdm import tqdm

    if "insufficient_quota" in line or "You exceeded your current quota" in line:
        run["insufficient_quota"] = True
    if "Traceback" in line or "RuntimeError:" in line or "OpenAI API request failed" in line:
        run["last_error_line"] = line

    if run.get("track_query_progress"):
        max_query_match = MAX_QUERY_PATTERN.match(line)
        if max_query_match is not None:
            apply_method_total(run, progress_bar, int(max_query_match.group(1)))
            return

        query_progress_match = QUERY_PROGRESS_PATTERN.match(line)
        if query_progress_match is not None:
            apply_method_progress(
                run,
                progress_bar,
                int(query_progress_match.group("current")),
                int(query_progress_match.group("total")),
            )
            return

    tqdm.write(f"[{run['label']}] {line}")


def finalize_successful_run(run, progress_bar):
    from tqdm import tqdm

    if run["finalized"]:
        return

    if run["progress_total"] == 0:
        inferred_total = run["progress_current"] if run["progress_current"] > 0 else 1
        apply_method_total(run, progress_bar, inferred_total)

    remaining = run["progress_total"] - run["progress_current"]
    if remaining > 0:
        progress_bar.update(remaining)
        run["progress_current"] += remaining
        progress_bar.refresh()

    if run["requires_manifest"]:
        write_manifest(run["manifest_path"], run, run["command"], run["output_path"])

    run["finalized"] = True
    tqdm.write(f"Completed {run['label']}: {run['output_path']}")


def start_run_process(run):
    from tqdm import tqdm

    if run["manifest_path"].exists():
        run["manifest_path"].unlink()

    if run["rerun_reason"]:
        tqdm.write(run["rerun_reason"])

    tqdm.write(f"Starting {run['label']}: {run['output_path']}")
    tqdm.write("Running: " + " ".join(run["command"]))
    run["process"] = subprocess.Popen(
        run["command"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def enqueue_process_output(run, event_queue):
    process = run["process"]
    assert process is not None
    assert process.stdout is not None

    try:
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if line:
                event_queue.put(("line", run["label"], line))
    finally:
        process.stdout.close()
        event_queue.put(("eof", run["label"], ""))


def terminate_run(run, reason):
    from tqdm import tqdm

    process = run["process"]
    if process is None or process.poll() is not None:
        return

    tqdm.write(f"Stopping {run['label']}: {reason}")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def format_resume_status(run):
    checkpoint_path = build_resume_path(run["output_path"])
    if checkpoint_path.exists():
        return f"checkpoint available at {checkpoint_path}"
    if run["output_path"].exists():
        return f"partial CSV present at {run['output_path']}, but no checkpoint"
    return "no checkpoint found"


def raise_quota_error(run):
    detail = format_resume_status(run)
    raise SystemExit(
        "OpenAI quota exhausted for "
        f"{run['label']}. "
        "The API returned insufficient_quota (HTTP 429). "
        "Restore quota or billing, then rerun the same command to resume if possible. "
        f"Resume status: {detail}."
    )


def run_method_serial(run):
    from tqdm import tqdm

    start_run_process(run)
    progress_bar = tqdm(total=0, desc="Overall progress", unit="query", dynamic_ncols=True)
    try:
        process = run["process"]
        assert process is not None
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            handle_method_output_line(run, line, progress_bar)

        return_code = process.wait()
        if return_code != 0:
            tqdm.write(f"{run['label']} failed with exit code {return_code}")
            if run["insufficient_quota"]:
                raise_quota_error(run)
            raise subprocess.CalledProcessError(return_code, run["command"])

        finalize_successful_run(run, progress_bar)
        return run["output_path"]
    finally:
        progress_bar.close()


def run_methods_parallel(runs):
    from tqdm import tqdm

    tqdm.write("Running methods in parallel: " + ", ".join(run["label"] for run in runs))
    event_queue = queue.Queue()
    runs_by_label = {run["label"]: run for run in runs}

    for run in runs:
        start_run_process(run)
        reader_thread = threading.Thread(target=enqueue_process_output, args=(run, event_queue), daemon=True)
        run["reader_thread"] = reader_thread
        reader_thread.start()

    progress_bar = tqdm(total=0, desc="Overall progress", unit="query", dynamic_ncols=True)
    active_labels = set(runs_by_label)
    progress_bar.set_postfix_str(f"{len(active_labels)} active")
    failure_run = None

    try:
        while active_labels:
            try:
                kind, label, payload = event_queue.get(timeout=0.2)
            except queue.Empty:
                kind = None
                label = None
                payload = None

            if kind == "line":
                handle_method_output_line(runs_by_label[label], payload, progress_bar)
            elif kind == "eof":
                runs_by_label[label]["reader_done"] = True

            finished_labels = []
            for active_label in list(active_labels):
                run = runs_by_label[active_label]
                process = run["process"]
                assert process is not None
                return_code = process.poll()
                if return_code is None or not run["reader_done"]:
                    continue
                if return_code != 0:
                    failure_run = run
                    break

                finalize_successful_run(run, progress_bar)
                finished_labels.append(active_label)

            if failure_run is not None:
                break

            for finished_label in finished_labels:
                active_labels.remove(finished_label)
                progress_bar.set_postfix_str(f"{len(active_labels)} active")

        if failure_run is not None:
            process = failure_run["process"]
            assert process is not None
            tqdm.write(f"{failure_run['label']} failed with exit code {process.returncode}")
            for active_label in list(active_labels):
                if active_label == failure_run["label"]:
                    continue
                active_run = runs_by_label[active_label]
                active_process = active_run["process"]
                assert active_process is not None
                if active_run["reader_done"] and active_process.poll() == 0:
                    finalize_successful_run(active_run, progress_bar)
                    continue
                terminate_run(runs_by_label[active_label], f"{failure_run['label']} failed")
            if failure_run["insufficient_quota"]:
                raise_quota_error(failure_run)
            raise subprocess.CalledProcessError(process.returncode, failure_run["command"])
    finally:
        progress_bar.close()
        for run in runs:
            reader_thread = run.get("reader_thread")
            if reader_thread is not None:
                reader_thread.join(timeout=1)


def read_csv_rows(csv_path):
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Empty experiment output: {csv_path}")
    return rows


def read_recall_series(csv_path, max_query):
    recalls = [0.0]
    for index, row in enumerate(read_csv_rows(csv_path), start=1):
        if index > max_query:
            break
        recall = float(row["recall"])
        recalls.append(recall)
        if recall >= 1.0:
            break
    return recalls


def compute_llm_ceiling(methods, results_dir):
    best_label = None
    best_recall = None
    for method in methods:
        if not method["requires_manifest"]:
            continue
        rows = read_csv_rows(results_dir / method["filename"])
        final_recall = float(rows[-1]["recall"])
        if best_recall is None or final_recall > best_recall:
            best_recall = final_recall
            best_label = method["label"]

    if best_recall is None or best_recall <= 0:
        raise SystemExit("Unable to normalize recalls: best final LLM recall must be positive.")

    return best_recall, best_label


def plot_series(methods, results_dir, batch_size, phi, max_query, ylabel, pdf_path, png_path, transform):
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(4.6, 2.7))
    for method in methods:
        csv_path = results_dir / method["filename"]
        recalls = transform(method, read_recall_series(csv_path, max_query))
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
    plt.xlabel(rf"number query ($\phi_{{{batch_size}}} = {phi}$)")
    plt.ylabel(ylabel)
    plt.xticks([0, phi, 2 * phi, 3 * phi], [0, phi, 2 * phi, 3 * phi])
    plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def build_true_ideal_plot(methods, results_dir, batch_size, phi, max_query, figure_prefix):
    pdf_path = FIGURES_DIR / f"{figure_prefix}_true-ideal.pdf"
    png_path = FIGURES_DIR / f"{figure_prefix}_true-ideal.png"
    plot_series(
        methods,
        results_dir,
        batch_size,
        phi,
        max_query,
        "recall",
        pdf_path,
        png_path,
        transform=lambda _method, recalls: recalls,
    )
    return pdf_path, png_path


def build_normalized_plot(methods, results_dir, batch_size, phi, max_query, figure_prefix, llm_ceiling):
    def transform(method, recalls):
        normalized = []
        for recall in recalls:
            scaled = recall / llm_ceiling
            if not method["requires_manifest"]:
                scaled = min(recall, llm_ceiling) / llm_ceiling
            normalized.append(min(scaled, 1.0))
        return normalized

    pdf_path = FIGURES_DIR / f"{figure_prefix}_ideal-normalized.pdf"
    png_path = FIGURES_DIR / f"{figure_prefix}_ideal-normalized.png"
    plot_series(
        methods,
        results_dir,
        batch_size,
        phi,
        max_query,
        "normalized recall",
        pdf_path,
        png_path,
        transform=transform,
    )
    return pdf_path, png_path


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Run the LLM-oracle variant of Figure 3 and create true-ideal and normalized plots."
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default=DEFAULT_DATASET,
        help="Dataset to run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size to use for all methods.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Reuse existing CSV files in results/<dataset> and only generate the plots.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun the experiments even if the expected CSV files already exist.",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-5-mini",
        help="Model to use for the OpenAI-backed methods.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["zero-shot", "few-shot"],
        default="zero-shot",
        help="Prompting mode for the OpenAI-backed methods.",
    )
    return parser.parse_args()


def main():
    from tqdm import tqdm

    args = parse_cli_args()
    ensure_dependencies()

    methods = build_methods(args)
    results_dir = get_results_dir(args.dataset)
    phi = compute_phi(Path("datasets") / args.dataset, args.batch_size)
    max_query = 3 * phi
    figure_prefix = (
        f"{args.dataset}_{args.batch_size}_llm-only_{sanitize_label(args.openai_model)}_{sanitize_label(args.prompt_mode)}"
    )

    if args.dataset == "cora" and args.batch_size == 10 and phi != 137:
        print(f"Warning: computed phi_{args.batch_size} for {args.dataset} is {phi}, expected 137.")

    if not args.skip_run:
        prepared_runs = [prepare_method_run(args, method, force=args.force) for method in methods]
        runnable_runs = []
        for run in prepared_runs:
            if run["skip"]:
                tqdm.write(run["skip_reason"])
            else:
                runnable_runs.append(run)

        if len(runnable_runs) == 1:
            run_method_serial(runnable_runs[0])
        elif len(runnable_runs) > 1:
            run_methods_parallel(runnable_runs)

    missing_outputs = [
        str(results_dir / method["filename"])
        for method in methods
        if not (results_dir / method["filename"]).exists()
    ]
    if missing_outputs:
        joined = "\n".join(missing_outputs)
        raise SystemExit(f"Missing experiment outputs:\n{joined}")

    llm_ceiling, llm_ceiling_label = compute_llm_ceiling(methods, results_dir)
    true_pdf_path, true_png_path = build_true_ideal_plot(
        methods,
        results_dir,
        args.batch_size,
        phi,
        max_query,
        figure_prefix,
    )
    normalized_pdf_path, normalized_png_path = build_normalized_plot(
        methods,
        results_dir,
        args.batch_size,
        phi,
        max_query,
        figure_prefix,
        llm_ceiling,
    )

    print(f"phi_{args.batch_size}({args.dataset}) = {phi}")
    print(f"Ideal-normalized ceiling: {llm_ceiling:.6f} from {llm_ceiling_label}")
    print(f"Saved plot to {true_pdf_path}")
    print(f"Saved plot to {true_png_path}")
    print(f"Saved plot to {normalized_pdf_path}")
    print(f"Saved plot to {normalized_png_path}")


if __name__ == "__main__":
    main()
