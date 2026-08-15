# Experiments

The maintained experiment CLI currently uses only `GroundTruthOracle`. It never
calls OpenAI or OpenRouter. API-backed runs use the same `run()` function but are
intentionally separate from paper reproduction so a bulk command cannot incur
provider cost unexpectedly.

## Install experiment extras

```bash
python -m pip install '.[data,plot,leiden]'
```

The installed library itself remains dependency-free. Extras are isolated by
purpose:

| Extra | Used for |
| --- | --- |
| `data` | streaming Parquet similarity graphs with PyArrow |
| `plot` | rendering maintained Figure 4 PNG/PDF artifacts |
| `leiden` | external igraph Louvain and Leiden communities for Table IV |

## Dataset layout

Run commands from a repository/data root containing:

```text
datasets/
  cora/groundtruth.csv
  camera/groundtruth.csv
  funding/groundtruth.csv
  wdc80/groundtruth.csv
  voters/groundtruth.csv
  synth_10000/groundtruth.csv
similarity_graph/
  cora.parquet
  camera.parquet
  funding.parquet
  wdc80.parquet
  voters.parquet
  synth_precision_0.5/synth_10000.parquet
  synth_precision_0.2/synth_10000.parquet
  synth_precision_0.05/synth_10000.parquet
```

Record CSVs, when present, are loaded for oracle views. Ground-truth files are
pair lists whose connected components define truth entities. Real graph
Parquets use `id1`, `id2`, `w` or `left_spec_id`, `right_spec_id`, `weight`.

## One ground-truth run

```bash
perbacco \
  --root . \
  --artifacts artifacts \
  run \
  --dataset cora \
  --method perbacco \
  --cda louvain \
  --batch-size 10 \
  --top-k 1000 \
  --seed 42 \
  --query-budget 411
```

`--method` accepts `perbacco`, `perbac`, `online`, and `subopt`.
`--cda` accepts `none` and `louvain` in the public one-run CLI. If
`--query-budget` is omitted, the command uses three times the computed lower
`phi` bound. Synthetic graphs select `--precision 0.5`, `0.2`, or `0.05`.

SubOpt loads only record/truth data and requires no similarity Parquet:

```bash
perbacco --root . run --dataset camera --method subopt --cda none
```

## Reproduction commands

```bash
# Exact Table III bounds
perbacco --root . reproduce --scope table-iii

# Funding/Voters community-backend comparison
perbacco --root . reproduce --scope table-iv

# 8 panels × 4 methods, ground-truth oracle
perbacco --root . reproduce --scope figure4

# All of the above
perbacco --root . reproduce --scope all
```

Table III is the exact paper reproduction. Figure 4 output must be described as
**maintained C implementation results**. See [research status](research-status.md)
before comparing its curves with the paper.

## Community backends

- Figure 4 pERbacco uses built-in deterministic C Louvain.
- A one-run `--cda louvain` uses the same built-in backend.
- Table IV evaluates externally computed seeded igraph Louvain and Leiden
  communities through `CDA.EXTERNAL`.
- pERbac, Online, and SubOpt do not run a community phase.

The backends share high-level paper parameters but are not partition-identical
to the prototype's NetworkX Louvain implementation.

## Artifacts and checkpoints

The default `artifacts/` directory is not version controlled. Each exact run ID
gets:

```text
artifacts/runs/<run-id>/
  curve.csv
  summary.json
  manifest.json
  timing.json
  checkpoints/query-<n>.pbj
```

Checkpoints are recorded at `phi`, `2 phi`, and `3 phi` when the run reaches
those queries. Aggregate commands also write `table-iii.json`, `table-iv.json`,
`figure4.json`, `figure4.png`, `figure4.pdf`,
`figure4-checkpoints.json`, and `deviations.json`.

A cached run is reused only when its configuration matches, its curve checksum
matches the manifest, and its implementation fingerprint matches the loaded C
library, result-affecting Python modules, Python runtime, and optional graph/
community dependency versions. This prevents curves created by an older engine
or backend from being silently relabeled as current output.

## Memory and runtime

Large runs execute in child processes so graph memory is released between
configurations and peak RSS is recorded independently. SubOpt avoids similarity
graphs entirely. The Camera graph has 9,840,153 edges; a previous macOS arm64
proof run of built-in-Louvain pERbacco peaked at about 1.31 GB. Treat 1.5 GB as a
planning floor, not a cross-platform guarantee, and leave additional headroom
for the Python interpreter, PyArrow buffers, and plotting.

Table III is lightweight. Full Figure 4 performs 32 curves and may take
substantial time. Interrupted checksum-valid runs are reused; journal snapshots
are replayable with the same immutable inputs.

## Reproducibility checklist

Record the repository commit, compiler, platform, Python version, optional-extra
versions, command line, dataset checksums, and generated manifest. Keep
`seed=42`, `batch_size=10`, `top_k=1000`, and the stated exact budget when
comparing with maintained reference runs. Do not compare an igraph/Leiden curve
to a built-in-C-Louvain curve as if only the scheduler changed.
