# Dependency-light pERbacco C library and experiment framework

## Summary

Rebuild the scheduling algorithms as a BSD-3-Clause C99 library with a stable C ABI, a dependency-free `ctypes` Python wrapper, and separate optional experiment tooling. V1 implements pERbacco, pERbac, Online, and ground-truth-only SubOpt; reproduces Table III exactly; records maintained-implementation Table IV and Figure 4 results with explicit deviations; and provides a tested OpenAI-compatible oracle interface without making paid API calls.

## Core library and public interfaces

- Create a zero-runtime-dependency C core using compact graph storage, union-find entity state, open-addressed candidate-edge tables, indexed heaps, and a local seeded PRNG. Avoid object-heavy graphs and quadratic complete-graph materialization.
- Expose opaque `pb_engine` state and operations to create an engine, get the next batch, submit a complete partition, inspect group members and statistics, snapshot/restore state, report errors, and free resources.
- Support pERbacco, pERbac, Online, and SubOpt. SubOpt requires ground-truth entity labels and cannot be combined with an API oracle.
- Implement the paper equations directly: mean and maximum benefits, GreedyHS1000, temperature behavior, exact query budgets, recursive heavy-community processing, and deterministic seeded tie-breaking.
- Implement deterministic weighted Louvain in C. Also accept externally computed heavy communities so optional Leiden experiments use the same C scheduler.
- Normalize finite non-negative weights to `[0,1]`; validate IDs, edges, graph shape, oracle partitions, and consistency before mutating state.

## Python, oracle, and migration layer

- Add a `perbacco` Python package exposing graph, engine, batch, statistics, ground-truth oracle, OpenAI-compatible oracle, and orchestration APIs. Map arbitrary external IDs deterministically to dense C IDs.
- Build and package the shared library through a top-level Makefile and minimal setuptools build hook for macOS and Linux.
- Implement a provider-neutral oracle protocol receiving entity views and returning a complete partition.
- Implement OpenAI Responses and Chat Completions transports with the standard library only, including OpenAI and OpenRouter presets, strict structured output, retries, usage accounting, validation, response journaling, and fake-server contract tests.
- Mine the `LLM_calls` and `sample-batch-llm` branches for behavior, but port concepts rather than merging their Python monoliths.
- Keep useful legacy top-level commands as compatibility entry points. Preserve current uncommitted README and paper files when updating documentation.
- License the new deliverable under BSD-3-Clause. Keep unlicensed related-work repositories as behavioral references only; do not copy their source.

## Experiment reproduction

- Keep the installed library dependency-free. Use optional extras only for Parquet conversion (`pyarrow`), plotting (`matplotlib`), and seeded external Louvain/Leiden communities (`python-igraph`, `leidenalg`).
- Remove pandas, NetworkX, Torch, SciPy, scikit-learn, IPython, brokenaxes, PySCIPOpt, and `binpacking` from the active workflow.
- Build graph nodes from dataset records, graph endpoints, and truth endpoints. Derive truth labels by connected components.
- Implement the Phi lower bound and a dependency-free exact/count-based bin-packing solver for the upper bound.
- Run Table III for all six datasets, Table IV for Funding and Voters with the specified thresholds and CDAs, and all eight maintained Figure 4 panels using `b=10`, seed `42`, `3 phi_10`, GreedyHS1000, and all four schedulers.
- Write reproducible run manifests, curves, summaries, checksums, timings, checkpoints, consolidated tables, Figure 4 PDF/PNG, and a deviation report under `artifacts/`.
- Parallelize only independent runs; keep each engine deterministic and single-threaded.

## Correctness and acceptance

- Eliminate known prototype faults: missing imports, query-budget off-by-one behavior, swapped method labels, zeroed gain history, incomplete benefit invalidation, stale representatives, invalid set operations, undefined sentinels, and inconsistent thresholds.
- Add C unit tests, deterministic randomized differential tests, invariants, Python tests, API fake-server tests, sanitizer checks, ABI smoke tests, and Linux/macOS build checks.
- Table III must match exactly. Table IV must be within `0.01` unless a larger delta is traced to a corrected prototype bug. Figure 4 checkpoints at `phi`, `2 phi`, and `3 phi` must be within `0.02` or receive a documented correctness explanation.
- SubOpt must implement the paper's ranked ground-truth graph and
  GreedyHS1000. Its exact-budget curve is recorded even when prototype tie or
  loop behavior prevents reproducing the paper's reported `0.98` at `phi`.
  Identical seeds must produce identical non-timing output.
- Camera's 9.84-million-edge graph must avoid quadratic expansion and stay below a 1.5-GB peak-RSS target.

## Assumptions

- The library consumes a precomputed similarity graph; blocking, similarity generation, and noisy-oracle reconciliation remain outside V1.
- Non-LLM scope is Table III, Table IV, and Figure 4. Figures 1 and 5 are deferred.
- BatchER and LLM-CER inform future oracle behavior, not selectable V1 schedulers.
- Oracle answers are internally consistent partitions. Transport and malformed-output failures are handled; probabilistic contradiction repair is deferred.

## Implementation status

The core, wrapper, oracle, experiment runner, compatibility layer, packaging,
and sanitizer coverage are implemented. Table III matches exactly. The Figure 4
runner emits all 32 maintained-C curves and Table IV emits all 10 configurations,
but those values are not an exact paper reproduction: corrected scheduler
behavior, community backends, deterministic ties, and unresolved residual gaps
are documented in `docs/research-status.md` and `docs/deviations.md`.

# Minimal documentation and GitHub Pages release

## Documentation deliverables

- Replace the repository README with a concise library-user entry point:
  install, local five-minute example, CI/docs badges, paper link, and deep links.
- Publish MkDocs Material documentation for getting started, core concepts,
  Python guide/reference, handwritten C guide/reference, oracle integration,
  experiments, research status, architecture, and development.
- Render the Python reference from complete public docstrings with
  `mkdocstrings`; keep the C reference explicit around `include/perbacco.h`.
- Keep canonical Python and C examples as executable files under `examples/`
  and include those files into documentation pages.
- Label Table III exact and Figure 4 “maintained C implementation results.”
  Document NetworkX-versus-C Louvain, stable ties, exact budgets, corrected
  invalidation/representatives, SubOpt changes, cache provenance, and unresolved
  large curve gaps.
- Link the paper through https://arxiv.org/abs/2606.24407; never publish local
  paper PDFs.

## Site and deployment deliverables

- Configure `site_url: https://stravanni.github.io/pERbacco/`, repository/edit
  links, strict navigation/link validation, search, code-copy controls, heading
  anchors, and light/dark palettes.
- Use an off-white/charcoal interface with one restrained red accent, system
  fonts, short line lengths, visible keyboard focus, and a single-column mobile
  fallback. Keep the landing page immediately useful and plot-free.
- Generate and validate one matching social-preview card after copy and palette
  are final; omit it if one retry cannot produce correct text.
- Add a Pages workflow that builds without deployment on pull requests and uses
  `configure-pages@v5`, `upload-pages-artifact@v4`, and `deploy-pages@v4` on
  `main` with the `github-pages` environment and required permissions.
- Ship library, package, documentation, and workflow through one
  `codex/library-docs-pages` pull request. Never commit papers, renders,
  experiment artifacts, build output, or credentials.

## Acceptance

- Execute the full C/Python suite, local-oracle quickstart, custom fake oracle,
  snapshot/restore example, and compiled C quickstart with live networking
  blocked for documentation examples.
- Pass strict C compilation, ASan/UBSan, Ruff check/format, whitespace checks,
  package build/fresh wheel install, and `mkdocs build --strict` locally and in
  CI as applicable.
- Inspect generated canonical URLs, titles, metadata, code, navigation,
  responsive/focus rules, alt text, and secret absence.
- After review and merge, enable GitHub Actions as the Pages source if needed,
  wait for deployment, and verify the public home, quickstart, Python reference,
  C reference, and research-status routes.

## Documentation release status

Implemented and locally verified on 2026-08-16. Post-merge Pages enablement and
public-URL verification remain intentionally pending review and merge.
