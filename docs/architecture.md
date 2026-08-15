# Architecture

## Core

`src/perbacco.c` is a single C99 translation unit with no runtime dependency
beyond the C standard library and `libm`. `include/perbacco.h` exposes an opaque
`pb_engine`, versioned configuration, batches, statistics, status codes, and
snapshot operations.

The immutable graph is a sorted compact edge array. The Python wrapper gives
the engine a borrowed mutable array so normalization and sorting do not require
a second 16-byte copy of every edge. The caller keeps that memory alive for the
engine lifetime. Candidate state uses union-find components, linked component
members, open-addressed maps for candidate/non-match pairs, adjacency vectors,
and an indexed max heap.

Oracle submission is validated before mutation. Known matches contract
components, known non-matches propagate to new representatives, and every
incident candidate whose benefit may change is re-evaluated. The engine never
trusts stale record representatives.

## Schedulers

- `PB_METHOD_PERBACCO`: mean benefit, recursive heavy communities, and the
  temperature policy from the paper.
- `PB_METHOD_PERBAC`: mean benefit without community preprocessing.
- `PB_METHOD_ONLINE`: maximum-edge-probability benefit.
- `PB_METHOD_SUBOPT`: ground-truth-only comparison. It directly reduces truth
  components and packs residual groups with a capacity-`b` dynamic program;
  similarity edges are neither loaded nor materialized.

GreedyHS considers the best 1,000 live benefits by default. Ties are explicit,
and the engine uses its own seeded PRNG so host-language hash iteration cannot
change a run.

The built-in CDA is a dependency-free deterministic weighted Louvain
implementation. It performs local modularity moves, contracts the resulting
communities into a weighted graph (including self-loops), and repeats coarse
levels until the configured modularity threshold is reached. The ABI also
accepts explicit disjoint communities, allowing the optional experiment layer
to use seeded igraph Louvain or Leiden while keeping all batch scheduling and
entity state in C.

## Python boundary

`python/perbacco/_native.py` declares the ABI with `ctypes`; there is no compiled
Python extension and no NumPy requirement. `core.py` owns external-ID mapping,
native memory lifetimes, snapshots, and ergonomic dataclasses. `runner.py`
adapts selected representatives into complete entity views for any object that
implements the oracle protocol.

The provider-neutral oracle in `oracle.py` supports:

- local ground-truth partitions;
- OpenAI Responses structured output;
- OpenAI-compatible Chat Completions structured output, including OpenRouter;
- complete-partition validation, exponential retry, token accounting, and
  append-only JSONL request/response journals.

## Experiment isolation

Parquet and community dependencies are imported only from `data.py` and
`communities.py`. Each large run executes in a child process, which gives an
independent peak-RSS measurement and releases graph memory on exit. Run IDs
encode dataset, precision, method, community backend, lambda, and exact query
budget. A run is reused only when its summary configuration matches and the
curve SHA-256 matches its manifest.

Maintained Figure 4 uses the built-in C Louvain path. Table IV additionally exercises the
seeded external Louvain/Leiden hook to compare CDA behavior. SubOpt uses the
paper's hidden ground-truth benefit graph, including its tiny size-rank factor,
and applies GreedyHS1000. A row-cursor max-heap emits only the globally heaviest
1,000 current truth edges, so the complete clique graph is never materialized.
Its truth-only loader also avoids Camera's 9.84-million-edge similarity graph.

Snapshots store a checksummed, versioned query journal instead of copying the
immutable graph. Restore validates ABI, configuration, graph, truth, and
community identity hashes, then replays accepted partitions to reconstruct
exactly the next selection state.

Experiment run manifests separately fingerprint the loaded native library,
Python modules that affect numeric results, Python runtime, and optional
graph/community dependency versions. Cache reuse requires that fingerprint,
the full run configuration, and the curve checksum to match. This is
deliberately stricter than snapshot compatibility: an experiment cache is a
claim about which implementation produced a result.
