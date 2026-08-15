# Python guide

The Python package maps external record IDs to the stable C ABI, owns native
memory, adapts batches into oracle views, and exposes immutable dataclasses for
configuration and results. It depends only on the standard library.

## Graph and record data

Use `Graph.from_edges()` for ordinary in-memory input. `Graph.ids` records the
deterministic external-ID order; `Graph.edges` uses dense positions in that
order. `Graph.records` is optional and may contain any JSON-like or application
objects your oracle can serialize.

`Graph.from_native_edges()` is a lower-level zero-copy path used by the Parquet
loader. Its contiguous native array must remain alive and unchanged for every
engine that borrows it.

## Configure an engine

`EngineConfig` is frozen and reusable:

```python
from perbacco import CDA, EngineConfig, Method

config = EngineConfig(
    method=Method.PERBACCO,
    cda=CDA.LOUVAIN,
    batch_size=10,
    top_k=1000,
    seed=42,
    max_queries=0,
    lambda_w=0.05,
    louvain_resolution=0.0,
    louvain_threshold=1e-4,
)
```

`max_queries=0` means unlimited. `louvain_resolution=0.0` asks the C engine for
the paper defaults: `0.75` up to 20,000 records and `0.50` above that threshold.

When `cda=CDA.EXTERNAL`, pass disjoint final heavy communities as external IDs:

```python
with Engine(graph, config, communities=[["a", "b", "c"], ["x", "y"]]) as engine:
    ...
```

Records omitted from external communities remain available in the residual
phase. Community storage, graph edges, and optional truth arrays are retained by
the wrapper for the engine lifetime.

## Batches and components

`Batch.records` contains external representative IDs for the oracle.
`Batch.representatives` contains dense IDs accepted by `group_members()`:

```python
batch = engine.next_batch()
if batch is not None:
    components = [engine.group_members(rep) for rep in batch.representatives]
```

Submit labels in exactly the same order as `batch.records`. The answer is
transactional: invalid lengths, impossible state transitions, and conflicts do
not partially mutate the engine.

## Runner and custom oracle

Any object with `partition(entities) -> OracleResult` satisfies the `Oracle`
protocol. This canonical fake oracle proves that no provider SDK is required:

```python
--8<-- "examples/python/custom_oracle.py"
```

`run()` keeps a complete event trace. If `total_truth_matches` is supplied, each
`RunEvent` includes recall. An `on_event(event, snapshot)` callback can persist a
checkpoint after every accepted answer.

## Statistics and budgets

`engine.stats` is a copy of native counters. `query_count` counts accepted
oracle answers. `discovered_matches` counts record pairs implied by accepted
component merges; oracle answers are authoritative, while optional truth labels
add conflict validation. `finished=False` with `next_batch() is None` means the
exact query budget was reached before the graph terminated.

## Snapshot and restore

Snapshots are deterministic journal bytes. They exclude immutable graph data
and are valid only with no outstanding batch. The source below is executed by
the documentation acceptance suite:

```python
--8<-- "examples/python/snapshot.py"
```

Persist the graph identity, complete `EngineConfig`, external communities, and
truth source alongside a snapshot. Restore validates the ABI version,
configuration, graph/truth/community identity hashes, and journal before replay.

## Errors and lifecycle

Use `Engine` as a context manager. `close()` is idempotent, but no other method
may be called after closure. Native failures raise `PerbaccoError`; its `status`
field is a `Status` enum value suitable for programmatic handling. Input errors
detected before C may raise `ValueError`, `KeyError`, or `OverflowError`.

Oracle transport and partition failures raise `OracleProtocolError`. They are
separate from engine state failures so applications can choose an API retry,
manual review, or checkpoint recovery policy.

See the [complete Python API reference](reference/python.md).
