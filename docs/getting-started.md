# Installation and first run

The installed Python package has no third-party runtime dependencies. It builds
and ships a small platform-native library loaded through `ctypes`.

## Requirements

- Python 3.10 or newer
- a C99 compiler (`cc`, Clang, or GCC)
- `make`

Clone the repository and install the package:

```bash
git clone https://github.com/Stravanni/pERbacco.git
cd pERbacco
python -m pip install .
```

For C-only use, build the static and shared libraries directly:

```bash
make
```

This creates `build/libperbacco.a` and either `build/libperbacco.so` on Linux or
`build/libperbacco.dylib` on macOS.

## Run the quickstart

```bash
python examples/python/quickstart.py
```

The complete source is included here from the executable file used by the test
suite:

```python
--8<-- "examples/python/quickstart.py"
```

The example has four records, two true entities, and a weighted similarity
graph. `GroundTruthOracle` supplies a local complete partition for each selected
batch. `run()` expands representatives to current component members, calls the
oracle, submits the answer, and records per-query events.

## Build a graph

`Graph.from_edges()` accepts arbitrary hashable external IDs and converts them
to deterministic dense 32-bit IDs for C:

```python
from perbacco import Graph

graph = Graph.from_edges(
    [
        ("record-17", "record-42", 0.91),
        ("record-17", "record-99", 0.12),
    ],
    nodes=["isolated-record"],
    records={
        "record-17": {"name": "Ada Lovelace"},
        "record-42": {"name": "A. Lovelace"},
    },
)
```

Edges are undirected. Self edges, duplicate pairs, negative weights, and
non-finite weights are rejected. The C core scales valid weights to `[0, 1]`.
Record dictionaries are opaque to the scheduler; they are carried only for an
oracle.

## Understand the engine loop

`run()` is the usual interface. The equivalent explicit loop is useful when an
application already has its own queue or checkpoint system:

```python
with Engine(graph, EngineConfig(batch_size=10)) as engine:
    while (batch := engine.next_batch()) is not None:
        labels = my_oracle_partition(batch.records)
        engine.submit_partition(labels)
```

There may be only one outstanding batch. A second `next_batch()` before
`submit_partition()` raises `PerbaccoError` with `Status.STATE`. `None` means
either that the graph is finished or the exact query budget is exhausted;
inspect `engine.stats.finished` to distinguish them.

## Select a method

```python
from perbacco import CDA, EngineConfig, Method

config = EngineConfig(
    method=Method.PERBACCO,
    cda=CDA.LOUVAIN,
    batch_size=10,
    top_k=1000,
    seed=42,
    max_queries=300,
)
```

| Method | Similarity graph | Community phase | Ground truth needed to schedule |
| --- | --- | --- | --- |
| `Method.PERBACCO` | yes | optional Louvain/external | no |
| `Method.PERBAC` | yes | no | no |
| `Method.ONLINE` | yes | no | no |
| `Method.SUBOPT` | no | no | **yes** |

Use `CDA.NONE` to disable communities, `CDA.LOUVAIN` for the built-in C backend,
or `CDA.EXTERNAL` with explicit disjoint heavy communities. Continue with the
[Python guide](python-guide.md) or read the [C guide](c-guide.md).
