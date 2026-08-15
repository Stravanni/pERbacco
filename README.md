# pERbacco

[![CI](https://github.com/Stravanni/pERbacco/actions/workflows/ci.yml/badge.svg)](https://github.com/Stravanni/pERbacco/actions/workflows/ci.yml)
[![Docs](https://github.com/Stravanni/pERbacco/actions/workflows/docs.yml/badge.svg)](https://stravanni.github.io/pERbacco/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-9e2a2b)](LICENSE)

pERbacco chooses the next bounded batch of records to send to an entity-resolution
oracle. The maintained implementation is a dependency-free C99 engine with a
zero-runtime-dependency Python wrapper and four selectable schedulers: pERbacco,
pERbac, Online, and the ground-truth-only SubOpt baseline.

## Install

A C99 compiler and Python 3.10+ are required. From a checkout:

```bash
python -m pip install .
```

Experiment-only dependencies remain optional:

```bash
python -m pip install '.[data,plot,leiden]'
```

## Five-minute example

The executable quickstart builds a similarity graph, runs pERbacco with a local
ground-truth oracle, and reports recall without making a network request:

```bash
python examples/python/quickstart.py
```

```python
from perbacco import Engine, EngineConfig, Graph, GroundTruthOracle, run

truth = {"a": "alice", "a-2": "alice", "b": "bob"}
graph = Graph.from_edges([("a", "a-2", 0.96), ("a", "b", 0.08)])

with Engine(graph, EngineConfig(batch_size=3)) as engine:
    result = run(engine, GroundTruthOracle(truth), total_truth_matches=1)

print(result.stats.query_count, result.events[-1].recall)
```

See the [installation and first-run guide](https://stravanni.github.io/pERbacco/getting-started/),
[Python reference](https://stravanni.github.io/pERbacco/reference/python/), and
[C reference](https://stravanni.github.io/pERbacco/reference/c/). The public C
header is [`include/perbacco.h`](include/perbacco.h).

## Research status

The project accompanies [*Entity Resolution via Batched Oracle
Queries*](https://arxiv.org/abs/2606.24407). Table III is reproduced exactly.
The current Figure 4 artifacts are **maintained C implementation results**, not
an exact reproduction of the published Python/NetworkX curves. The measured
differences and corrected prototype behaviors are documented in the
[research-status page](https://stravanni.github.io/pERbacco/research-status/).

The original Python scripts remain as research-history references. New library
work targets the C engine and `python/perbacco` package. Licensed under
[BSD-3-Clause](LICENSE).
