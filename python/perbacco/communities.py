from __future__ import annotations

import os
import random
import tempfile
from collections import deque
from pathlib import Path
from typing import Literal

from .core import Graph


def external_heavy_communities(
    graph: Graph,
    *,
    algorithm: Literal["louvain", "leiden"],
    lambda_w: float,
    batch_size: int,
    seed: int = 42,
) -> list[list[object]]:
    """Compute Algorithm 1 communities using optional igraph/Leiden dependencies."""

    cache = Path(tempfile.gettempdir()) / "perbacco-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    try:
        import igraph as ig
    except ImportError as exc:
        raise RuntimeError("external communities need the `perbacco[leiden]` extra") from exc
    if algorithm == "leiden":
        try:
            import leidenalg
        except ImportError as exc:
            raise RuntimeError("Leiden communities need the `perbacco[leiden]` extra") from exc
    ig.set_random_number_generator(random.Random(seed))

    endpoints = [(left, right) for left, right, _weight in graph.edges]
    maximum = max((weight for _left, _right, weight in graph.edges), default=0.0)
    weights = [weight / maximum if maximum else 0.0 for _left, _right, weight in graph.edges]
    native = ig.Graph(n=len(graph.ids), edges=endpoints, directed=False)
    native.es["weight"] = weights
    minimum_size = max(batch_size, 10)
    pending: deque[list[int]] = deque([list(range(len(graph.ids)))])
    heavy: list[list[int]] = []

    while pending:
        nodes = pending.popleft()
        if len(nodes) < minimum_size:
            continue
        subgraph = native.induced_subgraph(nodes)
        if algorithm == "louvain":
            partition = subgraph.community_multilevel(weights="weight", return_levels=False)
        else:
            partition = leidenalg.find_partition(
                subgraph,
                leidenalg.ModularityVertexPartition,
                weights="weight",
                seed=seed,
            )
        parts = [[nodes[local] for local in part] for part in partition]
        if len(parts) <= 1:
            continue
        for part in parts:
            if len(part) < minimum_size:
                continue
            induced = native.induced_subgraph(part)
            weight = sum(float(value) for value in induced.es["weight"])
            density = weight / (len(part) * (len(part) - 1) / 2)
            if density >= lambda_w:
                heavy.append(part)
            else:
                pending.append(part)

    heavy.sort(
        key=lambda part: (
            -sum(float(value) for value in native.induced_subgraph(part).es["weight"]),
            min(part),
        )
    )
    return [[graph.ids[index] for index in part] for part in heavy]
