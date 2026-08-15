from __future__ import annotations

import csv
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from pathlib import Path

from ._native import CEdge
from .core import Graph


def _scalar(value: object) -> Hashable:
    if isinstance(value, (int, float)):
        integer = int(value)
        return integer if integer == value else value
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return text


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[Hashable, Hashable] = {}
        self.size: dict[Hashable, int] = {}

    def add(self, item: Hashable) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1

    def find(self, item: Hashable) -> Hashable:
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            following = self.parent[item]
            self.parent[item] = root
            item = following
        return root

    def union(self, left: Hashable, right: Hashable) -> None:
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


def read_truth(path: str | Path, nodes: Iterable[Hashable] = ()) -> tuple[dict[Hashable, int], int]:
    dsu = _DisjointSet()
    for node in nodes:
        dsu.add(node)
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            left = _scalar(row[0])
            right = _scalar(row[1])
            dsu.union(left, right)
    roots: dict[Hashable, int] = {}
    labels: dict[Hashable, int] = {}
    for node in dsu.parent:
        root = dsu.find(node)
        if root not in roots:
            roots[root] = len(roots)
        labels[node] = roots[root]
    counts: dict[int, int] = {}
    for label in labels.values():
        counts[label] = counts.get(label, 0) + 1
    pair_count = sum(count * (count - 1) // 2 for count in counts.values())
    return labels, pair_count


def read_records(path: str | Path) -> dict[Hashable, dict[str, object]]:
    records: dict[Hashable, dict[str, object]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            return records
        id_column = "id" if "id" in reader.fieldnames else reader.fieldnames[0]
        for row in reader:
            record_id = _scalar(row.pop(id_column, ""))
            records[record_id] = {
                key: value for key, value in row.items() if value not in (None, "")
            }
    return records


@dataclass(frozen=True)
class Dataset:
    name: str
    graph: Graph
    truth: dict[Hashable, int]
    truth_pair_count: int

    @property
    def entity_sizes(self) -> tuple[int, ...]:
        counts: dict[int, int] = {}
        for label in self.truth.values():
            counts[label] = counts.get(label, 0) + 1
        return tuple(sorted(counts.values(), reverse=True))


def _graph_path(root: Path, name: str, precision: str | None) -> Path:
    if name.startswith("synth_"):
        selected = precision or "1.0"
        return root / "similarity_graph" / f"synth_precision_{selected}" / f"{name}.parquet"
    return root / "similarity_graph" / f"{name}.parquet"


def _record_path(root: Path, name: str) -> Path:
    path = root / "datasets" / name / f"{name.removeprefix('synth_')}.csv"
    if not path.exists():
        path = root / "datasets" / name / f"{name}.csv"
    return path


def load_truth_dataset(root: str | Path, name: str) -> Dataset:
    """Load only records and truth for SubOpt or Phi calculations.

    The ground-truth scheduler does not consume similarity edges. Avoiding the
    Parquet graph is particularly important for Camera's 9.84 million edges.
    """

    root_path = Path(root)
    record_path = _record_path(root_path, name)
    records = read_records(record_path) if record_path.exists() else {}
    truth_path = root_path / "datasets" / name / "groundtruth.csv"
    truth, truth_pair_count = read_truth(truth_path, records)
    graph = Graph.from_edges((), nodes=truth, records=records)
    return Dataset(name, graph, truth, truth_pair_count)


def load_dataset(root: str | Path, name: str, *, precision: str | None = None) -> Dataset:
    try:
        from pyarrow import parquet
    except ImportError as exc:
        raise RuntimeError("dataset loading needs the optional `perbacco[data]` extra") from exc

    root_path = Path(root)
    graph_path = _graph_path(root_path, name, precision)
    truth_path = root_path / "datasets" / name / "groundtruth.csv"
    record_path = _record_path(root_path, name)
    records = read_records(record_path) if record_path.exists() else {}
    parquet_file = parquet.ParquetFile(graph_path)
    names = set(parquet_file.schema_arrow.names)
    left_name = "id1" if "id1" in names else "left_spec_id"
    right_name = "id2" if "id2" in names else "right_spec_id"
    weight_name = "w" if "w" in names else "weight"
    if not {left_name, right_name, weight_name}.issubset(names):
        raise ValueError(f"unsupported similarity graph schema: {parquet_file.schema_arrow.names}")
    truth, truth_pair_count = read_truth(truth_path)
    nodes = set(truth) | set(records)
    for batch in parquet_file.iter_batches(columns=[left_name, right_name], batch_size=262_144):
        nodes.update(_scalar(value) for value in batch.column(0).to_pylist())
        nodes.update(_scalar(value) for value in batch.column(1).to_pylist())
    ids = tuple(sorted(nodes, key=lambda value: (type(value).__qualname__, repr(value))))
    dense = {record_id: index for index, record_id in enumerate(ids)}
    edge_array_type = CEdge * parquet_file.metadata.num_rows
    native_edges = edge_array_type()
    cursor = 0
    for batch in parquet_file.iter_batches(
        columns=[left_name, right_name, weight_name], batch_size=262_144
    ):
        left_values = batch.column(0).to_pylist()
        right_values = batch.column(1).to_pylist()
        weights = batch.column(2).to_pylist()
        for left, right, weight in zip(left_values, right_values, weights, strict=True):
            native_edges[cursor].u = dense[_scalar(left)]
            native_edges[cursor].v = dense[_scalar(right)]
            native_edges[cursor].weight = float(weight)
            cursor += 1
    if cursor != parquet_file.metadata.num_rows:
        raise RuntimeError("Parquet metadata row count changed while loading")
    graph = Graph.from_native_edges(ids, native_edges, records=records)
    missing = [node for node in graph.ids if node not in truth]
    if missing:
        next_label = max(truth.values(), default=-1) + 1
        for node in missing:
            truth[node] = next_label
            next_label += 1
    return Dataset(name, graph, truth, truth_pair_count)
