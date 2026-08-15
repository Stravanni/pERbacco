from __future__ import annotations

import ctypes
import math
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum

from . import _native


class Status(IntEnum):
    """Status codes returned by the native engine.

    Positive values describe normal terminal conditions, zero is success, and
    negative values are failures. Python methods raise `PerbaccoError`
    for negative statuses.
    """

    OK = 0
    DONE = 1
    BUDGET_EXHAUSTED = 2
    INVALID = -1
    NO_MEMORY = -2
    STATE = -3
    CONFLICT = -4
    OVERFLOW = -5
    VERSION = -6
    INTERNAL = -7


class Method(IntEnum):
    """Available next-batch schedulers.

    ``PERBACCO`` adds a community-detection phase to pERbac, ``PERBAC`` uses
    mean expected benefit, ``ONLINE`` uses the current maximum-benefit policy,
    and ``SUBOPT`` is the ground-truth-only upper-bound baseline.
    """

    PERBACCO = 0
    PERBAC = 1
    ONLINE = 2
    SUBOPT = 3


class CDA(IntEnum):
    """Community-detection backend used by `Engine`.

    ``NONE`` skips community batches, ``LOUVAIN`` uses the built-in deterministic
    C implementation, and ``EXTERNAL`` consumes communities supplied by the
    caller.
    """

    NONE = 0
    LOUVAIN = 1
    EXTERNAL = 2


class BatchKind(IntEnum):
    """Phase that produced a batch."""

    COMMUNITY = 0
    CURRENT = 1


class PerbaccoError(RuntimeError):
    """Native-engine failure.

    Attributes:
        status: Machine-readable `Status` associated with the failure.
    """

    def __init__(self, status: Status, message: str):
        """Initialize an error from a native status and diagnostic message."""
        super().__init__(message)
        self.status = status


def _sort_key(value: Hashable) -> tuple[str, str]:
    return (f"{type(value).__module__}.{type(value).__qualname__}", repr(value))


@dataclass(frozen=True)
class Graph:
    """Immutable similarity graph passed to the scheduling engine.

    Attributes:
        ids: External record identifiers in deterministic dense-ID order.
        edges: Undirected ``(u, v, weight)`` edges using dense integer IDs.
            The C engine borrows this sequence for the engine lifetime.
        records: Optional record attributes keyed by external record ID. These
            values are sent to an oracle by `perbacco.run`; the scheduler
            does not inspect them.
    """

    ids: tuple[Hashable, ...]
    edges: Sequence[tuple[int, int, float]]
    records: Mapping[Hashable, Mapping[str, object]]

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[tuple[Hashable, Hashable, float]],
        *,
        nodes: Iterable[Hashable] = (),
        records: Mapping[Hashable, Mapping[str, object]] | None = None,
    ) -> Graph:
        """Build a validated graph from external record identifiers.

        Args:
            edges: Undirected ``(left, right, weight)`` triples. Weights must be
                finite and non-negative; the C core normalizes them to ``[0, 1]``.
            nodes: Additional isolated record identifiers.
            records: Optional record attributes. Their keys also become nodes.

        Returns:
            A graph with stable dense IDs and sorted, deduplicated edges.

        Raises:
            ValueError: If the graph is empty or contains a self edge, duplicate
                edge, non-finite weight, or negative weight.
            OverflowError: If the graph exceeds the 32-bit C record-ID space.
        """
        raw_edges = list(edges)
        node_set = set(nodes)
        for left, right, _weight in raw_edges:
            node_set.add(left)
            node_set.add(right)
        if records:
            node_set.update(records)
        ids = tuple(sorted(node_set, key=_sort_key))
        if not ids:
            raise ValueError("a graph must contain at least one record")
        if len(ids) > 0xFFFFFFFF:
            raise OverflowError("the C ABI supports at most 2^32-1 records")
        dense = {record_id: index for index, record_id in enumerate(ids)}
        normalized: list[tuple[int, int, float]] = []
        seen: set[tuple[int, int]] = set()
        for external_left, external_right, raw_weight in raw_edges:
            left = dense[external_left]
            right = dense[external_right]
            weight = float(raw_weight)
            if left == right:
                raise ValueError(f"self edge for {external_left!r}")
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError("edge weights must be finite and non-negative")
            if right < left:
                left, right = right, left
            if (left, right) in seen:
                raise ValueError(f"duplicate edge between {external_left!r} and {external_right!r}")
            seen.add((left, right))
            normalized.append((left, right, weight))
        normalized.sort()
        return cls(ids, tuple(normalized), records or {})

    @classmethod
    def from_native_edges(
        cls,
        ids: Sequence[Hashable],
        edges: ctypes.Array[_native.CEdge],
        *,
        records: Mapping[Hashable, Mapping[str, object]] | None = None,
    ) -> Graph:
        """Wrap an existing contiguous C edge array without copying it.

        This lower-level constructor is intended for large dataset loaders. The
        caller must keep ``edges`` alive and unchanged for every engine using the
        returned graph, and is responsible for supplying valid dense endpoints.

        Args:
            ids: External IDs corresponding to dense positions in ``edges``.
            edges: A contiguous array of the wrapper's native edge structure.
            records: Optional record attributes keyed by external ID.

        Returns:
            A graph borrowing the supplied edge storage.
        """
        return cls(tuple(ids), NativeEdges(edges), records or {})

    @property
    def id_to_dense(self) -> dict[Hashable, int]:
        """Return a new mapping from external IDs to dense C record IDs."""
        return {record_id: index for index, record_id in enumerate(self.ids)}


class NativeEdges(Sequence[tuple[int, int, float]]):
    """Compact edge sequence backed by the exact public C structure."""

    def __init__(self, values: ctypes.Array[_native.CEdge]):
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        edge = self.values[index]
        return int(edge.u), int(edge.v), float(edge.weight)

    def __iter__(self) -> Iterator[tuple[int, int, float]]:
        for edge in self.values:
            yield int(edge.u), int(edge.v), float(edge.weight)


@dataclass(frozen=True)
class EngineConfig:
    """Scheduling and community-detection configuration.

    Attributes:
        method: Scheduler to run. ``SUBOPT`` requires ground-truth labels.
        cda: Community-detection backend. Use ``EXTERNAL`` with the
            ``communities`` argument to `Engine`.
        batch_size: Maximum number of current entity representatives per query;
            must be at least two.
        top_k: GreedyHS candidate limit. The paper experiments use ``1000``.
        seed: Seed for the engine-local deterministic pseudorandom generator.
        max_queries: Exact oracle-query limit. Zero means unlimited.
        lambda_w: Minimum normalized internal density for a heavy community.
        louvain_resolution: Louvain resolution; zero selects ``0.75`` up to
            20,000 records and ``0.50`` above that threshold.
        louvain_threshold: Minimum Louvain improvement before stopping.
    """

    method: Method = Method.PERBACCO
    cda: CDA = CDA.LOUVAIN
    batch_size: int = 10
    top_k: int = 1000
    seed: int = 42
    max_queries: int = 0
    lambda_w: float = 0.05
    louvain_resolution: float = 0.0
    louvain_threshold: float = 1e-4


@dataclass(frozen=True)
class Batch:
    """One outstanding oracle query selected by the engine.

    Attributes:
        kind: Phase that selected the batch.
        records: External IDs of the current entity representatives.
        representatives: Dense representative IDs aligned with ``records``.
            These are accepted by `Engine.group_members`.
    """

    kind: BatchKind
    records: tuple[Hashable, ...]
    representatives: tuple[int, ...]


@dataclass(frozen=True)
class Statistics:
    """Read-only engine counters at one point in a run.

    Attributes:
        query_count: Number of submitted oracle partitions.
        discovered_matches: Cumulative matching record pairs implied by accepted
            component merges. Oracle answers are authoritative; optional truth
            labels add conflict validation but are not required for this count.
        candidate_edges: Current cross-component similarity edges.
        community_batches: Submitted batches selected during the community phase.
        current_batches: Submitted batches selected from current components.
        community_records: Record count covered by accepted heavy communities.
        temperature: Current pERbacco temperature state.
        finished: Whether no further schedulable batch remains. This is false
            when iteration stopped only because ``max_queries`` was reached.
    """

    query_count: int
    discovered_matches: int
    candidate_edges: int
    community_batches: int
    current_batches: int
    community_records: int
    temperature: float
    finished: bool


def _message(handle: _native.EnginePointer, status: int) -> str:
    detail = _native.lib.pb_engine_last_error(handle)
    if detail:
        decoded = detail.decode("utf-8", errors="replace")
        if decoded:
            return decoded
    generic = _native.lib.pb_strerror(status)
    return generic.decode("utf-8", errors="replace") if generic else f"status {status}"


def _check(handle: _native.EnginePointer, status: int) -> None:
    if status != Status.OK:
        raise PerbaccoError(Status(status), _message(handle, status))


class Engine:
    """Stateful native next-batch scheduler.

    An engine permits exactly one outstanding batch: call `next_batch`,
    obtain a complete oracle partition, and call `submit_partition`
    before requesting another batch. Use the engine as a context manager or call
    `close` to release native memory. Methods other than ``close`` must not
    be called after closure.
    """

    def __init__(
        self,
        graph: Graph,
        config: EngineConfig | None = None,
        *,
        communities: Sequence[Sequence[Hashable]] | None = None,
        truth: Mapping[Hashable, Hashable] | None = None,
    ) -> None:
        """Create an engine borrowing the graph's immutable edge storage.

        Args:
            graph: Similarity graph whose edge storage remains alive with this
                engine through the wrapper's reference.
            config: Engine configuration. Defaults to `EngineConfig`.
            communities: Disjoint final heavy communities expressed as external
                record IDs. Required when ``config.cda`` is ``EXTERNAL``.
            truth: Optional ground-truth entity label for every graph record.
                Required by ``SUBOPT`` and needed for match/recall statistics.

        Raises:
            ValueError: If Python-side truth or community inputs are incomplete.
            KeyError: If a community contains an unknown record ID.
            PerbaccoError: If the native constructor rejects the configuration,
                graph, communities, or truth labels.
        """
        config = config or EngineConfig()
        self.graph = graph
        self.config = config
        self._handle = _native.EnginePointer()
        self._edge_array = self._make_edges(graph)
        self._config = self._make_config(config)
        self._community_storage = self._make_communities(graph, communities)
        self._truth_array = self._make_truth(graph, truth)
        self._create(False, None)

    @staticmethod
    def _make_edges(graph: Graph) -> ctypes.Array[_native.CEdge]:
        if isinstance(graph.edges, NativeEdges):
            return graph.edges.values
        edge_array_type = _native.CEdge * len(graph.edges)
        return edge_array_type(*(_native.CEdge(*edge) for edge in graph.edges))

    @staticmethod
    def _make_config(config: EngineConfig) -> _native.CConfig:
        native = _native.CConfig()
        _native.lib.pb_config_init(ctypes.byref(native))
        native.method = int(config.method)
        native.cda = int(config.cda)
        native.batch_size = config.batch_size
        native.top_k = config.top_k
        native.seed = config.seed
        native.max_queries = config.max_queries
        native.lambda_w = config.lambda_w
        native.louvain_resolution = config.louvain_resolution
        native.louvain_threshold = config.louvain_threshold
        return native

    @staticmethod
    def _make_communities(
        graph: Graph, communities: Sequence[Sequence[Hashable]] | None
    ) -> tuple[_native.CCommunities, object, object] | None:
        if communities is None:
            return None
        dense = graph.id_to_dense
        flat: list[int] = []
        offsets = [0]
        for community in communities:
            flat.extend(dense[item] for item in community)
            offsets.append(len(flat))
        offset_array = (ctypes.c_size_t * len(offsets))(*offsets)
        node_array = (ctypes.c_uint32 * len(flat))(*flat)
        native = _native.CCommunities(len(communities), offset_array, node_array)
        return native, offset_array, node_array

    @staticmethod
    def _make_truth(
        graph: Graph, truth: Mapping[Hashable, Hashable] | None
    ) -> ctypes.Array[ctypes.c_uint32] | None:
        if truth is None:
            return None
        missing = [record_id for record_id in graph.ids if record_id not in truth]
        if missing:
            raise ValueError(f"truth labels are missing {len(missing)} graph records")
        labels: dict[Hashable, int] = {}
        encoded: list[int] = []
        for record_id in graph.ids:
            label = truth[record_id]
            if label not in labels:
                labels[label] = len(labels)
            encoded.append(labels[label])
        return (ctypes.c_uint32 * len(encoded))(*encoded)

    def _create(self, restore: bool, snapshot: bytes | None) -> None:
        communities_pointer = None
        if self._community_storage is not None:
            communities_pointer = ctypes.byref(self._community_storage[0])
        truth_pointer = self._truth_array
        edge_pointer = self._edge_array if len(self.graph.edges) else None
        if restore:
            assert snapshot is not None
            snapshot_buffer = ctypes.create_string_buffer(snapshot)
            status = _native.lib.pb_engine_restore(
                len(self.graph.ids),
                edge_pointer,
                len(self.graph.edges),
                ctypes.byref(self._config),
                communities_pointer,
                truth_pointer,
                snapshot_buffer,
                len(snapshot),
                ctypes.byref(self._handle),
            )
        else:
            status = _native.lib.pb_engine_create_borrowed(
                len(self.graph.ids),
                edge_pointer,
                len(self.graph.edges),
                ctypes.byref(self._config),
                communities_pointer,
                truth_pointer,
                ctypes.byref(self._handle),
            )
        _check(self._handle, status)

    @classmethod
    def restore(
        cls,
        graph: Graph,
        snapshot: bytes,
        config: EngineConfig | None = None,
        *,
        communities: Sequence[Sequence[Hashable]] | None = None,
        truth: Mapping[Hashable, Hashable] | None = None,
    ) -> Engine:
        """Restore an engine from a deterministic query-journal snapshot.

        Immutable inputs and configuration must exactly match those used to make
        the snapshot. Snapshots do not contain graph edges, communities, or truth
        labels and can only be taken when no batch awaits submission.

        Args:
            graph: Original immutable similarity graph.
            snapshot: Bytes returned by `snapshot`.
            config: Original engine configuration.
            communities: Original external communities, if any.
            truth: Original truth mapping, if any.

        Returns:
            A new engine replayed through the last submitted query.

        Raises:
            PerbaccoError: If the snapshot version, checksum, journal, or supplied
                immutable inputs are incompatible.
        """
        config = config or EngineConfig()
        self = cls.__new__(cls)
        self.graph = graph
        self.config = config
        self._handle = _native.EnginePointer()
        self._edge_array = self._make_edges(graph)
        self._config = self._make_config(config)
        self._community_storage = self._make_communities(graph, communities)
        self._truth_array = self._make_truth(graph, truth)
        self._create(True, snapshot)
        return self

    def next_batch(self) -> Batch | None:
        """Select the next oracle batch.

        Returns:
            The next batch, or ``None`` when the scheduler is finished or the
            exact query budget is exhausted. Inspect `stats.finished` to
            distinguish those terminal conditions.

        Raises:
            PerbaccoError: If a previous batch is still outstanding or the engine
                is closed or invalid.
        """
        buffer = (ctypes.c_uint32 * self.config.batch_size)()
        native_batch = _native.CBatch()
        status = _native.lib.pb_engine_next_batch(
            self._handle, buffer, self.config.batch_size, ctypes.byref(native_batch)
        )
        if status == Status.DONE:
            return None
        if status == Status.BUDGET_EXHAUSTED:
            return None
        _check(self._handle, status)
        representatives = tuple(int(buffer[index]) for index in range(native_batch.len))
        return Batch(
            BatchKind(native_batch.kind),
            tuple(self.graph.ids[index] for index in representatives),
            representatives,
        )

    def submit_partition(self, labels: Sequence[Hashable]) -> None:
        """Atomically apply a complete partition of the outstanding batch.

        ``labels`` is aligned with ``Batch.records``. Values are arbitrary
        hashable cluster labels; equality is the only operation performed. The
        native engine validates the whole answer before changing state.

        Args:
            labels: One cluster label per entity in the outstanding batch.

        Raises:
            PerbaccoError: If no batch is outstanding, the label count is wrong,
                or the partition conflicts with known state or truth constraints.
        """
        encoded_by_label: dict[Hashable, int] = {}
        encoded: list[int] = []
        for label in labels:
            if label not in encoded_by_label:
                encoded_by_label[label] = len(encoded_by_label)
            encoded.append(encoded_by_label[label])
        array = (ctypes.c_uint32 * len(encoded))(*encoded)
        status = _native.lib.pb_engine_submit_partition(self._handle, array, len(encoded))
        _check(self._handle, status)

    def group_members(self, representative: int) -> tuple[Hashable, ...]:
        """Return external record IDs in a current entity component.

        Args:
            representative: Dense representative from ``Batch.representatives``.

        Returns:
            Current component members in deterministic dense-ID order.

        Raises:
            PerbaccoError: If the representative or engine state is invalid.
        """
        needed = ctypes.c_size_t()
        status = _native.lib.pb_engine_group_members(
            self._handle, representative, None, 0, ctypes.byref(needed)
        )
        if status not in (Status.OK, Status.OVERFLOW):
            _check(self._handle, status)
        output = (ctypes.c_uint32 * needed.value)()
        status = _native.lib.pb_engine_group_members(
            self._handle, representative, output, needed.value, ctypes.byref(needed)
        )
        _check(self._handle, status)
        return tuple(self.graph.ids[output[index]] for index in range(needed.value))

    @property
    def stats(self) -> Statistics:
        """Return a consistent copy of the current engine statistics."""
        native = _native.CStats()
        _check(self._handle, _native.lib.pb_engine_get_stats(self._handle, ctypes.byref(native)))
        return Statistics(
            native.query_count,
            native.discovered_matches,
            native.candidate_edges,
            native.community_batches,
            native.current_batches,
            native.community_records,
            native.temperature,
            bool(native.finished),
        )

    def snapshot(self) -> bytes:
        """Serialize the deterministic submitted-query journal.

        Returns:
            Versioned bytes suitable for `restore` with the same immutable
            inputs and configuration.

        Raises:
            PerbaccoError: If a batch is awaiting submission or the engine is
                closed or invalid.
        """
        size = ctypes.c_size_t()
        _check(
            self._handle,
            _native.lib.pb_engine_snapshot_size(self._handle, ctypes.byref(size)),
        )
        output = ctypes.create_string_buffer(size.value)
        written = ctypes.c_size_t()
        _check(
            self._handle,
            _native.lib.pb_engine_snapshot(self._handle, output, size.value, ctypes.byref(written)),
        )
        return output.raw[: written.value]

    def close(self) -> None:
        """Release native state; repeated calls are safe."""
        if getattr(self, "_handle", None):
            _native.lib.pb_engine_free(self._handle)
            self._handle = _native.EnginePointer()

    def __enter__(self) -> Engine:  # noqa: PYI034 - Python 3.10 has no typing.Self
        """Return this engine for context-manager use."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Release native state when leaving a context manager."""
        self.close()

    def __del__(self) -> None:
        self.close()
