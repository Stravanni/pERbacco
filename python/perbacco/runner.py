from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .core import BatchKind, Engine, Statistics
from .oracle import EntityView, Oracle


@dataclass(frozen=True)
class RunEvent:
    """Metrics emitted after one accepted oracle answer.

    Attributes:
        query: One-based number of submitted oracle queries.
        batch_kind: Engine phase that selected the query.
        batch_size: Number of current entity components sent to the oracle.
        discovered_matches: Cumulative matching record pairs discovered.
        recall: Cumulative recall when ``total_truth_matches`` was supplied,
            otherwise ``None``.
        input_tokens: Input tokens reported for this oracle call.
        output_tokens: Output tokens reported for this oracle call.
    """

    query: int
    batch_kind: BatchKind
    batch_size: int
    discovered_matches: int
    recall: float | None
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class RunResult:
    """Final state and per-query trace from `run`.

    Attributes:
        stats: Final engine statistics.
        events: Immutable sequence of accepted-query events.
    """

    stats: Statistics
    events: tuple[RunEvent, ...]


def run(
    engine: Engine,
    oracle: Oracle,
    *,
    total_truth_matches: int | None = None,
    on_event: Callable[[RunEvent, bytes], None] | None = None,
) -> RunResult:
    """Drive an engine until completion or exact budget exhaustion.

    For every selected representative, this function expands the current entity
    component and attaches record attributes from `perbacco.Graph`. It
    then submits the oracle's complete partition transactionally.

    Args:
        engine: Open engine with no outstanding batch.
        oracle: Synchronous partition oracle.
        total_truth_matches: Optional denominator used to compute event recall.
        on_event: Optional checkpoint callback receiving the event and a fresh
            restorable snapshot after each accepted answer.

    Returns:
        Final statistics and a complete per-query event trace.

    Raises:
        ValueError: If the oracle returns the wrong number of labels.
        PerbaccoError: If the engine rejects a transition or snapshot.
        OracleProtocolError: If the oracle fails or returns an invalid answer.
    """
    events: list[RunEvent] = []
    while (batch := engine.next_batch()) is not None:
        entities = []
        for representative in batch.representatives:
            members = engine.group_members(representative)
            entities.append(
                EntityView(
                    engine.graph.ids[representative],
                    members,
                    tuple(dict(engine.graph.records.get(member, {})) for member in members),
                )
            )
        answer = oracle.partition(entities)
        if len(answer.labels) != len(entities):
            raise ValueError("oracle returned the wrong number of labels")
        engine.submit_partition(answer.labels)
        stats = engine.stats
        recall = None
        if total_truth_matches is not None:
            recall = stats.discovered_matches / total_truth_matches if total_truth_matches else 1.0
        event = RunEvent(
            stats.query_count,
            batch.kind,
            len(batch.records),
            stats.discovered_matches,
            recall,
            int(answer.usage.get("input_tokens", 0)),
            int(answer.usage.get("output_tokens", 0)),
        )
        events.append(event)
        if on_event is not None:
            on_event(event, engine.snapshot())
    return RunResult(engine.stats, tuple(events))
